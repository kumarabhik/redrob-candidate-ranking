"""OFFLINE cross-encoder teacher (design_doc.md §7.4 stretch / explanation.md).

A cross-encoder jointly reads (JD, profile) and scores relevance far more precisely than the
bi-encoder cosine used for recall. It is too slow to run per-candidate at rank time (and the
rank step is network-free), so we PRECOMPUTE scores OFFLINE for the top-N recall candidates
and store them as artifacts/ce_scores.json {candidate_id: score}. rank.py late-fuses these as
an independent signal (a teacher distilled into the final blend). Candidates without a score
fall back to the neutral median, so the pipeline is correct with or without this artifact.

Run (offline, network allowed for model download), CPU-bound:
  python src/cross_encode.py --top 800
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from config import ARTIFACTS
from jd_aspects import build as build_aspects
from parse import stream_candidates
from recall import (aspect_sims, bm25_query_terms, build_bm25, dense_recall_score,
                    load_artifacts, rrf, tokenize)

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def jd_query_text(aspects: dict) -> str:
    role = aspects.get("role_title", "")
    musts = " ".join(a["query"] for a in aspects["must_have"])
    return f"{role}. {musts}"[:512]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=800, help="how many recall candidates to score")
    ap.add_argument("--max-chars", type=int, default=900)
    args = ap.parse_args()

    t0 = time.time()
    A = load_artifacts()
    aspects, qorder = A["aspects"], A["meta"]["query_order"]
    sims = aspect_sims(A["mat"], A["jd_vecs"])
    dense = dense_recall_score(sims, qorder, aspects)

    records = list(stream_candidates())
    corpus = [tokenize(c.profile_text()) for c in records]
    bm25 = build_bm25(corpus)
    bm25_s = np.asarray(bm25.get_scores(bm25_query_terms(aspects)), dtype=np.float32)
    order = rrf([np.argsort(-dense)[:4000], np.argsort(-bm25_s)[:4000]], n=args.top)

    print(f"loading cross-encoder {MODEL} ...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(MODEL)

    query = jd_query_text(aspects)
    pairs = [[query, records[i].profile_text()[:args.max_chars]] for i in order]
    print(f"scoring {len(pairs)} (JD, profile) pairs on CPU...")
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=True)

    out = {records[i].candidate_id: float(s) for i, s in zip(order, scores)}
    (ARTIFACTS / "ce_scores.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote artifacts/ce_scores.json ({len(out)} scores) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    build_aspects()
    main()
