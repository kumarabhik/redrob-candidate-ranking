"""ENTRYPOINT -- the ranking step reproduced at Stage 3 (design_doc.md §6, Stages A-F).

Constraints (agents.md §1): ≤5 min wall-clock, ≤16 GB RAM, CPU only, NO network. All heavy
work (embeddings, JD vectors, trained model) is loaded from artifacts/. BM25 is rebuilt from
the candidates file here (cheap). Produces a spec-valid top-100 CSV with fact-grounded reasons.

Run:
  python src/rank.py --candidates <candidates.jsonl> --out submission.csv
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from config import ARTIFACTS, CANDIDATES_JSONL
from features import feature_names, features_for, vectorize
from honeypot import is_honeypot
from labels import relevance_score
from parse import stream_candidates
from reasoning import reason_for
from recall import (aspect_sims, bm25_query_terms, build_bm25, dense_recall_score,
                    load_artifacts, shortlist, tokenize)

SHORTLIST_PER_LIST = 4000
SHORTLIST_TAKE = 8000
TOP_K = 100
TWIN_THRESHOLD = 0.985   # embedding cosine above which two profiles are treated as twins


def _unit(x: np.ndarray) -> np.ndarray:
    """Min-max scale to [0,1] (constant arrays -> ones)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)


def load_ce_scores(cand_ids: list[str]):
    """Optional cross-encoder teacher scores (artifacts/ce_scores.json: id -> score).
    Returns an array aligned to cand_ids, or None if the artifact is absent.
    Candidates without a precomputed score get the neutral median."""
    import json
    path = ARTIFACTS / "ce_scores.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    vals = [d[i] for i in cand_ids if i in d]
    if not vals:
        return None
    med = float(np.median(vals))
    return np.array([d.get(i, med) for i in cand_ids], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(CANDIDATES_JSONL))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--no-model", action="store_true",
                    help="rank by the rubric score instead of the trained model")
    args = ap.parse_args()

    t0 = time.time()
    A = load_artifacts()
    aspects, query_order = A["aspects"], A["meta"]["query_order"]
    names = feature_names(aspects)
    sims = aspect_sims(A["mat"], A["jd_vecs"])

    print("loading candidates + tokenizing for BM25...")
    records = list(stream_candidates(args.candidates))
    assert [c.candidate_id for c in records] == A["ids"], \
        "candidate order differs from build_index; rebuild artifacts"
    corpus = [tokenize(c.profile_text()) for c in records]

    # --- Stage A: hybrid recall (dense + BM25 -> RRF) --------------------
    dense = dense_recall_score(sims, query_order, aspects)
    bm25 = build_bm25(corpus)
    bm25_score = np.asarray(bm25.get_scores(bm25_query_terms(aspects)), dtype=np.float32)
    sl = shortlist(dense, bm25_score, SHORTLIST_PER_LIST, SHORTLIST_TAKE)
    print(f"  shortlist: {len(sl)}  ({time.time()-t0:.1f}s)")

    # --- Stage B + C: features + honeypot gate ---------------------------
    model = None
    if not args.no_model:
        import xgboost as xgb
        model = xgb.XGBRanker()
        model.load_model(str(ARTIFACTS / "model.bin"))

    keep = []
    keep_idx = []                    # original pool index (for twin de-dup via embeddings)
    X = []
    for i in sl:
        c = records[i]
        if is_honeypot(c):           # Stage C: hard gate
            continue
        f = features_for(c, sims[i], query_order, aspects)
        keep.append((c, f))
        keep_idx.append(int(i))
        X.append(vectorize(f, names))
    X = np.asarray(X, dtype=np.float32)

    # --- Stage D: rerank (ensemble late-fusion) --------------------------
    # Blend the LambdaMART score with the interpretable rubric and, if a cross-encoder
    # teacher artifact exists, its (independent) relevance score. Late fusion needs no
    # retraining and is robust to any single model's quirks.
    rubric = np.array([relevance_score(f, aspects) for _, f in keep], dtype=np.float32)
    model_s = model.predict(X) if model is not None else rubric
    ce = load_ce_scores([c.candidate_id for c, _ in keep])  # None if no artifact

    if args.no_model:
        scores = rubric
    elif ce is not None:
        scores = 0.50 * _unit(model_s) + 0.20 * _unit(rubric) + 0.30 * _unit(ce)
    else:
        scores = 0.60 * _unit(model_s) + 0.40 * _unit(rubric)

    order = sorted(range(len(keep)), key=lambda j: (-float(scores[j]), keep[j][0].candidate_id))

    # Behavioral-twin de-dup: greedily pick best-first, skipping any candidate that is a
    # near-duplicate (embedding cosine > TWIN_THRESHOLD) of an already-selected pick.
    mat = A["mat"]
    top, sel_vecs = [], None
    for j in order:
        v = mat[keep_idx[j]]
        if sel_vecs is not None and float((sel_vecs @ v).max()) > TWIN_THRESHOLD:
            continue
        top.append(j)
        sel_vecs = v[None, :] if sel_vecs is None else np.vstack([sel_vecs, v])
        if len(top) == TOP_K:
            break
    if len(top) < TOP_K:             # fallback: backfill if de-dup over-pruned
        for j in order:
            if j not in top:
                top.append(j)
            if len(top) == TOP_K:
                break

    # Calibrate to a clean, top-heavy, strictly non-increasing (0,1] CSV score.
    raw = np.array([scores[j] for j in top], dtype=float)
    lo, hi = raw.min(), raw.max()
    norm = (raw - lo) / (hi - lo) if hi > lo else np.ones_like(raw)
    norm = 0.06 + 0.93 * np.power(norm, 0.65)  # concave -> spreads the top

    # --- Stage E + F: reasoning + emit -----------------------------------
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, j in enumerate(top, 1):
            c, f = keep[j]
            reason = reason_for(c, f, aspects, rank)
            w.writerow([c.candidate_id, rank, f"{norm[rank-1]:.6f}", reason])

    # self-check: honeypot rate in top 100 (must be < 10%)
    top_ids = {keep[j][0].candidate_id for j in top}
    hp_in_top = sum(1 for c in records if c.candidate_id in top_ids and is_honeypot(c))
    print(f"  honeypot rate in top {TOP_K}: {hp_in_top}/{TOP_K}")
    print(f"done -> {args.out}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
