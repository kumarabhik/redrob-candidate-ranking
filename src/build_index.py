"""Offline pre-compute: dense embeddings for all profiles + JD aspect query vectors.

Streams candidates.jsonl IN ORDER, embeds each profile's text, and writes:
  artifacts/embeddings.npy      float32 [N, d], L2-normalized (cosine = dot)
  artifacts/jd_vectors.npy      float32 [Q, d] for aspects._query_order
  artifacts/candidate_ids.json  ordered ids (alignment guard for rank.py)
  artifacts/index_meta.json     embedder kind, dim, count

BM25 is NOT persisted here -- it is cheap to rebuild from the jsonl at rank time
(recall.py), which avoids a large artifact and any order/version skew.

Run (offline, network allowed for first model download):
  python src/build_index.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from config import ARTIFACTS, EXPECTED_CANDIDATE_COUNT
from embedder import get_embedder, save_embedder
from jd_aspects import build as build_aspects
from parse import stream_candidates


def main(prefer: str = "auto") -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    t0 = time.time()

    print("loading profile texts (streaming)...")
    ids: list[str] = []
    texts: list[str] = []
    for c in stream_candidates():
        ids.append(c.candidate_id)
        texts.append(c.profile_text())
    n = len(ids)
    print(f"  {n:,} profiles")
    assert n == EXPECTED_CANDIDATE_COUNT, f"expected {EXPECTED_CANDIDATE_COUNT}, got {n}"

    emb = get_embedder(prefer)
    print(f"embedder: {emb.kind}")
    emb.fit(texts)  # no-op for ST; fits LSA for fallback

    print("embedding profiles...")
    mat = emb.encode(texts)
    mat = np.ascontiguousarray(mat.astype(np.float32))
    print(f"  embeddings: {mat.shape} ({mat.nbytes/1e6:.0f} MB)")

    print("embedding JD aspect queries...")
    cfg = build_aspects()
    aspects_by_id = {a["id"]: a for a in cfg["must_have"] + cfg["nice_to_have"]}
    query_texts = [aspects_by_id[qid]["query"] for qid in cfg["_query_order"]]
    jd_vecs = emb.encode(query_texts).astype(np.float32)

    save_embedder(emb, ARTIFACTS / "embedder.pkl")  # enables embedding arbitrary JDs at serve time
    np.save(ARTIFACTS / "embeddings.npy", mat)
    np.save(ARTIFACTS / "jd_vectors.npy", jd_vecs)
    (ARTIFACTS / "candidate_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (ARTIFACTS / "index_meta.json").write_text(json.dumps({
        "embedder": emb.kind, "dim": int(mat.shape[1]), "count": n,
        "query_order": cfg["_query_order"],
    }, indent=2), encoding="utf-8")

    print(f"done in {time.time()-t0:.1f}s -> artifacts/{{embeddings,jd_vectors}}.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["auto", "st", "tfidf"], default="auto")
    main(ap.parse_args().embedder)
