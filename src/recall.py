"""Stage A -- hybrid recall (Pillar 1): BM25 (sparse) + dense embeddings fused with RRF.

Dense and sparse retrieval are complementary: BM25 catches exact must-have terms and
career-description phrasing; dense embeddings catch strong candidates who describe the same
work without the JD's buzzwords (the "Tier-5 without keywords" case). We fuse by Reciprocal
Rank Fusion (training-free, score-scale-agnostic).

The per-candidate dense similarity to each JD aspect (the [N, Q] matrix) is also reused as
semantic aspect-fit features by features.py -- so we compute it once here.
"""
from __future__ import annotations

import json
import re

import numpy as np

from config import ARTIFACTS

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --- artifact loading ----------------------------------------------------
def load_artifacts() -> dict:
    mat = np.load(ARTIFACTS / "embeddings.npy")
    jd_vecs = np.load(ARTIFACTS / "jd_vectors.npy")
    ids = json.loads((ARTIFACTS / "candidate_ids.json").read_text(encoding="utf-8"))
    meta = json.loads((ARTIFACTS / "index_meta.json").read_text(encoding="utf-8"))
    aspects = json.loads((ARTIFACTS / "aspects.json").read_text(encoding="utf-8"))
    return {"mat": mat, "jd_vecs": jd_vecs, "ids": ids, "meta": meta, "aspects": aspects}


# --- dense ---------------------------------------------------------------
def aspect_sims(mat: np.ndarray, jd_vecs: np.ndarray) -> np.ndarray:
    """[N, Q] cosine sims (vectors are pre-normalized, so dot == cosine)."""
    return mat @ jd_vecs.T


def dense_recall_score(sims: np.ndarray, query_order: list[str], aspects: dict) -> np.ndarray:
    """Weighted must-have semantic score per candidate -> [N]."""
    w = {a["id"]: a["weight"] for a in aspects["must_have"]}
    must_ids = [a["id"] for a in aspects["must_have"]]
    idx = [query_order.index(i) for i in must_ids]
    weights = np.array([w[i] for i in must_ids], dtype=np.float32)
    sub = sims[:, idx]
    return (sub * weights).sum(axis=1) / weights.sum()


# --- sparse (BM25) -------------------------------------------------------
def build_bm25(corpus_tokens: list[list[str]]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi(corpus_tokens)


def bm25_query_terms(aspects: dict) -> list[str]:
    """Query = role keywords + must-have skill vocab (the exact-term half of recall)."""
    terms: list[str] = ["ai", "machine", "learning", "ml", "engineer", "retrieval",
                         "ranking", "search", "recommendation", "embeddings", "python", "nlp"]
    for a in aspects["must_have"]:
        for s in a["skills"]:
            terms.extend(tokenize(s))
    # de-dup, keep order
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --- RRF fusion ----------------------------------------------------------
def rrf(ranked_lists: list[np.ndarray], k: int = 60, n: int | None = None) -> np.ndarray:
    """Reciprocal Rank Fusion. Each list is candidate indices best-first.
    Returns fused indices best-first."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, idx in enumerate(lst):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused = np.array([i for i, _ in order], dtype=np.int64)
    return fused[:n] if n else fused


def shortlist(dense_score: np.ndarray, bm25_score: np.ndarray,
              per_list: int = 4000, take: int = 8000) -> np.ndarray:
    """Top-`per_list` from dense and BM25 each -> RRF -> top `take` indices."""
    dense_top = np.argsort(-dense_score)[:per_list]
    bm25_top = np.argsort(-bm25_score)[:per_list]
    return rrf([dense_top, bm25_top], k=60, n=take)


if __name__ == "__main__":
    # Sanity: load artifacts, build BM25 from the jsonl, fuse, print a few top profiles.
    import time

    from parse import stream_candidates

    t0 = time.time()
    A = load_artifacts()
    sims = aspect_sims(A["mat"], A["jd_vecs"])
    dense = dense_recall_score(sims, A["meta"]["query_order"], A["aspects"])

    print("tokenizing corpus for BM25...")
    records = list(stream_candidates())
    corpus = [tokenize(c.profile_text()) for c in records]
    bm25 = build_bm25(corpus)
    bm25_score = np.array(bm25.get_scores(bm25_query_terms(A["aspects"])), dtype=np.float32)

    sl = shortlist(dense, bm25_score)
    print(f"shortlist size: {len(sl)}  (built in {time.time()-t0:.1f}s)")
    print("\ntop 10 by fused recall:")
    for rank, i in enumerate(sl[:10], 1):
        c = records[i]
        p = c.profile
        print(f"  {rank:2}. {c.candidate_id}  {p['current_title']} @ {p['current_company']} "
              f"| {p['years_of_experience']}y | dense={dense[i]:.3f}")
