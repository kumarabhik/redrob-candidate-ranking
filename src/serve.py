"""Multi-JD serving layer (system_design.md §11).

Loads the JD-INDEPENDENT candidate index once (embeddings, BM25, honeypot flags, embedder)
and ranks ANY job description by deriving its aspects, embedding its queries, and running
recall -> features -> honeypot gate -> rank -> reason. The expensive 100k work is shared
across all users/JDs; each rank_jd() call is the cheap per-JD part.

For arbitrary JDs we rank by the general JD-rubric (+ availability), since the trained
LambdaMART model's feature columns are tied to the default role's aspect ids. The default
hackathon JD still uses the full ensemble via rank.py.
"""
from __future__ import annotations

import time

import numpy as np

from config import ARTIFACTS, CANDIDATES_JSONL
from embedder import load_embedder
from features import feature_names, features_for, vectorize
from honeypot import is_honeypot
from jd_aspects import aspects_from_jd_text
from labels import relevance_score
from parse import stream_candidates
from reasoning import reason_for
from recall import bm25_query_terms, build_bm25, rrf, tokenize

TWIN_THRESHOLD = 0.985
# Aspect domains the LambdaMART model was trained on (the released AI-engineer role).
TRAINED_DOMAIN = {"retrieval", "vectordb", "ranking_eval", "python_ml", "llm"}
_CACHE: dict = {}


def _index():
    """Build/return the shared, JD-independent candidate index (once)."""
    if _CACHE:
        return _CACHE
    t0 = time.time()
    mat = np.load(ARTIFACTS / "embeddings.npy")
    emb = load_embedder(ARTIFACTS / "embedder.pkl")
    records = list(stream_candidates(CANDIDATES_JSONL))
    corpus = [tokenize(c.profile_text()) for c in records]
    bm25 = build_bm25(corpus)
    honeypot = np.array([is_honeypot(c) for c in records], dtype=bool)
    model = None
    try:                                  # slot-based model transfers across JDs
        import xgboost as xgb
        model = xgb.XGBRanker(); model.load_model(str(ARTIFACTS / "model.bin"))
    except Exception:  # noqa: BLE001
        model = None
    _CACHE.update(mat=mat, emb=emb, records=records, bm25=bm25, honeypot=honeypot, model=model)
    _CACHE["build_s"] = round(time.time() - t0, 1)
    return _CACHE


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)


def rank_jd(jd_text: str, role_title: str = "Role", top_k: int = 100,
            per_list: int = 4000, take: int = 8000, dedup: bool = True):
    """Rank the shared candidate pool for an arbitrary JD. Returns (rows, meta)."""
    ix = _index()
    mat, emb, records, bm25, honeypot, model = (ix["mat"], ix["emb"], ix["records"],
                                                ix["bm25"], ix["honeypot"], ix["model"])
    t0 = time.time()

    aspects = aspects_from_jd_text(jd_text, role_title)
    qorder = aspects["_query_order"]
    aspect_by_id = {a["id"]: a for a in aspects["must_have"] + aspects["nice_to_have"]}
    jd_vecs = emb.encode([aspect_by_id[q]["query"] for q in qorder]).astype(np.float32)
    sims = mat @ jd_vecs.T                                   # [N, |aspects|]

    # recall: weighted must-have dense + BM25 -> RRF
    must_ids = [a["id"] for a in aspects["must_have"]]
    w = np.array([aspect_by_id[i]["weight"] for i in must_ids], dtype=np.float32)
    midx = [qorder.index(i) for i in must_ids]
    dense = (sims[:, midx] * w).sum(axis=1) / max(w.sum(), 1e-9)
    bm25_s = np.asarray(bm25.get_scores(bm25_query_terms(aspects)), dtype=np.float32)
    order = rrf([np.argsort(-dense)[:per_list], np.argsort(-bm25_s)[:per_list]], n=take)

    # features + honeypot gate + (model + rubric) ensemble
    names = feature_names(aspects)
    keep, keep_idx, rubric, X = [], [], [], []
    for i in order:
        if honeypot[i]:
            continue
        f = features_for(records[i], sims[i], qorder, aspects)
        keep.append((records[i], f))
        keep_idx.append(int(i))
        rubric.append(relevance_score(f, aspects))
        X.append(vectorize(f, names))
    rubric = np.asarray(rubric, dtype=np.float32)
    # Domain routing: the trained model's feature schema is JD-agnostic, but its LEARNED
    # weights are AI/ML-specific (ai_skill_depth, systems_months, ...). Using it on an
    # unrelated role (e.g. Frontend) reintroduces AI bias. So use the model only when the
    # JD's dominant aspect is in its trained domain; otherwise rank by the neutral rubric.
    in_domain = bool(must_ids) and must_ids[0] in TRAINED_DOMAIN
    if model is not None and in_domain and len(X):
        ms = model.predict(np.asarray(X, dtype=np.float32))
        score = 0.6 * _unit(ms) + 0.4 * _unit(rubric)
        ranker = "model+rubric (in-domain)"
    else:
        score = rubric
        ranker = "rubric (general)"

    ordering = sorted(range(len(keep)), key=lambda j: (-float(score[j]), keep[j][0].candidate_id))

    # twin de-dup
    top, sel = [], None
    for j in ordering:
        v = mat[keep_idx[j]]
        if dedup and sel is not None and float((sel @ v).max()) > TWIN_THRESHOLD:
            continue
        top.append(j)
        sel = v[None, :] if sel is None else np.vstack([sel, v])
        if len(top) == top_k:
            break
    for j in ordering:                       # backfill if over-pruned
        if len(top) >= top_k:
            break
        if j not in top:
            top.append(j)

    raw = np.array([score[j] for j in top], dtype=float)
    lo, hi = (raw.min(), raw.max()) if len(raw) else (0.0, 0.0)
    norm = 0.06 + 0.93 * np.power((raw - lo) / (hi - lo) if hi > lo else np.ones_like(raw), 0.65)

    rows = []
    for rank, j in enumerate(top, 1):
        c, f = keep[j]
        rows.append({"candidate_id": c.candidate_id, "rank": rank,
                     "score": round(float(norm[rank - 1]), 4),
                     "title": c.profile.get("current_title"),
                     "company": c.profile.get("current_company"),
                     "years": c.profile.get("years_of_experience"),
                     "reasoning": reason_for(c, f, aspects, rank)})
    meta = {"shortlist": len(keep), "gated": int(honeypot[order].sum()),
            "must_aspects": must_ids, "ranker": ranker,
            "rank_ms": int((time.time() - t0) * 1000), "index_build_s": ix["build_s"]}
    return rows, meta


if __name__ == "__main__":
    import json

    jd = (ARTIFACTS / "jd_text.txt").read_text(encoding="utf-8") \
        if (ARTIFACTS / "jd_text.txt").exists() else "Senior AI Engineer, retrieval and ranking."
    rows, meta = rank_jd(jd, "Senior AI Engineer")
    print("meta:", json.dumps(meta))
    for r in rows[:8]:
        print(f"  {r['rank']:2}. {r['candidate_id']} {r['title']} @ {r['company']} ({r['years']}y)")
