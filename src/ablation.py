"""Ablation: what does each ranking stage contribute? (roadmap.md stretch goals)

Compares BM25-only / dense-only / RRF-fusion / RRF+LambdaMART-ensemble. Each variant's
top-100 is scored with the competition metrics using the JD rubric tier (labels.py) as the
reference relevance, plus the honeypot count in the top 100 (trap avoidance).

Honesty caveat: no hidden ground truth exists, so the rubric is the reference. This measures
how well each stage SURFACES and ORDERS rubric-strong candidates and AVOIDS traps -- a
relative comparison, not an absolute leaderboard score. Run: python src/ablation.py
"""
from __future__ import annotations

import time

import numpy as np

from evaluate import composite
from features import feature_names, features_for, vectorize
from honeypot import is_honeypot
from labels import relevance_score, relevance_tier
from parse import stream_candidates
from rank import _unit, SHORTLIST_PER_LIST, SHORTLIST_TAKE
from recall import (aspect_sims, bm25_query_terms, build_bm25, dense_recall_score,
                    load_artifacts, rrf, shortlist, tokenize)


def main() -> None:
    import xgboost as xgb

    t0 = time.time()
    A = load_artifacts()
    aspects, qorder = A["aspects"], A["meta"]["query_order"]
    names = feature_names(aspects)
    sims = aspect_sims(A["mat"], A["jd_vecs"])

    records = list(stream_candidates())
    corpus = [tokenize(c.profile_text()) for c in records]
    dense = dense_recall_score(sims, qorder, aspects)
    bm25 = build_bm25(corpus)
    bm25_s = np.asarray(bm25.get_scores(bm25_query_terms(aspects)), dtype=np.float32)

    dense_top = np.argsort(-dense)[:100]
    bm25_top = np.argsort(-bm25_s)[:100]
    rrf_order = rrf([np.argsort(-dense)[:SHORTLIST_PER_LIST],
                     np.argsort(-bm25_s)[:SHORTLIST_PER_LIST]], n=SHORTLIST_TAKE)
    rrf_top = rrf_order[:100]

    # LambdaMART ensemble over the RRF shortlist (mirrors rank.py, honeypot-gated)
    model = xgb.XGBRanker(); model.load_model(str(__import__("config").ARTIFACTS / "model.bin"))
    keep_idx, X = [], []
    for i in rrf_order:
        c = records[i]
        if is_honeypot(c):
            continue
        keep_idx.append(i)
        X.append(vectorize(features_for(c, sims[i], qorder, aspects), names))
    X = np.asarray(X, dtype=np.float32)
    rub_keep = np.array([relevance_score(features_for(records[i], sims[i], qorder, aspects),
                                         aspects) for i in keep_idx], dtype=np.float32)
    final = 0.60 * _unit(model.predict(X)) + 0.40 * _unit(rub_keep)
    ltr_top = [keep_idx[j] for j in np.argsort(-final)[:100]]

    # reference relevance (rubric tier) + honeypot for every index we need
    need = set(map(int, dense_top)) | set(map(int, bm25_top)) | set(map(int, rrf_top)) | set(ltr_top)
    tier, hp = {}, {}
    for i in need:
        f = features_for(records[i], sims[i], qorder, aspects)
        h = is_honeypot(records[i])
        tier[i] = 0 if h else relevance_tier(relevance_score(f, aspects, is_honeypot=h))
        hp[i] = h

    variants = {"bm25_only": list(bm25_top), "dense_only": list(dense_top),
                "rrf_fusion": list(rrf_top), "rrf+lambdamart": list(ltr_top)}

    print(f"\n{'variant':18} {'ndcg@10':>8} {'ndcg@50':>8} {'map':>6} {'p@10':>6} "
          f"{'composite':>10} {'honeypots@100':>14}")
    for name, idxs in variants.items():
        rels = [tier[int(i)] for i in idxs]
        m = composite(rels)
        nhp = sum(hp[int(i)] for i in idxs)
        print(f"{name:18} {m['ndcg@10']:8.4f} {m['ndcg@50']:8.4f} {m['map']:6.3f} "
              f"{m['p@10']:6.3f} {m['composite']:10.4f} {nhp:>14}")
    print(f"\n(reference = JD rubric tier; relative comparison. built in {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
