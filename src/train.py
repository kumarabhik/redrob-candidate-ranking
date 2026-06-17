"""Stage D training (offline): LambdaMART reranker over engineered features.

No ground truth is shipped, so we train on weak labels from the JD rubric (labels.py),
with honeypots forced to tier 0. XGBoost `rank:ndcg` directly optimizes the metric family
we are scored on. The trained model is small and infers in milliseconds on CPU at rank time.

Run (offline): python src/train.py
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from config import ARTIFACTS
from features import feature_names, features_for, vectorize
from honeypot import is_honeypot
from jd_aspects import build as build_aspects
from labels import relevance_score, relevance_tier
from parse import stream_candidates
from recall import aspect_sims, load_artifacts


def build_training_matrix():
    A = load_artifacts()
    aspects = A["aspects"]
    query_order = A["meta"]["query_order"]
    names = feature_names(aspects)
    sims = aspect_sims(A["mat"], A["jd_vecs"])

    X = np.zeros((len(A["ids"]), len(names)), dtype=np.float32)
    y = np.zeros(len(A["ids"]), dtype=np.int32)
    rubric = np.zeros(len(A["ids"]), dtype=np.float32)

    for i, c in enumerate(stream_candidates()):
        f = features_for(c, sims[i], query_order, aspects)
        X[i] = vectorize(f, names)
        hp = is_honeypot(c)
        score = relevance_score(f, aspects, is_honeypot=hp)
        rubric[i] = score
        y[i] = relevance_tier(score)
    return X, y, rubric, names, A["ids"]


def balanced_training_set(X, y, rubric, neg_per_pos: int = 12, hard_frac: float = 0.6,
                          seed: int = 42):
    """Positives (tier>=1) + sampled negatives (hard negatives by rubric + random).

    A single 100k group is ~99.5% zeros, which collapses the lambdarank gradient. We train
    on a balanced group: every positive plus a mix of HARD negatives (high rubric but tier 0
    -- keyword-stuffers, wrong-domain, near-misses) and RANDOM negatives (broad coverage).
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y >= 1)[0]
    neg = np.where(y == 0)[0]
    n_neg = min(len(neg), max(2000, neg_per_pos * len(pos)))

    # hard negatives = highest rubric among tier-0
    neg_by_rubric = neg[np.argsort(-rubric[neg])]
    n_hard = int(hard_frac * n_neg)
    hard = neg_by_rubric[:n_hard]
    rest = neg_by_rubric[n_hard:]
    rand = rng.choice(rest, size=min(len(rest), n_neg - n_hard), replace=False)
    sel = np.concatenate([pos, hard, rand])
    rng.shuffle(sel)
    return sel


def main() -> None:
    import xgboost as xgb

    t0 = time.time()
    build_aspects()  # ensure aspects.json fresh
    print("building training matrix (features for all 100k)...")
    X, y, rubric, names, ids = build_training_matrix()
    print(f"  X={X.shape}  built in {time.time()-t0:.1f}s")
    print("  full label distribution:", dict(sorted(Counter(y.tolist()).items())))

    sel = balanced_training_set(X, y, rubric)
    Xs, ys = X[sel], y[sel]
    print(f"  balanced training group: {len(sel)} docs, "
          f"labels {dict(sorted(Counter(ys.tolist()).items()))}")

    print("training XGBoost rank:ndcg (LambdaMART)...")
    ranker = xgb.XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@50",
        lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=16,
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=1.0,
        reg_lambda=1.0, n_jobs=-1, random_state=42,
    )
    ranker.fit(Xs, ys, group=[len(ys)])

    ranker.save_model(str(ARTIFACTS / "model.bin"))
    (ARTIFACTS / "feature_names.json").write_text(json.dumps(names), encoding="utf-8")

    # sanity: model ordering vs rubric ordering over the FULL pool
    pred = ranker.predict(X)
    top_model = set(np.argsort(-pred)[:100].tolist())
    top_rubric = set(np.argsort(-rubric)[:100].tolist())
    overlap = len(top_model & top_rubric)
    nonzero = int((ranker.feature_importances_ > 0).sum())
    imp = sorted(zip(names, ranker.feature_importances_), key=lambda t: -t[1])[:12]

    print(f"\ntop-100 overlap(model, rubric): {overlap}/100   (non-zero features: {nonzero})")
    print("top feature importances:")
    for n, v in imp:
        if v > 0:
            print(f"  {v:.3f}  {n}")
    print(f"\nsaved model.bin + feature_names.json  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
