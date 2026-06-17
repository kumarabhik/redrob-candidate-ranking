"""Sandbox demo (submission_spec §10.5): rank a SMALL candidate sample end-to-end.

The organizers' sandbox check feeds a <=100-candidate sample and verifies the ranker runs
and emits a ranked CSV within the CPU budget. This script is self-contained (no web deps):
it embeds the sample on the fly (TF-IDF+SVD over the sample), runs the same features +
honeypot gate + ensemble (model if artifacts/model.bin exists, else the rubric), and writes
a ranked CSV with fact-grounded reasons. The full 100k pipeline lives in rank.py.

Run:  python src/sandbox_demo.py --sample <sample_candidates.json> --out demo_ranked.csv
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from config import ARTIFACTS
from features import feature_names, features_for, vectorize
from honeypot import is_honeypot
from jd_aspects import build as build_aspects
from labels import relevance_score
from parse import Candidate
from reasoning import reason_for


def _sample_sims(records, aspects):
    """Dense aspect sims for the sample via a tiny TF-IDF+SVD fit on the sample itself.
    For very small samples this is approximate; lexical + behavioral features carry the rest."""
    qorder = [a["id"] for a in aspects["must_have"]] + [a["id"] for a in aspects["nice_to_have"]]
    texts = [c.profile_text() for c in records]
    if len(records) < 10:
        return np.zeros((len(records), len(qorder)), dtype=np.float32), qorder
    from embedder import TfidfSVDEmbedder
    emb = TfidfSVDEmbedder(dim=min(64, len(records) - 1))
    emb.fit(texts)
    mat = emb.encode(texts)
    aspect_by_id = {a["id"]: a for a in aspects["must_have"] + aspects["nice_to_have"]}
    jd = emb.encode([aspect_by_id[q]["query"] for q in qorder])
    return (mat @ jd.T).astype(np.float32), qorder


def rank_sample(records: list, top: int = 100):
    """Rank a list of Candidate objects. Returns (rows, n_gated) where each row is
    (candidate_id, rank, score, reasoning). Reused by the CLI and the Gradio app."""
    aspects = build_aspects()
    sims, qorder = _sample_sims(records, aspects)
    names = feature_names(aspects)

    model = None
    mb = ARTIFACTS / "model.bin"
    if mb.exists():
        try:
            import xgboost as xgb
            model = xgb.XGBRanker(); model.load_model(str(mb))
        except Exception:  # noqa: BLE001
            model = None

    keep, X = [], []
    for i, c in enumerate(records):
        if is_honeypot(c):
            continue
        f = features_for(c, sims[i], qorder, aspects)
        keep.append((c, f))
        X.append(vectorize(f, names))
    X = np.asarray(X, dtype=np.float32)

    rubric = np.array([relevance_score(f, aspects) for _, f in keep], dtype=np.float32)
    if model is not None and len(X):
        ms = model.predict(X)
        ms = (ms - ms.min()) / (ms.max() - ms.min()) if ms.max() > ms.min() else np.ones_like(ms)
        scores = 0.6 * ms + 0.4 * rubric
    else:
        scores = rubric

    order = sorted(range(len(keep)), key=lambda j: (-float(scores[j]), keep[j][0].candidate_id))
    top_idx = order[:min(top, len(order))]
    raw_s = np.array([scores[j] for j in top_idx], dtype=float)
    lo, hi = (raw_s.min(), raw_s.max()) if len(raw_s) else (0, 0)
    norm = 0.05 + 0.94 * ((raw_s - lo) / (hi - lo) if hi > lo else np.ones_like(raw_s))

    rows = []
    for rank, j in enumerate(top_idx, 1):
        c, f = keep[j]
        rows.append((c.candidate_id, rank, round(float(norm[rank - 1]), 6),
                     reason_for(c, f, aspects, rank)))
    return rows, len(records) - len(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="JSON list of candidate records")
    ap.add_argument("--out", default="demo_ranked.csv")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    raw = json.loads(open(args.sample, encoding="utf-8").read())
    records = [Candidate(r) for r in raw]
    rows, n_gated = rank_sample(records, args.top)

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reason in rows:
            w.writerow([cid, rank, f"{score:.6f}", reason])

    print(f"ranked {len(rows)} of {len(records)} candidates ({n_gated} honeypots gated) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
