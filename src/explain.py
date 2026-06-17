"""Explainability: per-candidate feature attributions via XGBoost SHAP contributions.

For the top picks in a submission, shows which features pushed the model score up or down
(exact tree SHAP values from XGBoost `pred_contribs`). This complements the natural-language
reasoning with a transparent, auditable breakdown -- directly serving the "Explainability"
judging criterion and the Stage-5 defence.

Run:  python src/explain.py --submission submission.csv --n 10
Writes artifacts/explanations.md and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from config import ARTIFACTS, CANDIDATES_JSONL
from features import feature_names, features_for, vectorize
from parse import stream_candidates
from recall import aspect_sims, load_artifacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--candidates", default=str(CANDIDATES_JSONL))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--top-features", type=int, default=6)
    args = ap.parse_args()

    import xgboost as xgb

    A = load_artifacts()
    aspects, qorder = A["aspects"], A["meta"]["query_order"]
    names = feature_names(aspects)
    sims = aspect_sims(A["mat"], A["jd_vecs"])
    id2idx = {cid: i for i, cid in enumerate(A["ids"])}

    with open(args.submission, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[: args.n]
    wanted = {r["candidate_id"] for r in rows}

    recs = {}
    for c in stream_candidates(args.candidates):
        if c.candidate_id in wanted:
            recs[c.candidate_id] = c
            if len(recs) == len(wanted):
                break

    model = xgb.XGBRanker(); model.load_model(str(ARTIFACTS / "model.bin"))
    booster = model.get_booster()

    X = np.array([vectorize(features_for(recs[r["candidate_id"]],
                                         sims[id2idx[r["candidate_id"]]], qorder, aspects), names)
                  for r in rows], dtype=np.float32)
    dm = xgb.DMatrix(X, feature_names=names)
    contribs = booster.predict(dm, pred_contribs=True)  # [n, F+1], last col = bias

    lines = ["# Per-candidate explanations (XGBoost SHAP contributions)\n"]
    for r, contrib in zip(rows, contribs):
        c = recs[r["candidate_id"]]
        feat_contrib = contrib[:-1]
        idx = np.argsort(-np.abs(feat_contrib))[: args.top_features]
        parts = [f"{'+' if feat_contrib[k] >= 0 else '-'}{abs(feat_contrib[k]):.3f} {names[k]}"
                 for k in idx]
        head = (f"## #{r['rank']} {r['candidate_id']} - {c.profile.get('current_title')} "
                f"@ {c.profile.get('current_company')} (score {r['score']})")
        lines += [head, "", "top drivers: " + ", ".join(parts), "", f"> {r['reasoning']}", ""]
        print(head)
        print("   " + " | ".join(parts))

    (ARTIFACTS / "explanations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote artifacts/explanations.md ({len(rows)} candidates)")


if __name__ == "__main__":
    main()
