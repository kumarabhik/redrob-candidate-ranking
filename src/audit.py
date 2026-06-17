"""Fairness / data-validation audit (design_doc.md §3 non-goals note; judging: data validation).

Compares the top-100 submission against the full pool across attributes that a recruiter
should sanity-check for skew: education tier, company size, location, availability, and
experience. This is a transparency tool, not a debiasing module: it surfaces where the
ranking concentrates so the skew can be judged (some skew is intended by the JD, e.g.
product-company and India-location preference; other skew would be a red flag).

Run:  python src/audit.py --submission submission.csv
"""
from __future__ import annotations

import argparse
import csv

from config import CANDIDATES_JSONL
from parse import stream_candidates


def _profile_stats(cands):
    n = len(cands)
    if n == 0:
        return {}
    tier1 = sum(1 for c in cands if any(e.get("tier") == "tier_1" for e in c.education))
    big = sum(1 for c in cands if c.profile.get("current_company_size") in
              ("1001-5000", "5001-10000", "10001+"))
    india = sum(1 for c in cands if "india" in (c.profile.get("country", "")).lower())
    otw = sum(1 for c in cands if c.signals.get("open_to_work_flag"))
    yrs = sum(c.years_of_experience for c in cands) / n
    rr = sum(float(c.signals.get("recruiter_response_rate", 0) or 0) for c in cands) / n
    gh = sum(1 for c in cands if float(c.signals.get("github_activity_score", -1) or -1) >= 0)
    return {"tier_1_edu_%": 100 * tier1 / n, "large_company_%": 100 * big / n,
            "india_%": 100 * india / n, "open_to_work_%": 100 * otw / n,
            "github_linked_%": 100 * gh / n, "avg_years_exp": yrs,
            "avg_recruiter_response": rr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--pool-sample", type=int, default=100000)
    args = ap.parse_args()

    with open(args.submission, encoding="utf-8") as fh:
        top_ids = {r["candidate_id"] for r in csv.DictReader(fh)}

    pool, top = [], []
    for i, c in enumerate(stream_candidates()):
        if i < args.pool_sample:
            pool.append(c)
        if c.candidate_id in top_ids:
            top.append(c)

    ps, ts = _profile_stats(pool), _profile_stats(top)
    print(f"{'attribute':24} {'pool':>10} {'top-100':>10} {'delta':>10}")
    for k in ps:
        d = ts[k] - ps[k]
        print(f"{k:24} {ps[k]:10.2f} {ts[k]:10.2f} {d:+10.2f}")
    print("\nReading: positive delta = over-represented in the top-100 vs pool. Expected/intended:"
          "\n  higher avg experience, India %, open-to-work %, recruiter-response (JD wants"
          "\n  available product-company seniors). Watch for unintended skew (e.g. tier_1 edu).")


if __name__ == "__main__":
    main()
