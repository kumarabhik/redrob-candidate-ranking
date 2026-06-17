"""Honeypot / impossible-profile detection (design_doc.md §7.3).

The dataset seeds ~80 honeypots with subtly impossible profiles. Ranking them into the
top 100 at >10% is an automatic disqualification, so we hard-filter them before reranking
(agents.md §1). Detection is deterministic and rule-based: each rule encodes a logical
impossibility, not a quality judgement. A candidate is a honeypot if ANY hard rule fires.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from config import (
    DURATION_TOLERANCE_MONTHS,
    SKILL_VS_CAREER_SLACK_MONTHS,
    SNAPSHOT_DATE,
)
from parse import Candidate

_SNAPSHOT = date.fromisoformat(SNAPSHOT_DATE)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _months_between(a: date, b: date) -> int:
    """Whole months from a to b (>=0 if b after a)."""
    return (b.year - a.year) * 12 + (b.month - a.month)


# Calibration note (2026-06-15, full-pool scan): the synthetic data is intentionally
# noisy. ~18.8% of candidates have an inverted salary band and a large fraction have a
# single skill whose duration exceeds total career. Those are NOT honeypot signals --
# treating them as hard rules flags 30% of the pool. The ~80 deliberate honeypots are
# caught by EGREGIOUS impossibilities that essentially never occur in normal records:
# career duration vs date-span mismatch, current-role-with-end-date, reversed dates,
# reversed education years, and MULTIPLE expert-with-0-months skills at once.
# Each reason is (code, message). is_honeypot() fires only on HARD codes.

HARD_CODES = {
    "ROLE_DUR_SPAN",      # duration_months disagrees with the actual date span
    "ROLE_CURRENT_END",   # is_current but has an end_date
    "ROLE_END_LT_START",  # end_date before start_date
    "ROLE_FUTURE",        # role starts after the snapshot
    "EDU_INV",            # education start_year > end_year
    "MULTI_EXPERT_ZERO",  # >=3 skills marked expert with 0 months used
}

# Soft codes: recorded as quality signals (fed to features later) but never used to
# hard-filter, because they are common in the noisy synthetic data.
SOFT_CODES = {"SALARY_INV", "SKILL_GT_CAREER", "EXPERT_ZERO"}

MIN_EXPERT_ZERO_FOR_HONEYPOT = 3


def coded_reasons(c: Candidate) -> list[tuple[str, str]]:
    """All consistency findings as (code, message) pairs (hard + soft)."""
    out: list[tuple[str, str]] = []
    yoe_months = c.years_of_experience * 12

    expert_zero = 0
    for sk in c.skills:
        dur = sk.get("duration_months")
        if sk.get("proficiency") == "expert" and dur == 0:
            expert_zero += 1
            out.append(("EXPERT_ZERO", f"skill '{sk.get('name')}' expert with 0 months"))
        if isinstance(dur, (int, float)) and dur > yoe_months + SKILL_VS_CAREER_SLACK_MONTHS:
            out.append(("SKILL_GT_CAREER",
                        f"skill '{sk.get('name')}' {dur}mo > career {yoe_months:.0f}mo"))
    if expert_zero >= MIN_EXPERT_ZERO_FOR_HONEYPOT:
        out.append(("MULTI_EXPERT_ZERO", f"{expert_zero} skills are expert with 0 months"))

    for role in c.career_history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        dur = role.get("duration_months")
        title = role.get("title")
        if role.get("is_current") and role.get("end_date") not in (None, ""):
            out.append(("ROLE_CURRENT_END", f"role '{title}' is_current but has end_date"))
        if start and end and end < start:
            out.append(("ROLE_END_LT_START", f"role '{title}' end before start"))
        if start and start > _SNAPSHOT:
            out.append(("ROLE_FUTURE", f"role '{title}' starts in the future"))
        if start and isinstance(dur, (int, float)):
            span_end = end if end else _SNAPSHOT
            span = _months_between(start, span_end)
            # One-sided: only an OVER-claim is impossible (can't log more months than the
            # calendar window allows). Under-claims are benign date gaps, not honeypots.
            if span >= 0 and dur > span + max(DURATION_TOLERANCE_MONTHS, 0.15 * span):
                out.append(("ROLE_DUR_SPAN", f"role '{title}' {dur}mo > span {span}mo"))

    sal = c.signals.get("expected_salary_range_inr_lpa", {})
    smin, smax = sal.get("min"), sal.get("max")
    if isinstance(smin, (int, float)) and isinstance(smax, (int, float)) and smin > smax:
        out.append(("SALARY_INV", f"salary min {smin} > max {smax}"))

    for ed in c.education:
        sy, ey = ed.get("start_year"), ed.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and sy > ey:
            out.append(("EDU_INV", f"education '{ed.get('institution')}' {sy} > {ey}"))

    return out


def honeypot_reasons(c: Candidate) -> list[str]:
    """Hard-impossibility messages only. Empty list => not a honeypot."""
    return [msg for code, msg in coded_reasons(c) if code in HARD_CODES]


def is_honeypot(c: Candidate) -> bool:
    return any(code in HARD_CODES for code, _ in coded_reasons(c))


if __name__ == "__main__":
    from collections import Counter

    from parse import stream_candidates

    n = 0
    hard_flagged = 0
    per_code: Counter[str] = Counter()       # candidates with >=1 finding of this code
    hard_examples: list[tuple[str, list[str]]] = []
    for c in stream_candidates():
        n += 1
        codes = {code for code, _ in coded_reasons(c)}
        for code in codes:
            per_code[code] += 1
        hard_msgs = honeypot_reasons(c)
        if hard_msgs:
            hard_flagged += 1
            if len(hard_examples) < 15:
                hard_examples.append((c.candidate_id, hard_msgs))

    print(f"scanned: {n:,}\n")
    print(f"HARD honeypots (excluded from top 100): {hard_flagged} ({100*hard_flagged/n:.3f}%)\n")
    print("per-code candidate counts (HARD=excluded, SOFT=feature signal only):")
    for code, v in per_code.most_common():
        kind = "HARD" if code in HARD_CODES else "soft"
        print(f"  [{kind}] {code:18} {v:6}  ({100*v/n:.2f}%)")
    print("\nhard-honeypot examples:")
    for cid, rs in hard_examples:
        print(f"  {cid}: {rs[0]}")
