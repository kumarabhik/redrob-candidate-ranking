"""Weak-label rubric (design_doc.md §7.4): no ground truth is shipped, so we derive a
graded relevance tier (0-4) from the JD's own explicit rubric. LambdaMART trains on these
to learn non-linear feature interactions; honeypots and hard disqualifiers are forced to 0.

The rubric is intentionally strict: the JD says it wants "10 great matches, not 1000 maybes",
so most candidates land in low tiers. relevance_score() returns a continuous [0,1] used both
for binning and as a deterministic fallback ranker in rank.py.
"""
from __future__ import annotations


from features import MUST_SLOTS, NICE_SLOTS, ordered_aspects


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sem_norm(sem: float) -> float:
    # MiniLM cosine for a relevant query-profile pair sits ~0.05-0.45; rescale to [0,1].
    return _clip((sem - 0.05) / 0.35)


def relevance_score(f: dict[str, float], aspects: dict, is_honeypot: bool = False) -> float:
    """Continuous JD-fit relevance in [0,1]. Honeypots -> 0."""
    if is_honeypot:
        return 0.0

    # --- core must-have fit (semantic + lexical + career evidence) --------
    # Read fixed weight-ordered slots (same ordering as features.py).
    num = den = 0.0
    for k, a in enumerate(ordered_aspects(aspects, "must_have")[:MUST_SLOTS], 1):
        w = a["weight"]
        sem = _sem_norm(f.get(f"sem_must_{k}", 0.0))
        hits = min(1.0, f.get(f"hit_must_{k}", 0.0) / 2.0)
        cev = f.get(f"cev_must_{k}", 0.0)
        fit = _clip(0.5 * sem + 0.3 * hits + 0.2 * cev)
        num += w * fit
        den += w
    core = num / den if den else 0.0

    # nice-to-have bonus (small, capped)
    nice = 0.0
    for k, a in enumerate(ordered_aspects(aspects, "nice_to_have")[:NICE_SLOTS], 1):
        nice += a["weight"] * min(1.0, f.get(f"hit_nice_{k}", 0.0) / 2.0)
    nice_bonus = min(0.10, 0.10 * nice)

    # --- contextual multipliers ------------------------------------------
    seniority = 0.1 + 0.9 * f.get("seniority_band", 0.0)
    location = f.get("location_fit", 0.5)
    availability = 0.3 + 0.7 * f.get("availability", 0.5)
    product = 1.0 - 0.5 * f.get("frac_consulting", 0.0)
    if f.get("all_consulting", 0.0) >= 1.0:
        product *= 0.6

    # verified-competence credibility: assessments + trustworthy endorsements + systems tenure
    cred = (f.get("assessment_ai_mean", 0.0) + f.get("endorsement_trust", 0.0)
            + f.get("systems_months", 0.0)) / 3.0
    credibility = 0.85 + 0.15 * cred

    # --- multiplicative penalties for explicit "do NOT want" signals ------
    pen = 1.0
    if f.get("non_eng_title", 0.0) >= 1.0 or f.get("frac_non_eng_titles", 0.0) > 0.5:
        pen *= 0.20                      # the "Marketing Manager with AI skills" trap
    if f.get("wrong_domain", 0.0) >= 1.0:
        pen *= 0.30                      # CV/speech/robotics primary, no IR
    if f.get("keyword_stuffer", 0.0) >= 1.0:
        pen *= 0.35                      # perfect skill list, no real systems work
    if f.get("research_only", 0.0) >= 1.0:
        pen *= 0.40                      # pure research, no production
    if f.get("managerial_title", 0.0) >= 1.0:
        pen *= 0.60                      # moved off the keyboard
    if f.get("job_hopper", 0.0) >= 1.0:
        pen *= 0.70                      # title-chaser pattern

    rel = core * seniority * location * availability * product * credibility * pen + nice_bonus
    return _clip(rel)


# tier thresholds (graded relevance for NDCG-style training)
_TIERS = [(0.68, 4), (0.52, 3), (0.38, 2), (0.24, 1)]


def relevance_tier(score: float) -> int:
    for thr, tier in _TIERS:
        if score >= thr:
            return tier
    return 0
