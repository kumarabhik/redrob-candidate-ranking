"""Stage E -- deterministic, fact-grounded reasoning (design_doc.md §7.5).

No LLM at rank time. Each reason is assembled from evidence slots that are only filled when
the underlying field exists / a feature actually fired, so hallucination is impossible by
construction. Phrasing varies by rank band and by which slots fired, so sampled rows read
as substantively different and the tone matches the rank (Stage-4 checks).
"""
from __future__ import annotations

from parse import Candidate

_ASPECT_PHRASE = {
    "retrieval": "embeddings/retrieval work",
    "vectordb": "vector-search / hybrid-search infra",
    "ranking_eval": "ranking & evaluation (NDCG/MAP, A/B)",
    "python_ml": "production Python/ML engineering",
    "llm_ft": "LLM fine-tuning (LoRA/PEFT)",
    "ltr": "learning-to-rank",
    "hrtech": "HR-tech/marketplace",
    "scale": "large-scale inference",
    "oss": "open-source ML",
}


def _matched_skills(c: Candidate, aspect: dict, limit: int = 3) -> list[str]:
    names = [s.get("name", "") for s in c.skills]
    low = {n.lower(): n for n in names}
    out = []
    for term in aspect["skills"]:
        for ln, orig in low.items():
            if term.lower() in ln and orig not in out:
                out.append(orig)
                break
        if len(out) >= limit:
            break
    return out


def _best_aspect(f: dict, aspects: dict):
    from features import MUST_SLOTS, ordered_aspects

    best, best_fit, best_k = None, -1.0, 0
    for k, a in enumerate(ordered_aspects(aspects, "must_have")[:MUST_SLOTS], 1):
        fit = f.get(f"sem_must_{k}", 0.0) + 0.3 * f.get(f"hit_must_{k}", 0.0) \
            + 0.2 * f.get(f"cev_must_{k}", 0.0)
        if fit > best_fit:
            best, best_fit, best_k = a, fit, k
    return best, best_k


def _top_concern(c: Candidate, f: dict) -> str | None:
    p = c.profile
    if f.get("non_eng_title", 0.0) >= 1.0:
        return f"current role is {p.get('current_title')}, outside core AI engineering"
    if f.get("all_consulting", 0.0) >= 1.0:
        return "career entirely at IT-services firms"
    if f.get("keyword_stuffer", 0.0) >= 1.0:
        return "lists AI skills but little hands-on systems evidence"
    if f.get("research_only", 0.0) >= 1.0:
        return "research-leaning profile with limited production signals"
    if f.get("wrong_domain", 0.0) >= 1.0:
        return "primarily CV/speech background rather than IR/retrieval"
    if f.get("managerial_title", 0.0) >= 1.0:
        return "currently in a management title (coding recency unclear)"
    if f.get("job_hopper", 0.0) >= 1.0:
        return f"frequent short stints (avg ~{f.get('avg_tenure_months', 0):.0f}mo)"
    if f.get("days_inactive", 0.0) > 120:
        return f"inactive ~{f.get('days_inactive', 0):.0f} days"
    if f.get("recruiter_response_rate", 1.0) < 0.3:
        return f"low recruiter response rate ({f.get('recruiter_response_rate', 0):.2f})"
    if f.get("notice_period_days", 0.0) > 75:
        return f"long notice ({f.get('notice_period_days', 0):.0f}d)"
    if f.get("seniority_band", 1.0) < 0.8:
        return f"experience ({f.get('yoe', 0):.1f}y) outside the ideal 6-8y band"
    if f.get("frac_consulting", 0.0) > 0.4:
        return "significant time at IT-services firms"
    if f.get("location_fit", 1.0) < 0.5 and f.get("in_india", 0.0) < 1.0:
        return f"based outside India ({p.get('location')})"
    return None


def reason_for(c: Candidate, f: dict, aspects: dict, rank: int) -> str:
    p = c.profile
    yoe = p.get("years_of_experience")
    title = p.get("current_title")
    company = p.get("current_company")

    # Specific fact slot (always grounded).
    fact = f"{yoe:.1f}y exp; {title} @ {company}"

    # JD-connection slot.
    a, k = _best_aspect(f, aspects)
    conn = ""
    if a is not None:
        sk = _matched_skills(c, a)
        phrase = _ASPECT_PHRASE.get(a["id"], a["name"])
        if sk:
            conn = f"matches {phrase} ({', '.join(sk)})"
        elif f.get(f"cev_must_{k}", 0.0) >= 1.0:
            conn = f"career history shows {phrase}"
        else:
            conn = f"semantic fit on {phrase}"

    concern = _top_concern(c, f)

    # Tone varies by rank band.
    if rank <= 10:
        lead = "Strong fit:"
    elif rank <= 50:
        lead = "Solid fit:"
    else:
        lead = "Borderline:"

    parts = [f"{lead} {fact}"]
    if conn:
        parts.append(conn)
    if concern:
        parts.append(f"concern: {concern}")
    text = "; ".join(parts) + "."
    # keep pure-ASCII to avoid any CSV encoding surprises downstream
    return text.encode("ascii", "ignore").decode("ascii")[:240]
