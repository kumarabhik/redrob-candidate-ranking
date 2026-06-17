"""Stage B -- per-candidate features (Pillar 2: structured person-job fit).

Every feature is derived from verbatim schema fields + the precomputed dense aspect
similarities. Features fall into groups (design_doc.md §7.2): aspect-fit (semantic + lexical
+ career evidence), seniority band, product-vs-services, behavioral availability, location,
credibility, and trap detectors (wrong-role title, consulting-only, keyword-stuffer,
job-hopper, research-only, managerial-not-coding). The LambdaMART reranker learns how to
weight them; the hand-built rubric in labels.py provides weak training targets.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from config import SNAPSHOT_DATE
from honeypot import coded_reasons
from parse import Candidate

_SNAP = date.fromisoformat(SNAPSHOT_DATE)

# Buzzword vocab = union of all aspect skills + common AI hype terms (for stuffer detection).
_AI_BUZZWORDS = {
    "rag", "llm", "llms", "langchain", "llamaindex", "embeddings", "vector", "pinecone",
    "qdrant", "milvus", "weaviate", "faiss", "pgvector", "lora", "qlora", "peft", "nlp",
    "transformers", "fine-tuning", "prompt", "semantic search", "bm25", "bge", "e5",
    "machine learning", "deep learning", "pytorch", "tensorflow", "huggingface",
    "recommendation", "ranking", "retrieval", "mlops", "diffusion",
}
# Strong evidence that someone actually BUILT ranking/search/recsys (career descriptions).
_SYSTEMS_EVIDENCE = ["recommend", "ranking", "retrieval", "search", "embedding", "relevance",
                     "personaliz", "recsys", "vector", "semantic", "information retrieval"]
_PRODUCTION_EVIDENCE = ["production", "deployed", "shipped", "built", "scaled", "launched",
                        "real users", "at scale", "serving", "latency"]
_MANAGERIAL = ["engineering manager", "director", "vp ", "vice president", "head of", "cto",
               "chief", " gm", "general manager"]


def _safe(v, default=0.0):
    return float(v) if isinstance(v, (int, float)) else default


# Fixed slot counts: feature schema is JD-agnostic (model transfers across roles).
MUST_SLOTS = 4
NICE_SLOTS = 5


def ordered_aspects(aspects: dict, key: str) -> list:
    """Aspects of `key` sorted by descending weight then id (stable slot assignment)."""
    return sorted(aspects.get(key, []), key=lambda a: (-a.get("weight", 0.0), a["id"]))


def candidate_text(c: Candidate) -> str:
    p = c.profile
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", ""),
             p.get("current_industry", "")]
    for r in c.career_history:
        parts += [r.get("title", ""), r.get("description", ""), r.get("industry", "")]
    return " ".join(parts).lower()


def career_text(c: Candidate) -> str:
    return " ".join(r.get("title", "") + " " + r.get("description", "")
                    for r in c.career_history).lower()


def skill_names(c: Candidate) -> list[str]:
    return [s.get("name", "").lower() for s in c.skills]


def _aspect_lexical(aspect: dict, skills_lc: list[str], ctext: str, cartext: str) -> tuple:
    """Return (skill_hits, career_evidence) for an aspect."""
    hits = 0
    career_ev = 0
    for term in aspect["skills"]:
        t = term.lower()
        if any(t in s for s in skills_lc):
            hits += 1
        if t in cartext:
            career_ev = 1
        elif t in ctext:
            career_ev = max(career_ev, 0)
    return hits, career_ev


def features_for(c: Candidate, sims_row: np.ndarray, query_order: list[str],
                 aspects: dict) -> dict[str, float]:
    f: dict[str, float] = {}
    p = c.profile
    sig = c.signals
    skills_lc = skill_names(c)
    ctext = candidate_text(c)
    cartext = career_text(c)
    cur_title = p.get("current_title", "").lower()
    qpos = {qid: i for i, qid in enumerate(query_order)}

    # --- Aspect fit: semantic (dense) + lexical + career evidence ---------
    # Fixed aspect SLOTS (ordered by weight), not aspect ids, so the feature schema is
    # identical for ANY JD and the trained model transfers across roles (see serve.py).
    must_sem = []
    for k, a in enumerate(ordered_aspects(aspects, "must_have")[:MUST_SLOTS], 1):
        sem = float(sims_row[qpos[a["id"]]])
        hits, cev = _aspect_lexical(a, skills_lc, ctext, cartext)
        f[f"sem_must_{k}"] = sem
        f[f"hit_must_{k}"] = float(hits)
        f[f"cev_must_{k}"] = float(cev)
        must_sem.append(sem)
    for k in range(len(must_sem) + 1, MUST_SLOTS + 1):     # pad empty slots
        f[f"sem_must_{k}"] = f[f"hit_must_{k}"] = f[f"cev_must_{k}"] = 0.0
    f["sem_must_mean"] = float(np.mean(must_sem)) if must_sem else 0.0
    f["sem_must_max"] = float(np.max(must_sem)) if must_sem else 0.0
    for k, a in enumerate(ordered_aspects(aspects, "nice_to_have")[:NICE_SLOTS], 1):
        f[f"sem_nice_{k}"] = float(sims_row[qpos[a["id"]]])
        hits, _ = _aspect_lexical(a, skills_lc, ctext, cartext)
        f[f"hit_nice_{k}"] = float(hits)
    for k in range(len(aspects["nice_to_have"]) + 1, NICE_SLOTS + 1):
        f[f"sem_nice_{k}"] = f[f"hit_nice_{k}"] = 0.0

    # --- Seniority band ---------------------------------------------------
    yoe = c.years_of_experience
    ex = aspects["experience"]
    f["yoe"] = yoe
    if ex["ideal_min"] <= yoe <= ex["ideal_max"]:
        band = 1.0
    elif ex["min"] <= yoe <= ex["max"]:
        band = 0.8
    elif yoe < ex["hard_floor"]:
        band = 0.1
    else:
        # linear falloff outside the broad band
        d = min(abs(yoe - ex["min"]), abs(yoe - ex["max"]))
        band = max(0.2, 0.8 - 0.1 * d)
    f["seniority_band"] = band

    # --- Career structure / job-hopping ----------------------------------
    durs = [_safe(r.get("duration_months")) for r in c.career_history]
    n_roles = len(c.career_history)
    f["n_roles"] = float(n_roles)
    f["avg_tenure_months"] = float(np.mean(durs)) if durs else 0.0
    f["max_tenure_months"] = float(np.max(durs)) if durs else 0.0
    short = sum(1 for d in durs if 0 < d < 18)
    f["short_stints"] = float(short)
    f["job_hopper"] = 1.0 if (n_roles >= 4 and f["avg_tenure_months"] < 18) else 0.0

    # --- Product vs services / consulting ---------------------------------
    firms = aspects["disqualifiers"]["consulting_firms"]
    companies = [r.get("company", "").lower() for r in c.career_history]
    cur_company = p.get("current_company", "").lower()
    cons = [any(fm in co for fm in firms) for co in companies]
    f["cur_in_consulting"] = 1.0 if any(fm in cur_company for fm in firms) else 0.0
    f["frac_consulting"] = float(np.mean(cons)) if cons else 0.0
    f["all_consulting"] = 1.0 if (companies and all(cons)) else 0.0
    cur_industry = p.get("current_industry", "").lower()
    f["it_services_industry"] = 1.0 if ("it services" in cur_industry or
                                        "consulting" in cur_industry) else 0.0

    # --- Trap detectors ---------------------------------------------------
    nf = aspects["disqualifiers"]
    f["non_eng_title"] = 1.0 if any(t in cur_title for t in nf["non_engineering_titles"]) else 0.0
    frac_non_eng = np.mean([any(t in (r.get("title", "").lower()) for t in
                            nf["non_engineering_titles"]) for r in c.career_history]) \
        if c.career_history else 0.0
    f["frac_non_eng_titles"] = float(frac_non_eng)
    f["research_only"] = 1.0 if (any(t in ctext for t in nf["research_terms"]) and
                                 not any(t in cartext for t in _PRODUCTION_EVIDENCE)) else 0.0
    wrong = sum(1 for t in nf["wrong_domain_terms"] if t in ctext)
    has_systems = any(t in cartext for t in _SYSTEMS_EVIDENCE)
    f["wrong_domain"] = 1.0 if (wrong >= 3 and not has_systems) else 0.0
    f["managerial_title"] = 1.0 if any(t in cur_title for t in _MANAGERIAL) else 0.0

    # Keyword-stuffer: many AI buzzwords listed, but no real systems evidence in career
    # and weak skill durations -> classic "perfect skill list, wrong substance".
    buzz = sum(1 for s in skills_lc if any(b in s for b in _AI_BUZZWORDS))
    ai_durs = [_safe(s.get("duration_months")) for s in c.skills
               if any(b in s.get("name", "").lower() for b in _AI_BUZZWORDS)]
    f["n_ai_skills"] = float(buzz)
    f["systems_evidence"] = 1.0 if has_systems else 0.0
    f["production_evidence"] = 1.0 if any(t in cartext for t in _PRODUCTION_EVIDENCE) else 0.0
    f["keyword_stuffer"] = 1.0 if (buzz >= 5 and not has_systems and
                                   (np.mean(ai_durs) if ai_durs else 0) < 12) else 0.0

    # --- Behavioral availability ------------------------------------------
    f["recruiter_response_rate"] = _safe(sig.get("recruiter_response_rate"))
    f["open_to_work"] = 1.0 if sig.get("open_to_work_flag") else 0.0
    f["interview_completion_rate"] = _safe(sig.get("interview_completion_rate"))
    oar = _safe(sig.get("offer_acceptance_rate"), -1.0)
    f["offer_acceptance_rate"] = oar if oar >= 0 else 0.5  # impute neutral if no history
    notice = _safe(sig.get("notice_period_days"), 90)
    f["notice_period_days"] = notice
    last = sig.get("last_active_date")
    try:
        days_inactive = (_SNAP - date.fromisoformat(last)).days if last else 365
    except (ValueError, TypeError):
        days_inactive = 365
    f["days_inactive"] = float(days_inactive)
    gh = _safe(sig.get("github_activity_score"), -1.0)
    f["github_activity"] = gh if gh >= 0 else 0.0
    f["github_linked"] = 1.0 if gh >= 0 else 0.0

    # demand / credibility
    f["saved_by_recruiters_30d"] = _safe(sig.get("saved_by_recruiters_30d"))
    f["search_appearance_30d"] = _safe(sig.get("search_appearance_30d"))
    f["profile_views_30d"] = _safe(sig.get("profile_views_received_30d"))
    f["profile_completeness"] = _safe(sig.get("profile_completeness_score"))
    f["verified"] = (1.0 if sig.get("verified_email") else 0.0) + \
                    (1.0 if sig.get("verified_phone") else 0.0)
    f["linkedin_connected"] = 1.0 if sig.get("linkedin_connected") else 0.0

    # composite availability multiplier (also used by labels.py rubric)
    avail = 1.0
    avail *= 1.0 if f["recruiter_response_rate"] >= 0.4 else 0.6 + f["recruiter_response_rate"]
    avail *= 1.0 if days_inactive <= 120 else max(0.4, 1.0 - (days_inactive - 120) / 365)
    avail *= 1.0 if f["open_to_work"] else 0.8
    avail *= 1.0 if notice <= 30 else max(0.7, 1.0 - (notice - 30) / 300)
    f["availability"] = float(min(1.0, max(0.0, avail)))

    # --- Location fit -----------------------------------------------------
    loc = (p.get("location", "") + " " + p.get("country", "")).lower()
    in_india = "india" in loc or any(ci in loc for ci in aspects["locations_preferred"])
    pref_city = any(ci in loc for ci in aspects["locations_preferred"])
    relocate = bool(sig.get("willing_to_relocate"))
    f["in_india"] = 1.0 if in_india else 0.0
    f["pref_city"] = 1.0 if pref_city else 0.0
    f["willing_relocate"] = 1.0 if relocate else 0.0
    f["location_fit"] = (1.0 if pref_city else 0.7 if (in_india and relocate)
                         else 0.5 if in_india else 0.25 if relocate else 0.1)

    # --- Enriched signals: depth, verification, credibility --------------
    ai_skill_objs = [s for s in c.skills
                     if any(b in s.get("name", "").lower() for b in _AI_BUZZWORDS)]
    ai_durs2 = [_safe(s.get("duration_months")) for s in ai_skill_objs]
    f["ai_skill_depth"] = float(min(1.0, np.mean(ai_durs2) / 48.0)) if ai_durs2 else 0.0
    # endorsement trust: endorsements backed by real usage duration (vs stuffing)
    trust = [_safe(s.get("endorsements")) * min(1.0, _safe(s.get("duration_months")) / 12.0)
             for s in ai_skill_objs]
    f["endorsement_trust"] = float(min(1.0, np.mean(trust) / 20.0)) if trust else 0.0
    # platform skill assessments = verified competence
    sa = sig.get("skill_assessment_scores", {}) or {}
    ai_assess = [v for k, v in sa.items()
                 if any(b in k.lower() for b in _AI_BUZZWORDS) and isinstance(v, (int, float))]
    f["assessment_ai_mean"] = float(np.mean(ai_assess) / 100.0) if ai_assess else 0.0
    f["assessment_count"] = float(len(sa))
    # education prestige
    tier_map = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4, "tier_4": 0.2, "unknown": 0.3}
    edu_tiers = [tier_map.get(e.get("tier", "unknown"), 0.3) for e in c.education]
    f["edu_tier"] = float(max(edu_tiers)) if edu_tiers else 0.0
    # certifications
    certs = c.certifications
    f["n_certs"] = float(len(certs))
    f["ai_cert"] = 1.0 if any(
        any(t in (ct.get("name", "") + ct.get("issuer", "")).lower() for t in
            ["aws", "gcp", "google cloud", "azure", "machine learning", "deep learning",
             "tensorflow", "pytorch", "ml "]) for ct in certs) else 0.0
    # response speed (faster reply -> more reachable)
    art = _safe(sig.get("avg_response_time_hours"), 168.0)
    f["response_speed"] = float(1.0 / (1.0 + art / 24.0))
    # current role IS an engineering/ML role (positive counter-signal to non_eng_title)
    f["current_role_is_ai"] = 1.0 if any(t in cur_title for t in
        ["engineer", "scientist", " ml", "machine learning", " ai", "nlp", "data ",
         "research", "developer", "architect"]) else 0.0
    # recency-weighted systems experience: months in roles that show systems evidence
    sys_months = sum(_safe(r.get("duration_months")) for r in c.career_history
                     if any(t in (r.get("title", "") + " " + r.get("description", "")).lower()
                            for t in _SYSTEMS_EVIDENCE))
    f["systems_months"] = float(min(1.0, sys_months / 60.0))

    # --- Soft data-quality signals (from honeypot soft codes) -------------
    codes = {code for code, _ in coded_reasons(c)}
    f["soft_salary_inv"] = 1.0 if "SALARY_INV" in codes else 0.0
    f["soft_skill_gt_career"] = 1.0 if "SKILL_GT_CAREER" in codes else 0.0
    f["soft_expert_zero"] = 1.0 if "EXPERT_ZERO" in codes else 0.0

    return f


def feature_names(aspects: dict) -> list[str]:
    """Stable feature order = keys produced by features_for for a probe candidate."""
    # Build from the aspect config deterministically by running on a dummy is overkill;
    # instead reconstruct the known key order.
    names: list[str] = []
    for k in range(1, MUST_SLOTS + 1):
        names += [f"sem_must_{k}", f"hit_must_{k}", f"cev_must_{k}"]
    names += ["sem_must_mean", "sem_must_max"]
    for k in range(1, NICE_SLOTS + 1):
        names += [f"sem_nice_{k}", f"hit_nice_{k}"]
    names += ["yoe", "seniority_band", "n_roles", "avg_tenure_months", "max_tenure_months",
              "short_stints", "job_hopper", "cur_in_consulting", "frac_consulting",
              "all_consulting", "it_services_industry", "non_eng_title", "frac_non_eng_titles",
              "research_only", "wrong_domain", "managerial_title", "n_ai_skills",
              "systems_evidence", "production_evidence", "keyword_stuffer",
              "recruiter_response_rate", "open_to_work", "interview_completion_rate",
              "offer_acceptance_rate", "notice_period_days", "days_inactive",
              "github_activity", "github_linked", "saved_by_recruiters_30d",
              "search_appearance_30d", "profile_views_30d", "profile_completeness",
              "verified", "linkedin_connected", "availability", "in_india", "pref_city",
              "willing_relocate", "location_fit",
              "ai_skill_depth", "endorsement_trust", "assessment_ai_mean", "assessment_count",
              "edu_tier", "n_certs", "ai_cert", "response_speed", "current_role_is_ai",
              "systems_months",
              "soft_salary_inv", "soft_skill_gt_career", "soft_expert_zero"]
    return names


def vectorize(feats: dict[str, float], names: list[str]) -> np.ndarray:
    return np.array([feats.get(n, 0.0) for n in names], dtype=np.float32)
