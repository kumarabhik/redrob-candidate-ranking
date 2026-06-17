"""Evaluation harness (design_doc.md §9): the competition's own metrics + a trap probe.

Two things live here:
  1. Metric functions (NDCG@k, MAP, P@k) -- the exact composite the hackathon scores on,
     reusable by ablation.py and any offline tuning.
  2. A crafted "trap probe": one ideal candidate + one of each documented trap type
     (Marketing-Manager-with-AI-skills, keyword-stuffer, consulting-only, honeypot, junior),
     with known-correct relevance. We assert the rubric AND the trained model order the ideal
     above every trap. This is a behavioral guarantee on the exact failure modes the JD warns
     about -- defensible in the Stage-5 interview.

No hidden ground truth exists, so the probe uses constructed-correct labels; real-pool
quality is examined comparatively in ablation.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jd_aspects import ASPECTS as _RAW_ASPECTS  # noqa: E402


# --- metrics -------------------------------------------------------------
def dcg(rels: list[float]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: list[float], k: int) -> float:
    ideal = sorted(ranked_rels, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(ranked_rels[:k]) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_rels: list[float], k: int, thr: float = 3.0) -> float:
    top = ranked_rels[:k]
    return sum(1 for r in top if r >= thr) / k if k else 0.0


def average_precision(ranked_rels: list[float], thr: float = 3.0) -> float:
    hits, s = 0, 0.0
    for i, r in enumerate(ranked_rels, 1):
        if r >= thr:
            hits += 1
            s += hits / i
    total_rel = sum(1 for r in ranked_rels if r >= thr)
    return s / total_rel if total_rel else 0.0


def composite(ranked_rels: list[float]) -> dict:
    """The hackathon's weighted composite + components."""
    m = {
        "ndcg@10": ndcg_at_k(ranked_rels, 10),
        "ndcg@50": ndcg_at_k(ranked_rels, 50),
        "map": average_precision(ranked_rels),
        "p@10": precision_at_k(ranked_rels, 10),
    }
    m["composite"] = (0.50 * m["ndcg@10"] + 0.30 * m["ndcg@50"]
                      + 0.15 * m["map"] + 0.05 * m["p@10"])
    return m


# --- crafted trap probe --------------------------------------------------
def _aspects():
    a = dict(_RAW_ASPECTS)
    a["_query_order"] = [x["id"] for x in a["must_have"]] + [x["id"] for x in a["nice_to_have"]]
    return a


def _good_signals():
    return {
        "profile_completeness_score": 95.0, "signup_date": "2024-01-01",
        "last_active_date": "2026-06-10", "open_to_work_flag": True,
        "profile_views_received_30d": 40, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.8, "avg_response_time_hours": 5.0,
        "skill_assessment_scores": {}, "connection_count": 300, "endorsements_received": 60,
        "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 30.0, "max": 55.0},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 70.0, "search_appearance_30d": 80,
        "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.6, "verified_email": True, "verified_phone": True,
        "linkedin_connected": True,
    }


def _role(company, title, start, end, dur, cur, desc, size="201-500", ind="Software"):
    return {"company": company, "title": title, "start_date": start, "end_date": end,
            "duration_months": dur, "is_current": cur, "industry": ind,
            "company_size": size, "description": desc}


def _skill(name, prof="advanced", end=10, dur=36):
    return {"name": name, "proficiency": prof, "endorsements": end, "duration_months": dur}


def probe_candidates() -> list[tuple[str, int, dict]]:
    """(label, expected_tier, raw) for one ideal + each documented trap."""
    base_edu = [{"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
                 "start_year": 2011, "end_year": 2015, "grade": None, "tier": "tier_1"}]

    ideal = {
        "candidate_id": "CAND_9000001",
        "profile": {"anonymized_name": "Ideal", "headline": "Senior ML Engineer | retrieval & ranking",
                    "summary": "Built production embeddings retrieval and a learning-to-rank "
                               "recommendation system deployed to real users at a product company.",
                    "location": "Pune", "country": "India", "years_of_experience": 7.0,
                    "current_title": "Senior ML Engineer", "current_company": "ProductCo",
                    "current_company_size": "201-500", "current_industry": "Software"},
        "career_history": [_role("ProductCo", "Senior ML Engineer", "2021-01-01", None, 65, True,
                                 "Built embeddings retrieval and learning-to-rank recommender "
                                 "deployed to real users; NDCG, A/B tests, vector search FAISS.")],
        "education": base_edu,
        "skills": [_skill("Embeddings", "expert", 20, 60), _skill("FAISS"),
                   _skill("Learning to Rank"), _skill("Recommendation Systems"),
                   _skill("Python", "expert", 30, 84), _skill("Information Retrieval")],
        "redrob_signals": _good_signals(),
    }

    marketing = {k: (v.copy() if isinstance(v, dict) else v) for k, v in ideal.items()}
    marketing["candidate_id"] = "CAND_9000002"
    marketing["profile"] = dict(ideal["profile"], current_title="Marketing Manager",
                                headline="Marketing Manager | growth",
                                summary="Led marketing campaigns and brand strategy.")
    marketing["career_history"] = [_role("BrandCo", "Marketing Manager", "2018-01-01", None, 90,
                                         True, "Ran marketing campaigns and content strategy.",
                                         ind="Marketing")]

    stuffer = dict(ideal, candidate_id="CAND_9000003")
    stuffer["profile"] = dict(ideal["profile"], summary="Generalist.")
    stuffer["career_history"] = [_role("SomeCo", "Software Engineer", "2018-01-01", None, 90,
                                       True, "Worked on various tasks and coordination.")]
    stuffer["skills"] = [_skill(n, "expert", 1, 2) for n in
                         ["RAG", "LangChain", "LLMs", "Pinecone", "Embeddings", "Vector Search"]]

    consulting = dict(ideal, candidate_id="CAND_9000004")
    consulting["profile"] = dict(ideal["profile"], current_company="Infosys",
                                 current_industry="IT Services")
    consulting["career_history"] = [_role("Infosys", "Software Engineer", "2017-01-01", None, 100,
                                          True, "Delivered client projects and support.",
                                          ind="IT Services")]

    honeypot = dict(ideal, candidate_id="CAND_9000005")
    honeypot["career_history"] = [_role("ProductCo", "ML Engineer", "2023-01-01", None, 160, True,
                                        "Built retrieval systems.")]  # 160mo in ~41mo span

    junior = dict(ideal, candidate_id="CAND_9000006")
    junior["profile"] = dict(ideal["profile"], years_of_experience=1.5,
                             current_title="Junior ML Engineer")

    return [("ideal", 4, ideal), ("marketing_trap", 0, marketing),
            ("keyword_stuffer", 0, stuffer), ("consulting_only", 1, consulting),
            ("honeypot", 0, honeypot), ("junior", 1, junior)]


def run_probe() -> int:
    from features import feature_names, features_for, vectorize
    from honeypot import is_honeypot
    from labels import relevance_score
    from parse import Candidate

    aspects = _aspects()
    qorder = aspects["_query_order"]
    names = feature_names(aspects)
    zero = np.zeros(len(qorder), dtype=np.float32)

    # try to load the trained model too
    model = None
    try:
        import xgboost as xgb
        from config import ARTIFACTS
        model = xgb.XGBRanker()
        model.load_model(str(ARTIFACTS / "model.bin"))
    except Exception as e:  # noqa: BLE001
        print(f"(model not loaded: {e}; probing rubric only)")

    rows = []
    for label, exp_tier, raw in probe_candidates():
        c = Candidate(raw)
        f = features_for(c, zero, qorder, aspects)
        hp = is_honeypot(c)
        rub = relevance_score(f, aspects, is_honeypot=hp)
        mscore = float(model.predict(vectorize(f, names).reshape(1, -1))[0]) if model else rub
        rows.append((label, exp_tier, rub, mscore, hp))

    print(f"{'candidate':16} {'exp_tier':8} {'rubric':>7} {'model':>9} honeypot")
    for label, exp, rub, ms, hp in rows:
        print(f"{label:16} {exp:^8} {rub:7.3f} {ms:9.3f}  {hp}")

    ideal = next(r for r in rows if r[0] == "ideal")
    failures = []
    for label, exp, rub, ms, hp in rows:
        if label == "ideal":
            continue
        if not (ideal[2] > rub):
            failures.append(f"rubric: ideal !> {label}")
        if model and not (ideal[3] > ms):
            failures.append(f"model: ideal !> {label}")
    # honeypot must be detected
    if not next(r for r in rows if r[0] == "honeypot")[4]:
        failures.append("honeypot not detected")

    print()
    if failures:
        for fmsg in failures:
            print("FAIL", fmsg)
        return 1
    print("PROBE PASS: ideal ranks above every trap (rubric"
          + (" + model" if model else "") + "); honeypot detected.")
    return 0


if __name__ == "__main__":
    sys.exit(run_probe())
