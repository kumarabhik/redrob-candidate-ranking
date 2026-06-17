"""Unit tests for feature extraction + rubric. Run: python tests/test_features.py

Uses the real sample_candidates.json plus targeted synthetic profiles to assert the trap
detectors fire (wrong-role title, consulting-only, keyword-stuffer) and that a clean strong
profile scores above an obvious non-fit under the rubric.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import feature_names, features_for, vectorize  # noqa: E402
from jd_aspects import ASPECTS  # noqa: E402
from labels import relevance_score  # noqa: E402
from parse import Candidate  # noqa: E402

ASPECTS = dict(ASPECTS)
ASPECTS["_query_order"] = [a["id"] for a in ASPECTS["must_have"]] + \
                          [a["id"] for a in ASPECTS["nice_to_have"]]
QORDER = ASPECTS["_query_order"]
NAMES = feature_names(ASPECTS)
ZERO_SIMS = np.zeros(len(QORDER), dtype=np.float32)


def strong_raw() -> dict:
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "A", "headline": "Senior ML Engineer | retrieval & ranking",
            "summary": "Built production retrieval and ranking systems at a product company.",
            "location": "Pune", "country": "India", "years_of_experience": 7.0,
            "current_title": "Senior ML Engineer", "current_company": "ProductCo",
            "current_company_size": "201-500", "current_industry": "Software",
        },
        "career_history": [{
            "company": "ProductCo", "title": "Senior ML Engineer", "start_date": "2021-01-01",
            "end_date": None, "duration_months": 65, "is_current": True, "industry": "Software",
            "company_size": "201-500",
            "description": "Built embeddings-based retrieval and a learning-to-rank "
                           "recommendation system deployed to real users; ran A/B tests, NDCG.",
        }],
        "education": [{"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
                       "start_year": 2011, "end_year": 2015, "grade": None, "tier": "tier_1"}],
        "skills": [
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 60},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 40},
            {"name": "Learning to Rank", "proficiency": "advanced", "endorsements": 8, "duration_months": 30},
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 84},
        ],
        "redrob_signals": _good_signals(),
    }


def _good_signals() -> dict:
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


def _feats(raw):
    return features_for(Candidate(raw), ZERO_SIMS, QORDER, ASPECTS)


def test_strong_profile_basic():
    f = _feats(strong_raw())
    assert f["seniority_band"] == 1.0
    assert f["systems_evidence"] == 1.0
    assert f["non_eng_title"] == 0.0
    assert f["availability"] > 0.8


def test_marketing_manager_trap():
    r = strong_raw()
    r["profile"]["current_title"] = "Marketing Manager"
    f = _feats(r)
    assert f["non_eng_title"] == 1.0
    # rubric should crush it relative to the engineer
    assert relevance_score(f, ASPECTS) < relevance_score(_feats(strong_raw()), ASPECTS)


def test_consulting_only_penalized():
    r = strong_raw()
    for role in r["career_history"]:
        role["company"] = "Infosys"
    r["profile"]["current_company"] = "Infosys"
    f = _feats(r)
    assert f["all_consulting"] == 1.0
    assert relevance_score(f, ASPECTS) < relevance_score(_feats(strong_raw()), ASPECTS)


def test_keyword_stuffer_detected():
    r = strong_raw()
    # strip real systems evidence, pile on buzzword skills with tiny durations
    r["career_history"][0]["description"] = "Did various tasks and coordination."
    r["profile"]["summary"] = "Generalist."
    r["skills"] = [
        {"name": n, "proficiency": "expert", "endorsements": 1, "duration_months": 2}
        for n in ["RAG", "LangChain", "LLMs", "Pinecone", "Embeddings", "Vector Search"]
    ]
    f = _feats(r)
    assert f["keyword_stuffer"] == 1.0
    assert f["systems_evidence"] == 0.0


def test_vectorize_length():
    f = _feats(strong_raw())
    v = vectorize(f, NAMES)
    assert v.shape == (len(NAMES),)
    assert np.isfinite(v).all()


def test_sample_candidates_parse():
    sample = json.loads((ROOT / "[PUB] India_runs_data_and_ai_challenge" /
                         "India_runs_data_and_ai_challenge" / "sample_candidates.json")
                        .read_text(encoding="utf-8"))
    assert len(sample) > 0
    for raw in sample[:20]:
        f = features_for(Candidate(raw), ZERO_SIMS, QORDER, ASPECTS)
        v = vectorize(f, NAMES)
        assert np.isfinite(v).all()
        assert 0.0 <= relevance_score(f, ASPECTS) <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
