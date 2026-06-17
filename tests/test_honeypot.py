"""Unit tests for honeypot detection. Run: python tests/test_honeypot.py

Builds minimal valid candidates and mutates one field at a time to assert each
hard rule fires (and that a clean profile does not).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from honeypot import coded_reasons, honeypot_reasons, is_honeypot  # noqa: E402
from parse import Candidate  # noqa: E402


def _codes(raw: dict) -> set[str]:
    return {code for code, _ in coded_reasons(Candidate(raw))}


def clean_raw() -> dict:
    """A clean, internally consistent candidate (snapshot = 2026-06-15)."""
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "ML Engineer",
            "summary": "Builds retrieval systems.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_company": "Acme",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "Acme",
                "title": "ML Engineer",
                "start_date": "2024-03-01",
                "end_date": None,
                "duration_months": 27,  # 2024-03 -> 2026-06
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Built hybrid retrieval and ranking.",
            },
            {
                "company": "OldCo",
                "title": "Data Scientist",
                "start_date": "2019-01-01",
                "end_date": "2024-01-01",
                "duration_months": 60,
                "is_current": False,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Recommender systems.",
            },
        ],
        "education": [
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2015, "end_year": 2019, "grade": None, "tier": "tier_1"},
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 80},
            {"name": "NLP", "proficiency": "advanced", "endorsements": 5, "duration_months": 40},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90.0, "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 10, "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.6, "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {}, "connection_count": 100,
            "endorsements_received": 15, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 50.0, "search_appearance_30d": 30,
            "saved_by_recruiters_30d": 3, "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }


def test_clean_is_not_honeypot():
    assert not is_honeypot(Candidate(clean_raw())), honeypot_reasons(Candidate(clean_raw()))


def test_single_expert_zero_is_soft_not_honeypot():
    # One expert-with-0-months is common noise -> soft signal, NOT a hard honeypot.
    r = clean_raw()
    r["skills"][0]["duration_months"] = 0  # Python expert, 0 months
    assert not is_honeypot(Candidate(r))
    assert "EXPERT_ZERO" in _codes(r)


def test_multi_expert_zero_is_honeypot():
    # >=3 expert-with-0-months simultaneously is the deliberate honeypot signature.
    r = clean_raw()
    for name in ["A", "B", "C"]:
        r["skills"].append(
            {"name": name, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        )
    assert is_honeypot(Candidate(r))
    assert "MULTI_EXPERT_ZERO" in _codes(r)


def test_skill_exceeds_career_is_soft():
    # Skill duration > total career is noisy in this dataset -> soft, not hard.
    r = clean_raw()
    r["skills"][1]["duration_months"] = 200  # 7yr career = 84mo
    assert not is_honeypot(Candidate(r))
    assert "SKILL_GT_CAREER" in _codes(r)


def test_is_current_with_end_date():
    r = clean_raw()
    r["career_history"][0]["end_date"] = "2025-01-01"  # current but ended
    assert is_honeypot(Candidate(r))


def test_duration_overclaim_is_honeypot():
    # Claiming far MORE tenure than the dates allow (the "8yr at a 3yr-old co" pattern).
    r = clean_raw()
    r["career_history"][1]["duration_months"] = 200  # dates span 60mo, claims 200
    assert is_honeypot(Candidate(r))


def test_duration_underclaim_is_not_honeypot():
    # Claiming fewer months than the span is a benign gap, not an impossibility.
    r = clean_raw()
    r["career_history"][1]["duration_months"] = 8  # dates span 60mo, claims 8
    assert not is_honeypot(Candidate(r))


def test_end_before_start():
    r = clean_raw()
    r["career_history"][1]["start_date"] = "2024-01-01"
    r["career_history"][1]["end_date"] = "2019-01-01"
    assert is_honeypot(Candidate(r))


def test_salary_inverted_is_soft():
    # ~19% of the pool has min>max -> soft signal only, never a hard honeypot.
    r = clean_raw()
    r["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 50.0, "max": 10.0}
    assert not is_honeypot(Candidate(r))
    assert "SALARY_INV" in _codes(r)


def test_education_years_inverted():
    r = clean_raw()
    r["education"][0]["start_year"] = 2019
    r["education"][0]["end_year"] = 2015
    assert is_honeypot(Candidate(r))


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
