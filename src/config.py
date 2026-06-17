"""Central config: paths and tunables. No literals scattered across modules (agents.md §3)."""
from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
CANDIDATES_JSONL = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES = DATA_DIR / "sample_candidates.json"
SCHEMA = DATA_DIR / "candidate_schema.json"
ARTIFACTS = ROOT / "artifacts"

EXPECTED_CANDIDATE_COUNT = 100_000

# --- "Today" for tenure/recency math -------------------------------------
# The dataset is a June-2026 snapshot (signup/last_active dates run up to 2026).
SNAPSHOT_DATE = "2026-06-15"

# --- Honeypot tolerances (see design_doc.md §7.3) ------------------------
# duration_months vs (end-start): allow rounding/partial-month slack.
DURATION_TOLERANCE_MONTHS = 2
# A skill's claimed usage cannot exceed total career by more than this slack.
SKILL_VS_CAREER_SLACK_MONTHS = 6
