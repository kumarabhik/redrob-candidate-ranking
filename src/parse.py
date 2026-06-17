"""Candidate data layer: streaming loader, typed record, profile-text builder, schema check.

The candidates file is ~465 MB / 100k lines of JSONL. We stream it line-by-line and never
call json.load on the whole file (agents.md §7). Typed access goes through Candidate, which
wraps the raw dict so feature/honeypot code uses verbatim schema names (agents.md §3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from config import CANDIDATES_JSONL, SCHEMA


# --- Typed record --------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """Thin typed wrapper over a raw candidate dict. Accessors mirror the schema."""

    raw: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return self.raw["candidate_id"]

    @property
    def profile(self) -> dict[str, Any]:
        return self.raw["profile"]

    @property
    def career_history(self) -> list[dict[str, Any]]:
        return self.raw["career_history"]

    @property
    def education(self) -> list[dict[str, Any]]:
        return self.raw.get("education", [])

    @property
    def skills(self) -> list[dict[str, Any]]:
        return self.raw.get("skills", [])

    @property
    def certifications(self) -> list[dict[str, Any]]:
        return self.raw.get("certifications", [])

    @property
    def languages(self) -> list[dict[str, Any]]:
        return self.raw.get("languages", [])

    @property
    def signals(self) -> dict[str, Any]:
        return self.raw["redrob_signals"]

    @property
    def years_of_experience(self) -> float:
        return float(self.profile["years_of_experience"])

    def profile_text(self) -> str:
        """Concatenated free-text used for BM25 + dense retrieval (Stage A).

        Headline + summary + current title/industry + each role's title/company/description.
        Deterministic and self-contained: no external lookups.
        """
        p = self.profile
        parts: list[str] = [
            p.get("headline", ""),
            p.get("summary", ""),
            p.get("current_title", ""),
            p.get("current_industry", ""),
        ]
        for role in self.career_history:
            parts.append(role.get("title", ""))
            parts.append(role.get("company", ""))
            parts.append(role.get("industry", ""))
            parts.append(role.get("description", ""))
        for sk in self.skills:
            parts.append(sk.get("name", ""))
        return " ".join(part for part in parts if part).strip()


# --- Streaming loaders ---------------------------------------------------
def stream_raw(path: Path = CANDIDATES_JSONL) -> Iterator[dict[str, Any]]:
    """Yield one raw candidate dict per JSONL line. Constant memory."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_candidates(path: Path = CANDIDATES_JSONL) -> Iterator[Candidate]:
    """Yield typed Candidate objects, streaming."""
    for raw in stream_raw(path):
        yield Candidate(raw)


def count_candidates(path: Path = CANDIDATES_JSONL) -> int:
    """Count records without holding them all in memory."""
    n = 0
    for _ in stream_raw(path):
        n += 1
    return n


# --- Schema validation ---------------------------------------------------
def validate_sample(n: int = 500, path: Path = CANDIDATES_JSONL) -> tuple[int, list[str]]:
    """Validate the first `n` records against candidate_schema.json.

    Full-pool validation is expensive; we validate a head sample as a structural
    sanity check (Phase 0 / roadmap verification matrix). Returns (checked, errors).
    """
    import jsonschema  # local import: only needed for validation

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    errors: list[str] = []
    checked = 0
    for raw in stream_raw(path):
        if checked >= n:
            break
        checked += 1
        for err in validator.iter_errors(raw):
            cid = raw.get("candidate_id", "<no id>")
            errors.append(f"{cid}: {err.message} (at {list(err.path)})")
    return checked, errors


if __name__ == "__main__":
    import sys

    print("Counting candidates (streaming)...", flush=True)
    total = count_candidates()
    print(f"records: {total:,}")

    checked, errs = validate_sample()
    print(f"schema-validated head sample: {checked} records, {len(errs)} errors")
    for e in errs[:10]:
        print("  ERR", e)

    if total != 100_000:
        print(f"WARNING: expected 100,000 records, got {total:,}")
        sys.exit(1)
    if errs:
        sys.exit(1)
    print("OK: 100,000 records, sample passes schema.")
