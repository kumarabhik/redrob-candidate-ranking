# Design Doc — Intelligent Candidate Discovery & Ranking

## 0. Document control

| Field | Value |
|---|---|
| Title | Intelligent Candidate Discovery & Ranking — System Design |
| Project | INDIA.RUNS hackathon (Redrob × Hack2Skill) |
| Status | Draft v1 (docs-first; implementation not yet started) |
| Audience | Implementers, code reviewers, Stage-5 interview panel |
| Created | 2026-06-15 |
| Last updated | 2026-06-15 |
| Related docs | [roadmap.md](roadmap.md) · [agents.md](agents.md) · [system.md](system.md) · [explanation.md](explanation.md) |
| Source of truth for rules | `[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/submission_spec.docx`, `job_description.docx`, `redrob_signals_doc.docx`, `candidate_schema.json` |

> **Convention:** every section opens with its key decision in bold and is self-contained enough to be quoted in review.

---

## 1. Executive summary

**We rank the top 100 of 100,000 candidates against one job description using a three-stage pipeline — hybrid retrieval for recall, structured person-job-fit features for scoring, and a LambdaMART reranker that directly optimizes NDCG — with a rule-based honeypot filter and a deterministic, fact-grounded reasoning generator.** The ranking step runs in ≤5 minutes on CPU with no network, satisfying the hackathon's reproducibility constraints, and every word of generated reasoning traces to a real field in the candidate's profile so it cannot hallucinate.

| Metric | Target |
|---|---|
| Composite score (`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`) | Maximize |
| Honeypot rate in top 100 | < 10% (hard disqualifier), aim ≈ 0% |
| Ranking-step runtime | ≤ 5 min wall-clock |
| Memory / compute | ≤ 16 GB RAM, CPU only, no network |
| Reasoning hallucination rate | 0 (fact-grounded by construction) |

---

## 2. Background & context

### 2.1 The challenge in one minute
Redrob ships a JD for a *Senior AI Engineer — Founding Team* role and a pool of 100,000 candidate profiles. We must output a CSV ranking the 100 best-fit candidates, each with a 1–2 sentence justification. Scoring is against a hidden ground-truth relevance tiering, revealed only after submissions close (no live leaderboard, 3 submissions max).

### 2.2 Why this is hard
- **The JD rewards reading between the lines.** The JD text explicitly states that "find candidates whose skills section contains the most AI keywords" is a *trap*. A Tier-5 candidate may never write "RAG" or "Pinecone" but have built a recommender at a product company; a "Marketing Manager" with a perfect skill list is not a fit.
- **The data is adversarial.** Keyword-stuffers, plain-language strong candidates, behavioral twins, and ~80 **honeypots** with subtly impossible profiles (e.g. 8 years at a 3-year-old company; "expert" in a skill with 0 months used). Ranking honeypots into the top 100 at >10% is an automatic disqualification.
- **No labels.** There is no training signal shipped with the data — the ground truth is hidden. Any learned ranker must be trained on weak/derived labels.
- **A real compute budget.** The ranking step must finish in 5 minutes on CPU with no network. Calling a hosted LLM per candidate (100K calls) is impossible and explicitly disallowed. This forces a small ranker over precomputed features.
- **Behavioral availability matters.** A perfect-on-paper candidate who hasn't logged in for 6 months with a 5% recruiter response rate is, for hiring purposes, unavailable and must be down-weighted.

### 2.3 Industry context (what we borrow)
This is a talent-search / person-job-fit problem. We borrow three well-established lines of research (§5) rather than inventing a method: hybrid retrieval with rank fusion (search engines), person-job-fit aspect modeling (online recruitment research), and gradient-boosted learning-to-rank (the workhorse of commercial ranking).

### 2.4 Constraints we accept
- Single JD (the released role); we do not build a general multi-JD matcher, though the design is JD-agnostic by config.
- English-language profiles assumed dominant; no translation layer.
- CPU-only at rank time; any GPU use is confined to offline pre-compute.
- We optimize for the published metric weights, not for a hypothetical production objective.

---

## 3. Goals & non-goals

### 3.1 Functional goals
1. Parse 100K candidate records from `candidates.jsonl` against `candidate_schema.json`.
2. Decompose the JD into structured requirement aspects.
3. Recall a high-relevance shortlist from the full pool.
4. Score and rank candidates with a metric-aligned learned ranker.
5. Detect and exclude honeypots / impossible profiles.
6. Emit a spec-valid 100-row CSV with fact-grounded reasoning.
7. Pass `validate_submission.py` with zero violations.

### 3.2 Non-functional goals
- Ranking step ≤ 5 min, ≤ 16 GB, CPU-only, no network.
- Fully reproducible from a single documented command (Stage-3 requirement).
- Every reasoning claim traceable to a profile field (Stage-4 requirement).
- Defensible architecture a human can walk through (Stage-5 requirement).

### 3.3 Non-goals
- No live LLM inference at rank time.
- No live leaderboard tuning / submission-spamming (3-submission cap; validate offline instead).
- No general-purpose resume parser beyond this schema.
- No fairness/debiasing module (out of scope for this challenge; noted as future work).

---

## 4. Requirements

### 4.1 Functional requirements
| ID | Requirement |
|---|---|
| FR-001 | Produce a CSV with header `candidate_id,rank,score,reasoning` and exactly 100 data rows. |
| FR-002 | Each rank 1–100 appears exactly once; `score` is non-increasing with rank; ties broken by `candidate_id` ascending. |
| FR-003 | Every `candidate_id` exists in `candidates.jsonl` and matches `^CAND_[0-9]{7}$`. |
| FR-004 | Decompose the JD into must-have, nice-to-have, disqualifier, and behavioral-modifier aspects. |
| FR-005 | Compute per-candidate aspect-fit features from verbatim schema fields (§7.2). |
| FR-006 | Flag honeypots via deterministic consistency rules (§7.4) and exclude them from the top 100. |
| FR-007 | Generate 1–2 sentence reasoning per candidate, every claim grounded in a real field. |
| FR-008 | Validate output with `validate_submission.py` before declaring done. |

### 4.2 Non-functional requirements
| ID | Requirement |
|---|---|
| NFR-001 | Ranking step (jsonl → CSV) completes in ≤ 5 min wall-clock on a 16 GB CPU-only machine. |
| NFR-002 | Ranking step makes zero network calls (no OpenAI/Anthropic/Cohere/Gemini/etc.). |
| NFR-003 | Peak memory ≤ 16 GB; intermediate disk ≤ 5 GB. |
| NFR-004 | One command reproduces the CSV from the candidates file (`python rank.py --candidates ... --out ...`). |
| NFR-005 | Pre-compute artifacts (embeddings, BM25 index, trained model) are versioned and regenerable by a documented script. |
| NFR-006 | Top-100 honeypot rate < 10%. |

---

## 5. Research pillars (the design is grounded in these)

> The challenge interviews on *why* we chose our approach. We anchor on three established research lines and adapt each to the compute budget.

### 5.1 Pillar 1 — Hybrid sparse+dense retrieval with Reciprocal Rank Fusion (RRF)
**Recall stage.** BM25 (sparse, exact-term) catches candidates whose history literally describes building search/ranking/recommendation systems; dense embeddings (semantic) catch strong candidates who phrase their experience without the JD's buzzwords. We fuse the two ranked lists with **Reciprocal Rank Fusion**: `RRF(d) = Σ 1/(k + rank_i(d))`. RRF needs no training and no score calibration — it operates on ranks, so it sidesteps the incomparability of BM25 scores (0–20+) and cosine similarity (0–1). Cited: Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009.

### 5.2 Pillar 2 — Ability-aware Person-Job Fit (APJFNN / PJFNN family)
**Feature + evidence stage.** Person-job-fit research models the match as a set of *abilities/requirements* with per-aspect attention, rather than a single bag-of-words similarity. We adapt this to our budget as **structured aspect scoring**: the JD is decomposed into requirement aspects (must-haves like "production embeddings/retrieval experience," disqualifiers like "consulting-only career" or "pure research, no production," behavioral modifiers like "actually available"), and each candidate gets a per-aspect fit score with the evidence that produced it. This directly operationalizes the JD's instruction to reason about *meaning*, not keyword overlap, and the per-aspect evidence feeds the reasoning generator. Cited: Qin et al., *Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach*, SIGIR 2018 (arXiv:1812.08947) and the PJFCANN co-attention line.

### 5.3 Pillar 3 — LambdaMART / GBDT learning-to-rank
**Reranking stage.** A gradient-boosted decision-tree ranker (LambdaMART) combines all features into a final score and **directly optimizes NDCG** — exactly the metric we are scored on (NDCG@10 and @50 dominate the composite at 0.80 combined weight). It trains and infers in milliseconds on CPU, fitting the no-network rank-step budget. Cited: Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*; implemented via XGBoost/LightGBM `rank:ndcg`.

### 5.4 Supporting technique — rule-based honeypot gating
Not a research "pillar" but essential: deterministic consistency checks (§7.4) hard-filter impossible profiles before reranking, keeping the top-100 honeypot rate near zero.

---

## 6. High-level architecture

```
                          OFFLINE PRE-COMPUTE (network allowed, GPU optional)
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  JD ──► aspect extraction ──► aspects.json                                  │
  │  candidates.jsonl ──► profile text ──► BM25 index  +  dense embedding matrix│
  │  weak labels (JD rubric + aspect scores) ──► train LambdaMART ──► model.bin │
  └───────────────────────────────────────────────────────────────────────────┘
                                      │ artifacts
                                      ▼
                  RANKING STEP  (≤5 min · ≤16 GB · CPU only · NO network)
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  A. RRF recall      100K ──► ~1–2K shortlist                                │
  │  B. feature build   per-candidate aspect-fit + seniority + behavioral + flags│
  │  C. honeypot gate   drop impossible profiles                               │
  │  D. LambdaMART      score shortlist ──► sort ──► top 100                    │
  │  E. reasoning       deterministic fact-grounded sentences                  │
  │  F. emit + validate CSV ──► validate_submission.py                         │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Detailed component design

### 7.1 Stage A — Hybrid recall (Pillar 1)
- **Profile text** = concatenation of headline, summary, current_title, and each `career_history[].title/company/description` (capped length).
- **BM25** over profile text using the JD's expanded query terms (must-have aspect terms + synonyms).
- **Dense**: precomputed sentence-embedding matrix (100K × d) for profiles; query = JD aspect embedding(s); cosine top-N via a flat in-memory matrix (CPU, vectorized NumPy — no ANN index needed at 100K scale within budget, but FAISS-flat is a fallback).
- **Fusion**: RRF over the two ranked lists, `k=60`. Output ~1,000–2,000 shortlist (recall-oriented; precision is the reranker's job).

### 7.2 Stage B — Feature construction (Pillar 2)
Features per candidate, all from verbatim schema fields:

| Feature group | Source fields | Intuition |
|---|---|---|
| Aspect-fit (must-haves) | `skills[]`, `career_history[].description/title`, `profile.summary` | Per-aspect match score (e.g. "production retrieval/embeddings experience"). |
| Seniority fit | `profile.years_of_experience`, `career_history[].duration_months/title` | JD targets 5–9 yrs with judgment; reward the band, don't hard-cut. |
| Product-vs-services | `career_history[].company/industry/company_size` | JD penalizes consulting-only careers; rewards product-company shipping. |
| Research-vs-shipping | `career_history[].description`, `skills[]` | JD down-weights pure-research-only backgrounds. |
| Behavioral availability | `redrob_signals.recruiter_response_rate`, `last_active_date`, `open_to_work_flag`, `interview_completion_rate`, `notice_period_days` | Multiplier for "actually hireable." |
| Engagement / credibility | `redrob_signals.github_activity_score`, `saved_by_recruiters_30d`, `verified_email/phone`, `profile_completeness_score` | External validation signals. |
| Location fit | `profile.location/country`, `redrob_signals.willing_to_relocate`, `preferred_work_mode` | JD prefers Pune/Noida or relocation-willing. |
| Honeypot flags | see §7.4 | Hard-negative indicators. |

### 7.3 Stage C — Honeypot gate (supporting technique)
Implemented in `src/honeypot.py`. **Calibration finding (full-pool scan, 2026-06-15):** the
synthetic data is intentionally noisy, so not every "impossible-looking" check is a honeypot
signal. We split the checks into **HARD** (logical impossibilities that essentially never occur
in normal records → exclude from top-100 eligibility) and **SOFT** (common in the pool → recorded
as feature/quality signals only, never used to hard-filter). Hard rules flag **40 / 100,000
(0.04%)** — a conservative, high-precision set, since a false positive here wrongly drops a real
candidate.

**HARD rules (exclude):**

- `career_history[]` duration **over-claim**: `duration_months > date_span + tolerance` (the "8 yrs at a 3-yr-old company" pattern). One-sided — under-claims are benign gaps.
- `is_current == true` but `end_date != null`.
- `end_date < start_date`, or `start_date` after the snapshot.
- ≥ 3 skills simultaneously `proficiency == "expert"` with `duration_months == 0` (`MULTI_EXPERT_ZERO`).
- `education[].start_year > end_year`.

**SOFT signals (feature only, NOT a filter):**

- `expected_salary_range_inr_lpa.min > .max` — present in **18.9%** of the pool, so noise, not a honeypot.
- `skills[].duration_months > years_of_experience·12` — present in **13.4%** of the pool.
- a *single* `expert` skill with `duration_months == 0`.

Snapshot date and tolerances live in `src/config.py`. Rationale: precision over recall — the
reranker plus soft features push any un-gated honeypots down without risking good candidates.

### 7.4 Stage D — LambdaMART rerank + ensemble (Pillar 3, `train.py` / `rank.py`)
- Library: XGBoost `rank:ndcg` (`src/train.py`).
- **Labels**: no ground truth, so train on **weak labels** from the JD rubric (`labels.py`) — a graded tier (0–4) from aspect-fit + disqualifier penalties + behavioral availability, honeypots forced to 0.
- **Balanced training (important)**: a single 100k group is ~99.5% tier-0, which collapses the lambdarank gradient (first attempt produced a degenerate all-zero-importance model). We train on a **balanced subset** — every positive (tier≥1) plus sampled hard negatives (high-rubric tier-0: keyword-stuffers, near-misses) and random negatives (~6.4k docs). Result: 28 active features, top features = semantic must-have fit, retrieval, vector-DB, location, seniority, behavioral.
- **Ensemble late-fusion** (`rank.py`): the final score blends the LambdaMART score with the interpretable rubric and an offline cross-encoder teacher (§7.4b): `0.5·model + 0.2·rubric + 0.3·CE` (or `0.6·model + 0.4·rubric` if the CE artifact is absent). Late fusion needs no retraining and is robust to any single model's quirks.
- **Output**: sort desc → ties by `candidate_id` asc → top 100; CSV scores min-max normalized to a strictly non-increasing (0,1] range (FR-002).

### 7.4b Cross-encoder teacher (offline, `cross_encode.py`)
A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) jointly reads (JD, profile) and scores relevance far more precisely than bi-encoder cosine, but is too slow per-candidate for the network-free rank step. So we **precompute** it OFFLINE over the top-N recall candidates → `artifacts/ce_scores.json` (id → score), and late-fuse at rank time. This is the standard retrieve-then-rerank teacher pattern and adds a semi-independent signal that reduces the weak-label circularity. Candidates without a precomputed score fall back to the neutral median, so the pipeline is correct with or without the artifact.

### 7.7 Evaluation & verification (`evaluate.py`, `ablation.py`)
- **Metrics**: exact competition metrics implemented (`ndcg@k`, `map`, `p@k`, weighted composite).
- **Trap probe**: one ideal candidate + one of each documented trap (Marketing-Manager-with-AI-skills, keyword-stuffer, consulting-only, honeypot, junior) with constructed-correct labels; we assert the rubric AND the trained model rank the ideal above every trap and that the honeypot is detected. **Status: PASS.**
- **Ablation** (rubric as reference): BM25-only 0.51 / dense-only 0.56 / RRF 0.51 / RRF+LambdaMART ensemble 0.97 composite; 0 honeypots in every variant's top 100. *Caveat: the LTR figure is optimistic because the model is trained toward the same rubric used as reference; the BM25/dense/RRF comparison is the unbiased part and confirms dense + fusion help.*

### 7.5 Stage E — Reasoning generation (fact-grounded, no LLM)
- For each top-100 candidate, assemble 1–2 sentences from **evidence slots** populated only by real fields: years + current title (specific fact), the strongest matched aspect (JD connection), and the most material gap/concern (honest concern, e.g. "120-day notice" or "low recruiter response rate").
- Sentence templates are varied by rank band and by which evidence slots fired, so the 10 sampled reasonings at Stage-4 are substantively different and tone matches rank.
- **Hallucination is impossible by construction**: a claim is only emitted if its source field exists and crosses a threshold.

### 7.6 Stage F — Emit & validate
- Write CSV in spec column order; run `validate_submission.py`; fail loudly on any violation. Also compute and log the top-100 honeypot rate as a self-check (must be < 10%).

---

## 8. Data model summary

Source: `candidate_schema.json`. A candidate record:
- `candidate_id` — `CAND_XXXXXXX`.
- `profile` — anonymized_name, headline, summary, location, country, years_of_experience, current_title, current_company, current_company_size (enum), current_industry.
- `career_history[]` (1–10) — company, title, start_date, end_date|null, duration_months, is_current, industry, company_size, description.
- `education[]` (0–5) — institution, degree, field_of_study, start_year, end_year, grade|null, tier (`tier_1..tier_4`/`unknown`).
- `skills[]` — name, proficiency (`beginner..expert`), endorsements, duration_months.
- `certifications[]` (opt) — name, issuer, year.
- `languages[]` (opt) — language, proficiency.
- `redrob_signals{}` — 23 behavioral fields (see `redrob_signals_doc.docx`): completeness, signup/last_active dates, open_to_work_flag, profile_views/applications_30d, recruiter_response_rate, avg_response_time_hours, skill_assessment_scores, connection_count, endorsements_received, notice_period_days, expected_salary_range_inr_lpa{min,max}, preferred_work_mode, willing_to_relocate, github_activity_score, search_appearance_30d, saved_by_recruiters_30d, interview_completion_rate, offer_acceptance_rate, verified_email, verified_phone, linkedin_connected.

---

## 9. Scoring-metric alignment

| Component | Weight | How the design targets it |
|---|---|---|
| NDCG@10 | 0.50 | LambdaMART optimizes NDCG; honeypot gate protects the top-10 from impossible profiles. |
| NDCG@50 | 0.30 | RRF recall ensures genuinely strong candidates are in the shortlist to fill 11–50. |
| MAP | 0.15 | Aspect-fit + disqualifier features push truly relevant (tier 3+) candidates above adjacent ones. |
| P@10 | 0.05 | Behavioral-availability multiplier avoids "perfect-on-paper but unavailable" in the top 10. |

---

## 10. Risks (see [roadmap.md](roadmap.md) §Risks for mitigations)
- Weak labels diverge from hidden ground truth → mitigate by anchoring labels to the JD's explicit rubric and sanity-checking against hand-labeled samples.
- Embedding model too large for budget → choose a compact CPU-friendly model; embeddings are precomputed offline anyway.
- Reasoning judged templated → vary by evidence slots + rank band; review 10 random rows manually.
- Recall misses a strong candidate → keep shortlist generous (~2K) and union BM25+dense before fusion.
