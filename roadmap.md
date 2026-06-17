# Roadmap — Intelligent Candidate Discovery & Ranking

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done
Related: [design_doc.md](design_doc.md) · [agents.md](agents.md) · [system.md](system.md) · [explanation.md](explanation.md)

## Table of contents
1. Why this project exists
2. North-Star metrics
3. System architecture
4. Tech-stack decisions
5. Repo layout (target)
6. Roadmap by phase (0–6)
7. Stretch goals
8. Out of scope
9. Risks & mitigations
10. Verification matrix

---

## 1. Why this project exists
Win the INDIA.RUNS hackathon's *Intelligent Candidate Discovery & Ranking* challenge: rank the top 100 of 100,000 candidates against the Redrob *Senior AI Engineer* JD, output a spec-valid CSV with fact-grounded reasoning, and survive five evaluation stages (format → scoring → code reproduction → reasoning review → defend-your-work interview). The hard part is reading the JD's *intent* (not keywords), avoiding ~80 honeypots, and doing it all in ≤5 min on CPU with no network. Full problem framing is in [design_doc.md](design_doc.md) §2.

## 2. North-Star metrics
| # | Goal | Measurement |
|---|---|---|
| 1 | Maximize composite score | `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` on hidden ground truth |
| 2 | Honeypot rate in top 100 < 10% | self-computed honeypot count / 100 (aim ≈ 0) |
| 3 | Ranking step ≤ 5 min | wall-clock of `python rank.py` on 16 GB CPU |
| 4 | Reproducible | single documented command regenerates the CSV |
| 5 | Reasoning passes Stage-4 | 10 random rows: specific facts, JD connection, honest concerns, no hallucination, variation |

## 3. System architecture
```
OFFLINE pre-compute:  JD→aspects.json | profiles→BM25 index + embedding matrix | weak labels→LambdaMART model.bin
RANK STEP (≤5min,CPU,no-net):  A RRF recall → B features → C honeypot gate → D LambdaMART → E reasoning → F CSV+validate
```
Detailed component design: [design_doc.md](design_doc.md) §6–§7.

## 4. Tech-stack decisions
| Layer | Choice | Why | Alternatives (why not) |
|---|---|---|---|
| Language | Python 3.11 | ecosystem for IR/ML; matches spec examples | — |
| Sparse retrieval | `rank_bm25` (or BM25 in scikit) | simple, CPU-cheap, no service | Elasticsearch (heavy, network) |
| Dense embeddings | compact CPU sentence-transformer (e.g. `bge-small`/`e5-small`/`all-MiniLM-L6-v2`) | small, strong, precomputed offline | large models (don't fit budget); OpenAI embeddings (network-banned at rank time) |
| Vector search | in-memory flat NumPy cosine (FAISS-flat fallback) | 100K×d fits RAM; exact; no infra | ANN/HNSW (unneeded at 100K within budget) |
| Fusion | Reciprocal Rank Fusion (k=60) | training-free, score-agnostic | linear score blend (needs calibration) |
| Reranker | XGBoost `rank:ndcg` (LightGBM `lambdarank` fallback) | directly optimizes NDCG; ms inference on CPU | neural reranker (slower, needs GPU/labels) |
| Reasoning | deterministic templated-from-fields | zero hallucination; no network | LLM at rank time (banned, slow) |
| Validation | provided `validate_submission.py` | authoritative format check | hand-rolled checks |

Rationale ties to the three research pillars in [design_doc.md](design_doc.md) §5.

## 5. Repo layout (target)
```
redrob/
├── docs/                  # these five docs (this turn)
├── src/
│   ├── parse.py           # jsonl → typed candidate records
│   ├── honeypot.py        # consistency-rule detector
│   ├── recall.py          # BM25 + dense + RRF
│   ├── features.py        # aspect-fit + seniority + behavioral features
│   ├── jd_aspects.py      # JD → aspects.json (offline)
│   ├── train.py           # weak labels → LambdaMART (offline)
│   ├── reasoning.py       # fact-grounded sentence builder
│   └── rank.py            # ENTRYPOINT: jsonl → CSV (rank step)
├── artifacts/             # bm25.idx, embeddings.npy, model.bin, aspects.json
├── tests/
├── requirements.txt
├── submission_metadata.yaml
└── README.md              # single reproduce command
```

## 6. Roadmap by phase
> Pick the next `[ ]`. Flip to `[~]` when work starts (one at a time). Flip to `[x]` only when code + a test + doc update all land. See [agents.md](agents.md) §workflow.

### Phase 0 — Foundation
- [x] Create `src/`, `artifacts/`, `tests/`, `requirements.txt`, `README.md`
- [x] Confirm toolchain installs (see [system.md](system.md)) — Python 3.11.9; numpy 2.2.6, pandas 3.0.3, jsonschema 4.26.0, rank_bm25 0.2.2 (sentence-transformers/xgboost deferred to Phase 2/4)
- [x] `parse.py`: stream-load `candidates.jsonl` (465 MB) without exhausting RAM; asserts 100,000 records; head-sample validates against `candidate_schema.json` (0 errors)

### Phase 1 — Schema parsing + honeypot detector
- [x] Typed candidate record (`Candidate`) + profile-text builder (`parse.py`)
- [x] `honeypot.py`: §7.3 rules with **hard/soft split** (naive rules flagged 30% of the pool); 10/10 unit tests pass
- [x] Report honeypot count over full pool — **40 hard honeypots (0.04%)**, high-precision; soft signals (salary-inv 18.9%, skill>career 13.4%) kept for features

### Phase 2 — Hybrid recall (Pillar 1)
- [x] Offline: build dense embedding matrix (TF-IDF+LSA default; MiniLM/BGE optional) + JD vectors → `artifacts/` (`build_index.py`, `embedder.py`); BM25 rebuilt at rank time
- [x] `recall.py`: BM25 top-N + dense top-N → RRF fuse → ~5.7K shortlist
- [x] Sanity: top picks are product-co Sr ML/NLP engineers (Microsoft/Paytm/Meta/Apple/...)

### Phase 3 — Aspect features (Pillar 2)
- [x] `jd_aspects.py`: JD → must-have / nice-to-have / disqualifier / behavioral aspects (offline, hand-authored)
- [x] `features.py`: 66 features incl. trap detectors (wrong-role, consulting-only, keyword-stuffer, job-hopper, research-only, managerial)
- [x] Unit-test feature extraction on sample candidates (`test_features.py` 6/6)

### Phase 4 — LambdaMART rerank (Pillar 3)
- [x] Weak-label generator from JD rubric (`labels.py`; honeypots → tier 0)
- [x] `train.py`: XGBoost `rank:ndcg` on **balanced subset** (fixed degenerate single-group training) → `model.bin`; 28 active features
- [x] Rerank shortlist; emit top 100 with non-increasing scores

### Phase 5 — Reasoning generator
- [x] `reasoning.py`: evidence slots → 1–2 varied sentences (pure-ASCII), every claim field-backed
- [x] Spot-checked rows: specific facts + JD connection + honest concern, tone matches rank band

### Phase 6 — End-to-end + submission assets
- [x] `rank.py` wires A–F; single command produces CSV; **~40–48s** (< 5 min)
- [x] `validate_submission.py` → **valid**; top-100 honeypot rate **0%**
- [x] `submission_metadata.yaml` filled (team/sandbox fields TODO); `run_all.ps1`; `SANDBOX.md`; sandbox app
- [ ] Deploy sandbox to HF Spaces; push to GitHub; final dry-run on a clean 16 GB CPU box

### Phase 7 — Cross-encoder teacher (GOAT)
- [x] `cross_encode.py`: offline ms-marco cross-encoder over top-N recall → `ce_scores.json`
- [x] `rank.py` **ensemble late-fusion**: 0.5·LambdaMART + 0.2·rubric + 0.3·cross-encoder (0.6/0.4 if no CE)

### Phase 8 — Evaluation & ablation (GOAT)
- [x] `evaluate.py`: NDCG@10/@50, MAP, P@10 + **trap probe** (ideal > every documented trap) — PASS
- [x] `ablation.py`: BM25 / dense / RRF / +LambdaMART comparison (ensemble composite 0.97 vs 0.51–0.56)

### Phase 9 — Power-ups (depth, robustness, explainability, ops)
- [x] Enriched features 66 → **76** (AI-skill depth, endorsement-trust, assessment alignment, edu tier, certs, response speed, `current_role_is_ai`, recency-weighted systems tenure) + credibility factor in rubric; retrained (rubric overlap 65/100)
- [x] Cross-encoder teacher widened to **top 2000**
- [x] **Behavioral-twin de-dup** (embedding cosine > 0.985) + **score calibration** in `rank.py`
- [x] `explain.py`: per-candidate **XGBoost SHAP** attributions → `artifacts/explanations.md`
- [x] `REPORT.md`, `Dockerfile` + `.dockerignore` (Stage-3 reproduction), `.gitignore`, `.github/workflows/ci.yml`
- [ ] Optional MiniLM/BGE embeddings (`--embedder st`, slow on this CPU) + retrain; deploy sandbox; push to GitHub

### Phase 10 — Productization (math rigor, system design, multi-user UI)
- [x] `MATH.md`: formal spec of every component (BM25, RRF, aspect-fit, rubric, LambdaMART/NDCG lambdas, ensemble, calibration, composite metric, complexity)
- [x] `system_design.md`: multi-tenant architecture (shared JD-independent index + stateless rank workers, API, stores, async index builder, scaling, failure modes)
- [x] Persisted fitted embedder (`embedder.pkl`) so arbitrary JDs can be embedded at serve time
- [x] `jd_aspects.aspects_from_jd_text()`: auto-derive aspects from ANY JD via a skill taxonomy (neutral disqualifiers so it isn't biased to the AI role)
- [x] `serve.py`: `rank_jd(jd_text)` reusing a cached shared candidate index (multi-JD, ~3–5s warm)
- [x] `app.py`: user-friendly multi-JD Gradio UI (templates + paste-your-own → ranked table + reasons + CSV); concurrent users
- [x] Verified distinct sensible rankings for AI / Data-Engineer / Frontend JDs; main submission still valid

### Phase 11 — Submission-ready (packaging, pitch, audit, hardening)
- [x] Root `app.py` entrypoint (HF Spaces) + `HF_SPACE_README.md` frontmatter
- [x] `PITCH.md` — Idea Submission Template content (all sections)
- [x] `src/audit.py` — fairness/data-validation report (top-100 vs pool); finding: tier-1 edu skew disclosed as a watch item
- [x] `tests/test_recall.py` (4) + `tests/test_serve.py` (1); full suite green (honeypot 10, features 6, recall 4, serve 1, probe PASS)
- [x] README hero + quickstart
- [ ] Deploy `app.py` to HF Spaces; fill `submission_metadata.yaml` TODOs; push to GitHub; record walkthrough video

## 7. Stretch goals
- [ ] Swap in MiniLM/BGE dense embeddings (offline; CPU-slow at 100k — in progress) + retrain
- [ ] Calibrated/Platt score output for nicer monotonic spread
- [ ] Behavioral-twin near-duplicate suppression in the top 100

## 8. Out of scope
- Live LLM inference at rank time · multi-JD generalization · fairness/debiasing module · submission-spamming the leaderboard.

## 9. Risks & mitigations
| Risk | Likelihood | Mitigation |
|---|---|---|
| Weak labels diverge from hidden truth | Med | Anchor labels to JD's explicit rubric; hand-label ~30 samples to sanity-check |
| Embedding model too big for budget | Low | Compact model; embeddings precomputed offline |
| Reasoning flagged as templated | Med | Vary by evidence slots + rank band; manual 10-row review |
| Recall misses a strong candidate | Med | Generous ~2K shortlist; union before RRF |
| Rank step > 5 min | Low | Precompute everything; vectorized NumPy; profile early |
| Honeypot slips into top 100 | Low | Hard gate before rerank; self-check honeypot rate at emit |

## 10. Verification matrix
| Phase | How we know it's real |
|---|---|
| 0 | `parse.py` prints 100000; schema validation passes |
| 1 | honeypot unit tests green; full-pool count is sane |
| 2 | shortlist recall sanity on sample candidates |
| 3 | feature unit tests; aspects.json reviewed by hand |
| 4 | offline NDCG on hand-labeled sample beats BM25-only baseline |
| 5 | 10-row manual reasoning review passes all 6 Stage-4 checks |
| 6 | `validate_submission.py` zero violations; runtime < 5 min; honeypot rate < 10% |
