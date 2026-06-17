# system.md — Environment ground truth

Authoritative record of the dev environment and a rolling checkpoint log. Don't guess versions — run the command and paste output. Stale facts are worse than missing facts.

## Update protocol
1. Update this file in the same session you change the environment.
2. Toolchain entries need a verified date; mark unverified ones `TO VERIFY`.
3. Add a new checkpoint block (newest first) at the end of each working session.
4. Never delete old checkpoints — they're the handoff trail.
5. If a doc rule turns out wrong, note it here and fix it in [agents.md](agents.md).

## Host facts
| Fact | Value |
|---|---|
| OS | Windows 11 Home Single Language (10.0.26200) |
| Shell | PowerShell 5.1 (primary); Git Bash available |
| Project root | `c:\Users\kumar\Downloads\XOXO\redrob` |
| Data dir | `…\redrob\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\` |
| Candidate file | `candidates.jsonl` (~465 MB, 100,000 records, uncompressed, present) |
| Rank-step target machine | 16 GB RAM, CPU only, no network (per spec) |

## Installed toolchain

| Tool | Version | Path | Verified | Notes |
|---|---|---|---|---|
| Python | 3.11.9 | `C:\Users\kumar\AppData\Local\Programs\Python\Python311\` | 2026-06-15 | matches spec target |
| pip | 24.0 | (bundled) | 2026-06-15 | upgrade to 26.x available; not required |
| numpy | 2.2.6 | site-packages | 2026-06-15 | embedding matrix |
| pandas | 3.0.3 | site-packages | 2026-06-15 | CSV emit |
| jsonschema | 4.26.0 | site-packages | 2026-06-15 | Draft-7 schema validation |
| rank_bm25 | 0.2.2 | site-packages | 2026-06-15 | sparse retrieval |
| sentence-transformers | NOT INSTALLED | — | — | install when Phase 2 starts (offline embedding) |
| xgboost | NOT INSTALLED | — | — | install when Phase 4 starts (`rank:ndcg`) |
| lightgbm | NOT INSTALLED | — | — | optional `lambdarank` fallback |
| faiss-cpu | NOT INSTALLED | — | — | optional; only if flat NumPy cosine is too slow |

## Not installed (needed later by roadmap)

- `sentence-transformers` (+ torch) — Phase 2 dense embeddings. Heavy download; deliberately deferred so Phase 0–1 stayed fast.
- `xgboost` — Phase 4 reranker.
- Install via `pip install -r requirements.txt` when those phases begin, then update the table above.

## Environment quirks
- **PowerShell vs Bash**: this repo's scripts are POSIX-flavored examples; on Windows run via Git Bash or translate (`/dev/null` → `$null`, forward slashes, etc.).
- **465 MB jsonl**: never `json.load` the whole file; stream line-by-line. Loading all 100K parsed dicts is feasible in 16 GB but watch peak memory when also holding the embedding matrix.
- **CPU only at rank time**: any GPU acceleration is confined to offline embedding pre-compute; the reproduced rank step must not require a GPU.
- **No network at rank time**: ensure no import triggers a model download (pre-download/cache the embedding model into `artifacts/` and load from disk).
- **Embedding matrix size**: 100K × d floats — at d=384, float32 ≈ 154 MB; fine. At d=768 ≈ 307 MB; still fine.

## Checkpoint log (newest first)

### 2026-06-18 — Round 6: submission-ready packaging, pitch, audit, hardening
- **Deploy packaging:** moved the Gradio UI to root `app.py` (HF Spaces convention; removed `src/app.py`); added `HF_SPACE_README.md` (Space frontmatter). Verified root `app.py` UI smoke (`_rank` returns 10 rows + CSV, routes to in-domain model).
- **Pitch:** `PITCH.md` fills the Idea Submission Template (solution, JD understanding, methodology, explainability/data-validation, workflow, architecture, results, tech, assets).
- **Fairness/data-validation:** `src/audit.py` compares top-100 vs pool. **Finding:** intended skew confirmed (India +21, open-to-work +41, github-linked +46, recruiter-response +0.21, exp in 6-8 band). **Watch item:** tier-1 education 6.4% pool → 62% top-100 (+55.6). Disclosed, not silently changed: likely correlational (strong product-ML candidates skew tier-1 here) and `edu_tier` is not a top model feature. Follow-up: sensitivity-test by zeroing `edu_tier`.
- **Hardening:** added `tests/test_recall.py` (RRF/tokenize, 4) and `tests/test_serve.py` (rank_jd shape + domain routing, 1). Full suite green: honeypot 10/10, features 6/6, recall 4/4, serve 1/1, trap probe PASS.
- **README hero + quickstart** added; installed gradio 6.18.0.
- **Re-verified:** submission still `valid`, 0 honeypots.
- **Next-session focus:** deploy `app.py` to HF Spaces + fill `submission_metadata.yaml` TODOs + push to GitHub + record a walkthrough video; optional MiniLM/BGE upgrade; optional edu_tier sensitivity test.

### 2026-06-17 — Round 5: resolve the multi-JD limitations
- **Fixed-slot feature schema:** refactored `features.py` so aspect features are weight-ordered SLOTS (`sem_must_1..4`, `hit/cev_must_k`, `sem/hit_nice_k`) instead of id-keyed columns. Updated `labels.py` + `reasoning.py` to read slots. Feature schema is now JD-agnostic, so the trained LambdaMART transfers across roles. Retrained (top features now `sem_must_4/3/1`, `sem_must_mean`, `ai_skill_depth`). Submission re-validated (valid, 0 honeypots, 34s); probe PASS.
- **Domain routing in `serve.py`:** the slot schema lets the model run on any JD, but its LEARNED weights are AI/ML-specific, so on a Frontend JD it dragged ML profiles up. Resolved by routing: model+rubric for in-domain JDs (dominant aspect ∈ {retrieval, vectordb, ranking_eval, python_ml, llm}), neutral rubric otherwise. Verified: AI → Meta Sr AI Eng #1 (model), Frontend → real frontend engineers (rubric), Data → Senior Data Engineers (rubric). `meta.ranker` reports which path ran.
- **UI now actually runs:** installed gradio 6.18.0; functionally tested `app._rank()` end-to-end (returns table + CSV + status), not just `py_compile`. Note: the Frontend #1 is a DevOps eng whose fact-grounded reason is "matches Frontend (Next.js, TypeScript)" — a legitimately frontend-skilled candidate, not a bug.
- **Net:** the three previously-noted limitations are resolved or shown defensible.

### 2026-06-17 — Round 4: math rigor + system design + multi-user/multi-JD + UI
- **User-request context**: "make it more mathematically correct; is there no system design? no UI? make it user-friendly; make it for more than 1 user."
- **Work completed (all run + verified)**:
  - `MATH.md`: formal spec for BM25, dense cosine, RRF, feature map, honeypot predicate, weak-label rubric, LambdaMART lambda gradients, ensemble, calibration, the exact composite metric, and per-JD complexity (the proof that candidate embeddings + BM25 are JD-independent → shareable).
  - `system_design.md`: multi-tenant production architecture — shared read-only candidate index + stateless rank workers, JD-aspect service, async index builder, API sketch, sequence flows, scaling/capacity, failure modes.
  - Persisted the fitted embedder → `artifacts/embedder.pkl` (re-ran `build_index --embedder tfidf`; embeddings deterministic so `model.bin` stays valid; **submission re-validated, 0 honeypots, 33s**).
  - `jd_aspects.aspects_from_jd_text()`: auto-derives aspects from arbitrary JD text via a skill taxonomy; **neutral disqualifiers** (fixed a bug where the AI-role's non-eng-title list penalized Frontend candidates).
  - `serve.py` `rank_jd()`: caches the shared index once (~29s), then ranks any JD by the general rubric in **~3–5s**. Verified distinct sensible top-5 for AI Engineer / Data Engineer / Frontend JDs.
  - `app.py`: user-friendly multi-JD Gradio UI (templates + paste-your-own, slider, ranked table + reasons + CSV download); supports concurrent users.
- **Notes / honest limits**: the multi-JD demo ranks by the general rubric (the trained LambdaMART + cross-encoder are tied to the default role's aspect schema, so they power `rank.py`, not arbitrary JDs). Frontend demo: ranks 2–5 are real frontend engineers; #1 was a DevOps eng with strong frontend skills — acceptable for the rubric-only path. `gradio` not installed locally (in requirements; HF Spaces installs it) — `app.py` is `py_compile`-checked, `serve.py` fully run.
- **Next-session focus**: optional MiniLM/BGE embeddings + retrain; deploy `app.py` to HF Spaces; fill metadata TODOs; push to GitHub.

### 2026-06-15 — Round 3 power-ups (feature enrichment, dedup, explainability, Docker/CI)
- **User-request context**: "make it bigger/stronger/badass, do the next 10 steps."
- **Decision**: stopped the MiniLM embedding rebuild for the 2nd time (still ~40 min to go at ~10s/batch and would clobber the working `embeddings.npy`); committed to TF-IDF + cross-encoder and invested in higher-leverage additions instead. MiniLM remains one flag away (`build_index.py --embedder st`).
- **Work completed (all run + verified)**:
  - Enriched `features.py` 66 → **76 features**: AI-skill depth, endorsement-trust (endorsements × real usage), platform skill-assessment alignment, education tier, certifications, response speed, `current_role_is_ai` counter-signal, recency-weighted `systems_months`. Folded a verified-competence credibility factor into `labels.py`.
  - Retrained LambdaMART: 29 active features, rubric overlap **65/100** (up from 56); `ai_skill_depth` is the #5 feature.
  - Widened cross-encoder teacher to **top 2000** (`ce_scores.json`).
  - `rank.py`: **behavioral-twin de-dup** (embedding cosine > 0.985 suppressed, with backfill) + **score calibration** (concave top-heavy, strictly non-increasing).
  - `explain.py`: per-candidate **XGBoost SHAP** feature attributions → `artifacts/explanations.md`.
  - `REPORT.md` (results/ablation/top-10), `Dockerfile` + `.dockerignore` (Stage-3 reproduction), `.gitignore`, `.github/workflows/ci.yml` (runs tests + trap probe on push).
  - End-to-end re-run: rank **~60s**, **valid**, **0 honeypots**, trap probe **PASS**. Top-10 = Meta/Sarvam/CRED/Zoho/Meesho/LinkedIn/Flipkart/Freshworks/Ola/Amazon (all product-co retrieval/recsys, 6.5–8y).
- **Current truth constraints / next-session focus**: optional MiniLM/BGE embeddings (retrain after); deploy sandbox to HF Spaces + fill metadata TODOs; push to GitHub (CI will run). Ablation LTR figure still optimistic (disclose).

### 2026-06-15 — Phases 2–4 + GOAT additions (full working pipeline)
- **User-request context**: "do the next 10 steps ... make it bigger and stronger, goat level." Build the full ranking pipeline + advanced components for a top-10 finish.
- **Verified tool reality**: installed scikit-learn 1.9.0, xgboost 3.2.0, sentence-transformers 5.5.1 (+ torch 2.12.0), and (for the sandbox) gradio. **Key finding: transformer *bi-encoder* over 100k profiles is very slow on this CPU (killed a ~20-min run with no output); the *cross-encoder* over 600 pairs is fine (~28s).** So the default dense embedder is TF-IDF+SVD (fast, offline); MiniLM/BGE is an optional offline upgrade.
- **Work completed (all run + verified)**:
  - Stage A recall: `jd_aspects.py` (JD→aspects.json), `embedder.py` (ST + TF-IDF/LSA fallback, `--embedder` switch), `build_index.py` (embeddings.npy 100k×256 in ~4.5min via TF-IDF), `recall.py` (BM25 ∪ dense → RRF, shortlist 5757; top picks are product-co Sr ML/NLP engineers — sane).
  - Stage B features: `features.py` (66 features incl. trap detectors) — `test_features.py` 6/6.
  - Stage C honeypot gate (from Phase 1) — 40 hard, 0 in top 100.
  - Stage D rerank: `labels.py` rubric → tiers (fixed degenerate single-group training by **balanced subset** sampling: positives + hard/random negatives, 6435-doc group → 28 non-zero features, 56/100 rubric overlap); `train.py` XGBoost rank:ndcg → model.bin.
  - GOAT: `cross_encode.py` (offline ms-marco cross-encoder teacher → ce_scores.json, 600 scores); `rank.py` **ensemble late-fusion** 0.5·model+0.2·rubric+0.3·CE (or 0.6/0.4 without CE); `evaluate.py` (NDCG/MAP/P@k + trap probe — PASS); `ablation.py` (ensemble 0.97 vs BM25 0.51/dense 0.56/RRF 0.51, rubric reference); `sandbox_demo.py` + `sandbox_app.py` (Gradio) + `SANDBOX.md`; `submission_metadata.yaml`; `run_all.ps1`.
  - **Rank step: ~40–48s, CPU, no network. `validate_submission.py`: valid. 0 honeypots in top 100.**
- **Validation completed**: test_honeypot 10/10, test_features 6/6, trap probe PASS, validator valid, ablation table produced.
- **Current live state**: end-to-end working; `submission.csv` produced and valid. All artifacts present except the optional MiniLM `embeddings.npy` upgrade (rebuild running in background `b4vhry91k`).
- **Current truth constraints / next-session focus**: (1) if MiniLM build finishes, re-run train + rank to swap in stronger embeddings and re-validate. (2) Deploy `sandbox_app.py` to HF Spaces and fill `sandbox_link` + team fields in `submission_metadata.yaml`. (3) Push to GitHub. (4) Ablation's LTR number is optimistic (model trained toward the rubric reference) — note in interview; the BM25/dense/RRF comparison is the unbiased part.

### 2026-06-15 — Phase 0–1 implementation (the "first 10 steps")
- **User-request context**: User flattened the docs to the `redrob` root and said "start" → execute the first 10 roadmap steps (Phase 0 foundation + Phase 1 parsing/honeypot).
- **Verified tool reality**: Python 3.11.9, pip 24.0. Installed numpy 2.2.6, pandas 3.0.3, jsonschema 4.26.0, rank_bm25 0.2.2 (versions in table above). `candidates.jsonl` confirmed at 464.69 MB. PowerShell quirk: the data folder name has literal `[` `]` — must use `Get-ChildItem -LiteralPath`, plain paths glob to empty.
- **Work completed**:
  - Scaffolded `src/`, `artifacts/`, `tests/`, `requirements.txt`, `README.md`.
  - `src/config.py` (paths + tunables), `src/parse.py` (streaming loader, `Candidate` typed wrapper, `profile_text()`, head-sample schema validation), `src/honeypot.py` (consistency detector), `tests/test_honeypot.py` (10 cases).
  - `parse.py`: streams 100,000 records; head-sample (500) schema-validates with 0 errors.
  - **Honeypot calibration finding**: naive "any impossibility" rules flagged **30%** of the pool because the synthetic data is intentionally noisy (inverted salary 18.9%, skill-duration>career 13.4%). Refactored to a **hard/soft split** — hard rules (over-claimed tenure, current-role-with-end-date, reversed dates, reversed edu years, ≥3 expert-with-0-months) flag **40 / 100,000 (0.04%)**; the noisy checks are demoted to soft feature signals. Updated `design_doc.md` §7.3 to match.
- **Validation completed**: `tests/test_honeypot.py` 10/10 pass; full-pool honeypot scan reproducible via `python src/honeypot.py`.
- **Current live state**: data layer + honeypot gate working and tested. No retrieval/features/reranker yet. `artifacts/` empty.
- **Current truth constraints / next-session focus**: Phase 2 (Pillar 1 recall) — install `sentence-transformers`, write `src/jd_aspects.py` (JD → aspects), `src/build_index.py` (BM25 index + dense embedding matrix → `artifacts/`), and `src/recall.py` (BM25 ∪ dense → RRF → ~1–2K shortlist). Decide embedding model (compact, CPU/offline) and cache it into `artifacts/` so rank-time stays no-network.

### 2026-06-15 — Docs-first session
- **User-request context**: User is competing in the INDIA.RUNS (Redrob × Hack2Skill) hackathon and asked for a planning-doc suite modeled on the methodology of the `Credit_Card` project under `XOXO/`, with the design grounded in the 3 best research approaches. Scope this turn: docs only; code deferred.
- **Verified tool reality**: none — no runtime commands run this session. Toolchain table is all `TO VERIFY`.
- **Work completed**: created `docs/` with five cross-referenced files — `design_doc.md`, `roadmap.md`, `agents.md`, `system.md`, `explanation.md`. Architecture decided: hybrid RRF recall (Pillar 1) → person-job-fit aspect features (Pillar 2) → LambdaMART rerank (Pillar 3), with rule-based honeypot gating and a deterministic fact-grounded reasoning generator. Confirmed the real submission constraints (≤5 min / 16 GB / CPU / no-network) and the real candidate schema field names from `candidate_schema.json` / `sample_candidates.json`.
- **Validation completed**: none (no code yet). Docs cross-link checked by hand.
- **Current live state**: planning docs exist; no `src/`, no artifacts, no code.
- **Next-session focus**: Phase 0 in [roadmap.md](roadmap.md) — scaffold `src/`/`artifacts/`/`tests/`, write `requirements.txt`, run installs and fill in the verified toolchain table above, then `parse.py` to stream-load and count 100,000 records. Then Phase 1 honeypot detector.
