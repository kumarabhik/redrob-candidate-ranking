# Intelligent Candidate Discovery & Ranking

> Rank the top 100 of 100,000 candidates for a job description in ~40s on CPU, with no network and zero hallucinated reasons. Built for the INDIA.RUNS (Redrob) challenge.

**`valid` submission · 0 honeypots in top 100 · ~40s rank step (budget 5 min) · explainable (SHAP) · multi-JD web UI**

Reads candidates on *meaning and evidence*, not keyword overlap: hybrid retrieval → person-job-fit features + trap detectors → honeypot gate → LambdaMART + cross-encoder ensemble → fact-grounded reasons.

### Quickstart
```bash
pip install -r requirements.txt
python src/jd_aspects.py && python src/build_index.py --embedder tfidf && python src/train.py
python src/rank.py --candidates candidates.jsonl --out submission.csv   # the submission
python app.py                                                           # the multi-JD web UI
```

**Docs:** [design_doc.md](design_doc.md) · [MATH.md](MATH.md) · [system_design.md](system_design.md) · [roadmap.md](roadmap.md) · [REPORT.md](REPORT.md) · [PITCH.md](PITCH.md) · [agents.md](agents.md) · [system.md](system.md) · [explanation.md](explanation.md)

**Two entry points:** `src/rank.py` produces the optimized single-JD competition submission (LambdaMART + cross-encoder ensemble). `app.py` + `src/serve.py` are the **multi-user, multi-JD** service + web UI: any recruiter pastes any JD and ranks the shared 100k pool (see [system_design.md](system_design.md)).

## Approach (3 research pillars + cross-encoder teacher)
1. Hybrid BM25 + dense retrieval fused with Reciprocal Rank Fusion (recall).
2. Person-Job-Fit structured aspect scoring (50+ features + evidence + trap detectors).
3. LambdaMART / GBDT learning-to-rank that directly optimizes NDCG (reranking),
   **late-fused** with the interpretable rubric and an **offline cross-encoder teacher**.
Plus a rule-based honeypot gate and a deterministic, hallucination-proof reasoning generator.

## Layout
```
src/   parse, honeypot, jd_aspects, embedder, build_index, recall, features,
       labels, train, reasoning, rank (ENTRYPOINT), cross_encode, evaluate,
       ablation, sandbox_demo, sandbox_app
artifacts/  aspects.json, embeddings.npy, jd_vectors.npy, candidate_ids.json,
            index_meta.json, model.bin, feature_names.json, ce_scores.json
tests/      test_honeypot.py (10), test_features.py (6)
```

## Current results (verified 2026-06-15)
- 100,000 records parsed; head-sample schema-valid.
- Honeypot gate: 40 hard honeypots (0.04%); **0 in the top 100**.
- Rank step: **~40–48s** wall-clock, CPU only, no network (budget is 5 min).
- `validate_submission.py`: **valid**. Trap probe: **pass** (ideal > every trap).
- Ablation (rubric reference): full ensemble composite **0.97** vs BM25 0.51 / dense 0.56 / RRF 0.51.

## Reproduce
```bash
# offline pre-compute (network allowed) — once / when artifacts change
python src/jd_aspects.py
python src/build_index.py
python src/train.py

# ranking step (≤5 min, CPU, no network) — produces the submission
python src/rank.py --candidates "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --out submission.csv

# validate format
python "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

## Status
Phase 0–1 in progress (parsing + honeypot detection). See [roadmap.md](roadmap.md) for the phase checklist.
