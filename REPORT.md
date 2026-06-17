# Results Report — Intelligent Candidate Discovery & Ranking

Reproducible summary of the system's behavior on the released 100,000-candidate pool for the
*Senior AI Engineer* JD. See [design_doc.md](design_doc.md) for the architecture and
[explanation.md](explanation.md) for the plain-language walkthrough.

## Headline
- **100,000** candidates parsed; head-sample schema-valid (0 errors).
- Rank step: **~45–60s** wall-clock, CPU only, **no network** (budget: 5 min / 16 GB).
- `validate_submission.py`: **valid**. Honeypots in top 100: **0 / 100**.
- Trap probe: **PASS** (ideal candidate ranks above every documented trap; honeypot detected).

## Pipeline
Hybrid recall (BM25 + dense, RRF) → 76 person-job-fit features + trap detectors → honeypot
hard-gate → LambdaMART (XGBoost `rank:ndcg`, weak JD-rubric labels, balanced training) →
**ensemble late-fusion** with the rubric and an offline **cross-encoder teacher** →
behavioral-twin de-dup → calibrated scores → fact-grounded reasoning.

## Ablation (reference = JD rubric tier; relative comparison)
| Variant | NDCG@10 | NDCG@50 | MAP | P@10 | Composite | Honeypots@100 |
|---|---|---|---|---|---|---|
| BM25 only | 0.491 | 0.619 | 0.381 | 0.400 | 0.508 | 0 |
| Dense only | 0.520 | 0.597 | 0.354 | 0.500 | 0.517 | 0 |
| RRF fusion | 0.476 | 0.612 | 0.382 | 0.400 | 0.499 | 0 |
| **RRF + LambdaMART ensemble** | **1.000** | **0.965** | **0.930** | **1.000** | **0.979** | 0 |

> Honesty caveat: the LambdaMART row is optimistic — the model is trained on weak labels from
> the same rubric used here as the reference, so it largely reproduces the rubric ordering. The
> unbiased signal is the BM25 / dense / RRF comparison, which confirms dense retrieval and
> fusion each surface rubric-strong candidates that the other misses. The cross-encoder teacher
> (a different model family) supplies the semi-independent signal at the top of the list.

## Top model features (XGBoost importance)
`sem_must_mean` (0.21), `sem_vectordb` (0.15), `sem_python_ml` (0.13), `sem_retrieval` (0.12),
`ai_skill_depth` (0.10), `sem_ranking_eval`, `seniority_band`, `pref_city`, `location_fit`,
`systems_months`. Per-candidate SHAP attributions are in `artifacts/explanations.md`
(`python src/explain.py`).

## Top 10 (current submission)
| Rank | Title @ Company | Exp | Why |
|---|---|---|---|
| 1 | Senior AI Engineer @ Meta | 7.9y | ranking & evaluation (NDCG/MAP, A/B) |
| 2 | Senior Data Scientist @ Sarvam AI | 7.4y | embeddings/retrieval (IR, Semantic Search, RAG); notice 90d |
| 3 | Recommendation Systems Engineer @ CRED | 8.0y | vector search (Weaviate, Qdrant, Milvus) |
| 4 | Recommendation Systems Engineer @ Zoho | 6.6y | vector search (Qdrant, Milvus); notice 120d |
| 5 | Applied ML Engineer @ Meesho | 7.1y | vector search (Elasticsearch, FAISS, pgvector); notice 90d |
| 6 | Staff ML Engineer @ LinkedIn | 8.0y | production Python/ML |
| 7 | Senior Data Scientist @ Flipkart | 6.5y | vector search (Weaviate, OpenSearch, Elasticsearch) |
| 8 | Applied ML Engineer @ Freshworks | 8.0y | vector search (Pinecone, Weaviate, Qdrant) |
| 9 | Machine Learning Engineer @ Ola | 7.1y | production Python/ML (scikit-learn, MLOps) |
| 10 | Recommendation Systems Engineer @ Amazon | 6.5y | ranking & eval (Learning to Rank, RecSys) |

All ten are product-company retrieval/ranking/recommendation engineers in the JD's 6–8y ideal
band, with honest concerns (e.g. notice period) surfaced in the reasoning.

## Honeypot defense
40 hard honeypots detected over the pool (0.04%) via deterministic impossibility rules
(over-claimed tenure, current-role-with-end-date, reversed dates/edu-years, ≥3 expert-with-0-
months). Hard-gated before reranking; **0 reach the top 100**. Noisy synthetic signals
(inverted salary 18.9%, skill-duration>career 13.4%) are kept as soft features, not filters.

## Reproduce
```
python src/jd_aspects.py
python src/build_index.py --embedder tfidf     # or --embedder st for MiniLM/BGE (slow, offline)
python src/cross_encode.py --top 2000          # optional cross-encoder teacher
python src/train.py
python src/rank.py --candidates candidates.jsonl --out submission.csv
python "[PUB] .../validate_submission.py" submission.csv
```
or `pwsh run_all.ps1`. Tests: `python tests/test_honeypot.py`, `python tests/test_features.py`,
`python src/evaluate.py` (trap probe).
