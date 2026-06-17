# Pitch — Intelligent Candidate Discovery & Ranking

Deck content for the Idea Submission Template. One section per slide. Backed by
[design_doc.md](design_doc.md), [MATH.md](MATH.md), [system_design.md](system_design.md),
[REPORT.md](REPORT.md).

## 1. Solution Overview
- **What we built:** a three-stage ranking engine that returns the top 100 of 100,000 candidates for a JD, each with a fact-grounded reason, in ~40s on CPU with no network.
- **Pipeline:** hybrid recall (BM25 + dense, fused by RRF) → structured person-job-fit features + trap detectors → honeypot hard-gate → LambdaMART reranker, late-fused with an offline cross-encoder teacher → behavioral-twin de-dup → explainable reasoning.
- **What differentiates us from traditional matching:** we rank on *meaning and evidence*, not keyword overlap. We explicitly model the JD's "do NOT want" signals (wrong-role, consulting-only, keyword-stuffer, research-only, job-hopper), weight *where/how* a skill was used over its mere presence, fold in behavioral availability, hard-filter impossible profiles, and ground every reason in real fields so it cannot hallucinate.

## 2. JD Understanding & Candidate Evaluation
- **Key requirements extracted:** production embeddings/retrieval; vector-DB / hybrid-search ops; ranking + evaluation (NDCG/MRR/MAP, A/B); strong Python; 5–9y (ideal 6–8) at product (not services) companies; Pune/Noida or relocation; genuinely available/reachable.
- **Most important signals:** semantic + lexical fit on the must-have aspects, **career-evidence** (built retrieval/ranking/recsys, in the description, not just the skills list), product-vs-services history, seniority band, and a behavioral-availability multiplier (recruiter response rate, recency, notice, interview completion).
- **Beyond keywords:** the JD calls keyword-counting a trap. Our features down-weight skill-list stuffing without career evidence, and disqualifier features actively penalize a "Marketing Manager with a perfect AI skill list."

## 3. Ranking Methodology
- **Retrieve:** BM25 (exact terms) ∪ dense embeddings (semantics) → **Reciprocal Rank Fusion** (training-free) → ~6k shortlist.
- **Score:** 76 features per candidate (weight-ordered aspect slots, seniority, product-vs-services, behavioral, verified-competence, location, trap detectors).
- **Rank:** **LambdaMART** (XGBoost `rank:ndcg`) trained on weak labels from the JD's own rubric (honeypots → tier 0; balanced training to avoid gradient collapse), then **late-fused** `0.5·model + 0.2·rubric + 0.3·cross-encoder`.
- **Combine signals:** multiplicative rubric (fit × seniority × availability × product × credibility × trap-penalties) for labels + interpretability; the tree learns non-linear interactions; the cross-encoder adds a semi-independent teacher signal.

## 4. Explainability & Data Validation
- **How decisions are explained:** every candidate gets a 1–2 sentence reason assembled from real fields (years, title, matched skills, the strongest aspect, the most material concern), plus exact **XGBoost SHAP** feature attributions (`explain.py`).
- **No hallucination by construction:** a claim is emitted only if its source field exists and crosses a threshold. No LLM at rank time.
- **Inconsistent / low-quality / suspicious profiles:** a deterministic honeypot gate removes logical impossibilities (over-claimed tenure, current-role-with-end-date, reversed dates, ≥3 expert-with-0-months). Noisy-but-legal signals (inverted salary 18.9%, skill-dur>career 13.4%) are kept as soft features, not filters. A fairness/data-validation audit (`audit.py`) checks the top-100 distribution against the pool.

## 5. End-to-End Workflow
```
JD text → aspects (auto-derived) → embed queries
        → recall (BM25 ∪ dense → RRF) → features → honeypot gate
        → LambdaMART + rubric + cross-encoder ensemble → twin de-dup
        → calibrated scores → fact-grounded reasons → top-100 CSV → validate
```

## 6. System Architecture
- **Single-JD (submission):** `rank.py` over precomputed artifacts (embeddings, BM25, model, cross-encoder), CPU, no network, ≤5 min.
- **Multi-user product:** shared **JD-independent** candidate index (built once) + stateless rank workers; each recruiter's JD is the cheap per-request part (~3–5s). Web UI (`app.py`), serving layer (`serve.py`), async index builder, artifact store, metadata DB. Full diagram in [system_design.md](system_design.md).

## 7. Results & Performance
- **Valid** submission; **0 honeypots** in the top 100; rank step **~40s** (budget 5 min, CPU, no network).
- **Ablation** (rubric reference): full ensemble composite **0.97** vs BM25 0.51 / dense 0.56 / RRF 0.51.
- **Trap probe PASS:** an ideal candidate outranks every documented trap; honeypot detected.
- **Top-10** are product-company retrieval/ranking/recsys engineers at 6–8y (Meta, LinkedIn, Amazon, Flipkart, …) with honest concerns surfaced.
- **Generalizes:** distinct sensible rankings for AI / Data-Engineer / Frontend JDs via the multi-JD layer.

## 8. Technologies Used
Python 3.11; `rank_bm25` (sparse); TF-IDF + TruncatedSVD / sentence-transformers (dense, offline); NumPy (vectorized cosine); XGBoost `rank:ndcg` (LambdaMART); `cross-encoder/ms-marco-MiniLM` (offline teacher); Gradio (UI); Docker + GitHub Actions (reproduction + CI). Chosen for CPU-only, no-network, sub-5-minute reproducibility with explainability.

## 9. Submission Assets
GitHub repo (`github.com/kumarabhik/...`), this deck, the validated `submission.csv`, the hosted Gradio sandbox (HF Spaces), and a short walkthrough video.
