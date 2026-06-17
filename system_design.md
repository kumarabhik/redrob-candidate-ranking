# System Design — Multi-User Candidate Ranking Service

How the single-JD hackathon pipeline becomes a **multi-tenant product** that serves many
recruiters, each ranking their own JD against a shared candidate pool. Companion to
[design_doc.md](design_doc.md) (algorithm), [MATH.md](MATH.md) (formalism), and the runnable
multi-JD layer in `src/serve.py` + UI in `src/app.py`.

> Distinction from [system.md](system.md): that file is the dev-environment ground truth +
> checkpoints. *This* file is the production architecture.

## 1. Why this shape
Key invariant (proved in [MATH.md](MATH.md) §9): **candidate embeddings and the BM25 index are
JD-independent.** Only the JD aspect vectors, per-candidate features, and the rerank are
per-JD. So the expensive 100k work is built **once and shared**; each user's request is cheap.
That single fact drives the whole design: one shared "candidate index" service, many concurrent
lightweight "rank a JD" requests.

## 2. Requirements
**Functional**
- FR-1 A user submits JD text (paste or template) and gets the top-K ranked candidates with scores, reasons, and per-candidate explanations.
- FR-2 Many users/JDs concurrently; each JD is isolated (no cross-talk in results).
- FR-3 Arbitrary JDs (auto-derive aspects from text), not a hardcoded role.
- FR-4 Download results as the spec CSV; save/recall past JD "searches".
- FR-5 Honeypots never surface; reasons never hallucinate.

**Non-functional**
- NFR-1 Warm per-JD latency p95 < 3 s for top-100 over 100k (cached index).
- NFR-2 Cold index build (embeddings + BM25) is offline/async, not on the request path.
- NFR-3 Horizontal scale: stateless rank workers behind a shared read-only index.
- NFR-4 No hosted-LLM call on the synchronous rank path (cost/latency/privacy); cross-encoder + JD-aspect extraction are offline/cached.
- NFR-5 Tenant isolation + auth; candidate PII handled per policy.

## 3. Architecture
```
                         ┌────────────────────────── Web UI (Gradio / React) ──────────────────────────┐
                         │  paste JD → Rank → results table + reasons + SHAP panel + CSV download        │
                         └───────────────┬──────────────────────────────────────────────────────────────┘
                                         │ HTTPS / JSON
                                ┌────────▼─────────┐     auth, rate-limit, tenant routing
                                │   API Gateway     │
                                └────────┬─────────┘
                 ┌───────────────────────┼───────────────────────────┐
        POST /rank (sync, fast)          │                  POST /index/refresh (async)
                 │                        │                           │
        ┌────────▼─────────┐     ┌────────▼─────────┐        ┌────────▼─────────┐
        │  Rank Worker(s)   │     │ JD-Aspect Svc    │        │ Index Builder     │  (batch/cron)
        │  (stateless)      │────▶│ JD text→aspects  │        │ embeddings+BM25   │
        │ recall→feat→gate  │     │ (+ embed queries)│        │ +CE teacher+train │
        │ →rerank→reason    │     └──────────────────┘        └────────┬─────────┘
        └───┬───────┬───────┘                                          │ writes
            │ reads  │ reads                                            ▼
   ┌────────▼──┐  ┌──▼─────────────┐                       ┌────────────────────────┐
   │ Candidate │  │ Embedder + Model│                      │  Artifact Store (S3)    │
   │  Index    │  │  (in-process)   │◀─────────────────────│ embeddings.npy, bm25,   │
   │ (RAM/ANN) │  └─────────────────┘                      │ embedder.pkl, model.bin │
   └───────────┘                                           │ ce_scores, aspects      │
   ┌──────────────────┐                                    └────────────────────────┘
   │ Metadata DB (PG)  │  tenants, users, saved searches, JD history, result snapshots
   └──────────────────┘
```

## 4. Components
| Component | Responsibility | State |
|---|---|---|
| **API Gateway** | TLS, authN/Z, per-tenant rate limits, request routing | stateless |
| **Rank Worker** | the Stage A–F pipeline for one JD; holds the shared index in RAM | stateless (warm cache) |
| **JD-Aspect Service** | turn arbitrary JD text into aspects (must/nice/disqualifier vocab + queries) and embed them with the persisted embedder | stateless |
| **Index Builder** | offline: profile embeddings, BM25, cross-encoder teacher, LambdaMART training; publishes a versioned artifact set | batch |
| **Artifact Store (S3)** | versioned `embeddings.npy`, `bm25`, `embedder.pkl`, `model.bin`, `ce_scores`, `aspects` | durable |
| **Candidate Index** | the 100k embedding matrix (+ optional FAISS/HNSW) + tokenized corpus, memory-mapped per worker | read-only |
| **Metadata DB (Postgres)** | tenants, users, saved JDs, search history, result snapshots, audit log | durable |

## 5. API (sketch)
```
POST /v1/rank
  body: { jd_text: str, top_k?: int=100, dedup?: bool=true }
  → 200 { search_id, results:[{candidate_id, rank, score, reasoning, drivers:[{feature,contrib}]}],
          honeypot_rate, latency_ms }
GET  /v1/search/{search_id}          → saved snapshot (FR-4)
GET  /v1/search/{search_id}/csv      → spec CSV
POST /v1/index/refresh               → enqueue offline rebuild (admin)
GET  /v1/index/version               → active artifact version
```
Multi-tenancy: every row carries `tenant_id`; the candidate index can be global (shared pool)
or per-tenant (BYO candidates) by namespacing the artifact set.

## 6. Request flow (warm path, sync)
```
user → POST /rank {jd_text}
  1. JD-Aspect Svc: jd_text → aspects A; embed queries Q = φ(q_a)        ~50ms
  2. Worker: dense S = E·Qᵀ ; BM25 scores(Q)                            ~300ms (vectorized, cached E+index)
  3. RRF fuse → shortlist M≈8k                                          ~20ms
  4. features Ψ over M (sims slice + cached honeypot flags)             ~1–2s
  5. honeypot gate (precomputed set) → rerank f(x) → ensemble → dedup   ~100ms
  6. reasoning + SHAP drivers + calibrate                               ~100ms
  → results (p95 < 3s)
```
The 465 MB raw JSONL is **never** touched on this path; only the prebuilt matrix + tokens.

## 7. Offline / async path
```
admin → POST /index/refresh → queue → Index Builder:
  stream candidates → embeddings.npy + embedder.pkl  (GPU here if available)
  build BM25 ; cross-encoder teacher over top recall ; train LambdaMART
  → publish artifacts vN+1 to S3 → workers hot-swap on version bump (blue/green)
```
Pre-computation is allowed to be slow (minutes) because it is off the request path (mirrors the
hackathon's "rank step ≤ 5 min" constraint, generalized to "request path stays cheap").

## 8. Scaling & capacity
- **Memory/worker:** $E$ at $N{=}100\text{k}, d{=}384$ float32 ≈ 154 MB; tokens ≈ a few hundred MB; model.bin tiny. Fits a 2–4 GB worker; memory-map $E$ to share across processes.
- **Throughput:** workers are stateless and CPU-bound on steps 2+4; scale horizontally behind the gateway. A single warm worker handles ~0.5–1 rank/s; add replicas for QPS.
- **Index size growth:** beyond ~1–5 M candidates, replace the flat NumPy cosine with an ANN index (FAISS-IVF/HNSW) for sublinear recall; BM25 → OpenSearch.
- **Caching:** identical JD text → cache the result snapshot (content hash key).

## 9. Multi-user concerns
- **Isolation:** results computed per request from shared read-only artifacts; no shared mutable state, so concurrent JDs cannot interfere.
- **Fairness/abuse:** per-tenant rate limits + a queue for `/index/refresh`.
- **Privacy:** candidate PII stays in the artifact/DB tier; the rank path emits only ids + derived reasons; no candidate text leaves the VPC to any LLM (NFR-4).
- **Auditability:** every search snapshot + the artifact `version` is persisted, so any ranking is reproducible (Stage-3-style reproduction, per tenant).

## 10. Failure modes & mitigations
| Failure | Mitigation |
|---|---|
| Artifact store unavailable | workers keep last-good index in RAM; serve stale with a version banner |
| Bad JD (empty / garbage) | JD-Aspect Svc validates; fall back to keyword-only recall + warning |
| Index/model version skew | single atomic `version` pointer; workers refuse mismatched artifact sets |
| Cross-encoder artifact missing | ensemble degrades gracefully to model+rubric (see [MATH.md](MATH.md) §6) |
| Worker OOM on huge pool | memory-map embeddings; shard the index; cap top_k |

## 11. What the repo ships toward this
`src/serve.py` (`rank_jd(jd_text)` reusing cached candidate artifacts) is the in-process Rank
Worker + JD-Aspect Service; `src/app.py` is the Web UI; `build_index.py` (now persisting
`embedder.pkl`) is the Index Builder; `artifacts/` stands in for the Artifact Store. The
Postgres metadata tier, gateway, and S3 are the production substitutions, not needed for the
hackathon submission but designed for here.
