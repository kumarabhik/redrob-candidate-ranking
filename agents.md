# agents.md — Contributor & agent guide

Rules and conventions for anyone (human or AI agent) working in this repo. If a vague request conflicts with these docs, the docs win — surface the conflict, don't silently resolve it.

## 0. Read order (every session, do this first)
1. [system.md](system.md) — what's actually installed and the latest checkpoint (don't guess tool versions).
2. [roadmap.md](roadmap.md) — pick the next `[ ]` task; understand the phase gates.
3. [design_doc.md](design_doc.md) — the authoritative architecture and the three research pillars.
4. [explanation.md](explanation.md) — plain-language model of the system (and interview answers).

## 1. Hard rules (non-negotiable; these are what the hackathon disqualifies on)
- **No network in the ranking step.** `rank.py` and everything it imports at rank time must make zero external calls — no OpenAI/Anthropic/Cohere/Gemini/any hosted service. (NFR-002)
- **Rank step ≤ 5 min, ≤ 16 GB RAM, CPU only.** Anything that doesn't fit goes to offline pre-compute. (NFR-001/003)
- **Honeypots never enter the top 100.** Run the honeypot gate before reranking; self-check the rate at emit. (FR-006)
- **Every reasoning claim traces to a real field.** If the source field doesn't exist or doesn't cross threshold, don't say it. No invented skills, employers, or experience. (FR-007)
- **Spec-valid CSV or it doesn't ship.** Exactly 100 rows, each rank 1–100 once, score non-increasing, ties by `candidate_id` asc, ids match `^CAND_[0-9]{7}$`. Always run `validate_submission.py`. (FR-001/002/003/008)
- **One reproduce command.** The CSV must regenerate from `python rank.py --candidates ... --out ...` with no manual edits. (NFR-004)

## 2. Local dev loop
```
# pre-compute (offline, network OK) — run once / when artifacts change
python src/jd_aspects.py            # JD → artifacts/aspects.json
python src/build_index.py           # → artifacts/bm25.idx, embeddings.npy
python src/train.py                 # → artifacts/model.bin

# rank step (the thing that gets reproduced at Stage 3)
python src/rank.py --candidates candidates.jsonl --out submission.csv
python "[PUB] .../validate_submission.py" submission.csv   # must pass
```

## 3. Conventions
- Python 3.11; format with `black`; type-hint public functions.
- Pure functions for feature/score logic (easy to unit-test, no hidden state).
- Read big files by streaming (`candidates.jsonl` is 465 MB) — never `json.load` the whole thing.
- All tunables (RRF `k`, shortlist size, thresholds) live in one config block, not scattered literals.
- Field access uses the verbatim schema names from `candidate_schema.json` — no renaming.

## 4. Where to put new things
| You're adding… | Put it in… | First update… |
|---|---|---|
| A new candidate feature | `src/features.py` | [design_doc.md](design_doc.md) §7.2 feature table |
| A new honeypot rule | `src/honeypot.py` | [design_doc.md](design_doc.md) §7.3 |
| A new JD aspect | `src/jd_aspects.py` → `aspects.json` | [design_doc.md](design_doc.md) §5.2 |
| A retrieval change | `src/recall.py` | [roadmap.md](roadmap.md) §4 tech-stack table |
| A reranker/label change | `src/train.py` / `src/rank.py` | [design_doc.md](design_doc.md) §7.4 |
| A reasoning template | `src/reasoning.py` | [design_doc.md](design_doc.md) §7.5 |

## 5. Roadmap workflow
- Work the next `[ ]` in [roadmap.md](roadmap.md) §6. Flip to `[~]` when you start (one `[~]` per agent at a time).
- Flip to `[x]` only when: (1) code merged, (2) a test added/updated, (3) the relevant doc section updated. Never delete `[x]` items — they're the audit trail.

## 6. Testing strategy
- **Unit**: honeypot rules (hand-built impossible profiles), feature extraction (on `sample_candidates.json`), CSV formatting.
- **Sanity**: recall contains known-plausible samples; offline NDCG on a ~30-row hand-labeled set beats a BM25-only baseline.
- **Integration**: full `rank.py` run → `validate_submission.py` zero violations + runtime under 5 min + honeypot rate < 10%.
- A phase isn't `[x]` without at least one test.

## 7. Common pitfalls
- Loading the whole 465 MB jsonl into memory — stream it.
- Comparing BM25 and cosine scores directly — they're incomparable; fuse by rank (RRF).
- Forgetting the score column must be non-increasing — sort then assign monotonic scores.
- Letting `duration_months` features leak honeypot signals as "strong" — gate first.
- Embedding at rank time — embeddings are a precomputed artifact.

## 8. Anti-patterns (PR rejected)
- Any hosted-LLM call inside the rank step.
- Reasoning that mentions skills/employers not in the profile (hallucination).
- All-identical or name-only-templated reasoning.
- Keyword-count ranking (the JD explicitly calls this a trap).
- GPU dependency or >16 GB peak in the rank step.

## 9. PR checklist (paste into every PR)
- [ ] What changed & why
- [ ] Roadmap item flipped (`[~]`/`[x]`) and which
- [ ] Test added/updated
- [ ] Doc section updated (design_doc/roadmap/system as relevant)
- [ ] Rank step still no-network, ≤5 min, ≤16 GB (if touched)
- [ ] `validate_submission.py` still passes (if output path touched)

## 10. Glossary
- **NDCG@k** — Normalized Discounted Cumulative Gain over top k; rewards putting high-relevance candidates near the top. The dominant scoring metric.
- **MAP / P@k** — Mean Average Precision / Precision at k; the smaller-weight components.
- **RRF** — Reciprocal Rank Fusion; combine ranked lists by `Σ 1/(k+rank)`, training-free. (Pillar 1)
- **Person-Job Fit / aspect-fit** — modeling the match as scored requirement aspects, not keyword overlap. (Pillar 2)
- **LambdaMART** — gradient-boosted-tree ranker that directly optimizes NDCG. (Pillar 3)
- **Honeypot** — a deliberately impossible profile; must be kept out of the top 100.
- **Tier-5** — a genuinely strong candidate (the JD's term) who may not use buzzwords.
- **Weak labels** — relevance tiers derived from the JD rubric since no ground truth is shipped.

## 11. When this file is wrong
If reality contradicts a rule here, fix the rule in the same PR that exposes it, and note it in the [system.md](system.md) checkpoint. Stale rules are worse than missing ones.
