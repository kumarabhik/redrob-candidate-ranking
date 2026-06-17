# explanation.md — Plain-language walkthrough & interview prep

This is the "explain it to a smart friend" doc, tuned to the hackathon's **Stage-5 defend-your-work interview**. If you can talk through this file, you can defend the system.
Related: [design_doc.md](design_doc.md) · [roadmap.md](roadmap.md) · [agents.md](agents.md) · [system.md](system.md)

---

## 1. The project in one line
Given one job description and 100,000 candidate profiles, we return the 100 best-fit people — ranked, scored, and each with an honest one-line reason — in under 5 minutes on a laptop CPU with no internet.

## 2. High-level design (HLD) — the three ideas, in plain words

Think of it as **find → judge → order**, with a **bouncer** at the door and a **clerk** who writes the notes.

**Find (Pillar 1 — Hybrid retrieval + RRF).** We can't carefully score all 100K people in 5 minutes, so first we cast a wide net and pull a shortlist of ~1–2K. We do it two ways at once: a keyword search (BM25) that catches people who literally describe building search/ranking/recommendation systems, and a meaning-based search (embeddings) that catches strong people who describe the same work in different words. We merge the two lists with **Reciprocal Rank Fusion** — a trick that combines rankings without needing the two scoring scales to agree. *Why this over one method:* keyword search misses paraphrases; meaning search misses exact must-have terms. Together they cover each other.

**Judge (Pillar 2 — Person-Job-Fit aspect scoring).** The JD says, almost in these words, "don't just count AI keywords — reason about what we mean." So we break the JD into **aspects**: must-haves (production embeddings/retrieval experience), disqualifiers (consulting-only career, pure-research-no-production), and behavioral modifiers (is the person actually available?). Each candidate gets a fit score per aspect, with the evidence that produced it. *Why:* this is exactly how recruitment research (the APJFNN/person-job-fit line) models a match, and it gives us interpretable evidence we reuse for the reasons.

**Order (Pillar 3 — LambdaMART).** We feed all those features into a small gradient-boosted-tree ranker (LambdaMART) that **directly optimizes NDCG** — the very metric we're scored on. It learns non-linear combinations ("lots of experience only helps if it's at a product company") that a hand-weighted formula would miss, and it runs in milliseconds on CPU. *Why not a neural ranker:* slower, hungrier, and we have no labeled data to feed it; trees are the right tool here.

**The bouncer (honeypot gate).** Before ordering, we throw out the ~80 "impossible" profiles (8 years at a 3-year-old company, "expert" in a skill used 0 months). Simple consistency rules. Ranking these in your top 100 above 10% is an instant disqualification, so this is cheap insurance.

**The clerk (reasoning).** For each of the final 100, we assemble a one-to-two-sentence reason **only from fields that actually exist** in their profile — years + title, the strongest matched aspect, and the most honest concern. No language model at this step, so it literally cannot make something up.

## 3. Low-level design (LLD) — what owns what
| Stage | File (target) | Job |
|---|---|---|
| Load | `src/parse.py` | stream 465 MB jsonl → typed records, build profile text |
| Bouncer | `src/honeypot.py` | consistency rules → honeypot flags |
| Find | `src/recall.py` | BM25 + dense + RRF → ~1–2K shortlist |
| JD aspects | `src/jd_aspects.py` | JD → `aspects.json` (offline, hand-reviewed) |
| Judge | `src/features.py` | per-candidate feature vector from schema fields |
| Order | `src/train.py` + `src/rank.py` | weak-label → LambdaMART; score → top 100 |
| Clerk | `src/reasoning.py` | fact-grounded sentences |
| Entry | `src/rank.py` | wires it all; one command → CSV |

Full field-level detail is in [design_doc.md](design_doc.md) §7.

## 4. End-to-end trace: "JD in → ranked 100 out"
1. Offline: JD → `aspects.json`; all profiles → BM25 index + embedding matrix; weak labels → trained `model.bin`. (Network allowed here.)
2. `rank.py` streams `candidates.jsonl`.
3. **Recall**: BM25 top-N ∪ dense top-N → RRF → ~1–2K shortlist.
4. **Features**: each shortlisted candidate → aspect-fit + seniority + product-vs-services + behavioral-availability + honeypot flags.
5. **Gate**: drop honeypots.
6. **Rerank**: LambdaMART scores → sort desc, ties by `candidate_id` asc → top 100.
7. **Reason**: build one honest line each from real fields.
8. **Emit**: CSV (`candidate_id,rank,score,reasoning`), scores non-increasing → run `validate_submission.py` → log honeypot rate. Done, in < 5 min.

## 5. Likely interview questions (and how to answer)

**Q: Why not just call GPT-4/Claude on each candidate?**
100,000 calls won't fit a 5-minute CPU budget and the spec bans network calls in the ranking step — it's modeling a real production system that has to scale to 200K candidates with low latency. Our LLM-equivalent reasoning happens offline (JD aspect extraction) and the rank step is a small local model over precomputed features.

**Q: You have no labels — how did you train the ranker?**
Weak supervision. We derive a relevance tier (0–5) for a training set from the JD's own explicit rubric — must-have hits, disqualifier hits, and behavioral availability — and force honeypots to tier 0. LambdaMART then learns the non-linear shape of that rubric. We sanity-check against ~30 candidates we hand-labeled so the weak labels aren't drifting.

**Q: How do you avoid the honeypots?**
A deterministic gate before ranking: skill duration exceeding total experience, "expert" with zero months, current role with an end date, salary min > max, tenure exceeding company age, etc. They're impossible by construction, so simple consistency checks catch them. We also log the top-100 honeypot rate at emit as a safety net.

**Q: How is the reasoning not hallucinated?**
It's assembled from evidence slots that are only filled when the underlying field exists and crosses a threshold — no free-text generation at rank time. If a candidate has no GitHub signal, the reason never mentions GitHub. That's why we can guarantee zero hallucination, which is exactly what Stage-4 checks.

**Q: Why LambdaMART and not a deep ranker?**
Three reasons: it directly optimizes NDCG (our scoring metric), it runs in milliseconds on CPU (our budget), and it works well with a few dozen engineered features and no large labeled set — which is our situation. A neural ranker would need a GPU and more labels for no expected gain at this scale.

**Q: How does keyword-stuffing not beat you?**
Two defenses. The JD explicitly calls keyword-counting a trap, so our features weight *where* and *how* a skill was used (career history, product company, duration) over mere presence in a skills list, and the disqualifier aspects actively penalize wrong-role profiles (e.g. a "Marketing Manager" with a perfect skill list). The reranker learns these interactions rather than rewarding raw keyword count.

**Q: You train the ranker on labels you made up — isn't that circular?**
Partly, yes, and we're honest about it: the LambdaMART labels come from our JD rubric, so the model largely learns the rubric's shape. We mitigate it two ways. First, we late-fuse an **offline cross-encoder teacher** — a different model family that jointly reads (JD, profile) and gives a semi-independent relevance signal — so the final score isn't purely our rubric. Second, our ablation is explicit that the LTR-vs-rubric number is optimistic; the unbiased part (BM25 vs dense vs RRF) still shows fusion helps. With ground-truth labels we'd retrain directly on them.

**Q: Where does the cross-encoder run, given no network at rank time?**
Offline. We precompute cross-encoder scores for the top recall candidates into an artifact (`ce_scores.json`) and just look them up during the network-free rank step. It's the standard retrieve-then-rerank teacher pattern: the expensive model runs once offline, its signal is distilled into a cheap lookup + blend. ~600 pairs score in ~28s on CPU.

**Q: How did you validate quality with no leaderboard?**
A trap probe and an ablation. The probe builds one ideal candidate and one of each documented trap (Marketing-Manager-with-AI-skills, keyword-stuffer, consulting-only, honeypot, junior) and asserts our ranker puts the ideal above all of them — it passes. The ablation compares BM25/dense/RRF/+LambdaMART on the competition metrics. We also self-check the top-100 honeypot rate (0%) on every run.

**Q: Why TF-IDF+LSA embeddings instead of a transformer?**
Pragmatism under the CPU budget: a transformer bi-encoder over 100k profiles was too slow on our machine, while TF-IDF+SVD builds in ~4.5 min offline and gives strong recall (top picks are exactly product-company retrieval/ranking engineers). The architecture is embedder-agnostic (one flag swaps in MiniLM/BGE), and the cross-encoder teacher recovers the semantic precision where it matters most — the top of the list.

**Q: What would you improve with more time?**
Swap in MiniLM/BGE bi-encoder embeddings (offline), a learning-to-rank model trained on real recruiter-feedback labels instead of the weak rubric, behavioral-twin de-duplication in the top 100, and a fairness audit.
