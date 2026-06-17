---
title: Candidate Ranker
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
short_description: Rank candidates for any JD with explainable, honeypot-filtered results
---

# Candidate Ranker — sandbox

Paste any job description and rank the top candidates from a 100k pool, with fact-grounded
reasons and honeypot filtering. Built for the INDIA.RUNS (Redrob) challenge.

**To deploy this Space:**
1. Create a new HuggingFace Space → SDK: **Gradio**.
2. Push this repo's `app.py`, `src/`, `requirements.txt`, and the small artifacts
   (`artifacts/aspects.json`, `model.bin`, `feature_names.json`, `embedder.pkl`,
   `embeddings.npy`, `candidate_ids.json`, `index_meta.json`, `jd_text.txt`).
3. Rename **this file** to `README.md` in the Space (the frontmatter above configures it).
4. The Space builds from `requirements.txt` and serves `app.py` automatically.

See [SANDBOX.md](SANDBOX.md) for alternatives (Colab, Docker, local).
