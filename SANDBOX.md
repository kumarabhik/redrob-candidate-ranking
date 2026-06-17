# Sandbox deployment (submission requirement §10.5)

The submission requires a **working hosted sandbox** where organizers can run the ranker on a
small candidate sample. We ship a Gradio app: [src/sandbox_app.py](src/sandbox_app.py).

It accepts a JSON list of ≤100 candidate records (same schema as `sample_candidates.json`),
runs the full ranker (features → honeypot gate → LambdaMART + rubric ensemble) on CPU, and
returns a ranked table + downloadable CSV — well within the 5-minute CPU budget.

## Option A — HuggingFace Spaces (recommended, free)
1. Create a new Space → SDK: **Gradio**.
2. Add these files to the Space repo:
   - `src/` (all modules), `artifacts/aspects.json`, `artifacts/model.bin`,
     `artifacts/feature_names.json` (the small artifacts; the 100k `embeddings.npy` is NOT
     needed — the sandbox embeds the small sample on the fly).
   - `requirements.txt`
   - an `app.py` at the repo root containing:
     ```python
     import sys; sys.path.insert(0, "src")
     from sandbox_app import demo
     demo.launch()
     ```
3. Set the Space to use `requirements.txt`. It will build and serve automatically.
4. Put the Space URL in `submission_metadata.yaml` → `sandbox_link`.

## Option B — Google Colab
Upload the repo, `pip install -r requirements.txt`, then run
`python src/sandbox_app.py` (Gradio prints a public share URL), or call
`rank_sample(...)` from `sandbox_demo.py` in a notebook cell. Share the notebook link.

## Option C — local smoke test (no web)
```bash
python src/sandbox_demo.py --sample "[PUB] .../sample_candidates.json" --out demo_ranked.csv
```

## Notes
- The sandbox embeds the sample with a small TF-IDF+SVD fit on the sample itself, so it does
  not need the 100k embedding matrix. Lexical + behavioral + trap features carry most of the
  signal at small scale; the full semantic recall is exercised by the 100k `rank.py`.
- No network calls happen during ranking in either the sandbox or `rank.py`.
