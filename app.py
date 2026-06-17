"""HuggingFace Spaces entrypoint (Gradio SDK looks for app.py at the repo root).

Multi-JD web UI: paste any job description, rank the shared 100k candidate pool, get
scores + fact-grounded reasons + a downloadable CSV. See system_design.md for the design
and SANDBOX.md for deployment. The ranking logic lives in src/serve.py.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr           # noqa: E402
import pandas as pd           # noqa: E402

from config import ARTIFACTS  # noqa: E402
import serve                  # noqa: E402

PRESETS = {
    "Senior AI Engineer (default role)": (
        (ARTIFACTS / "jd_text.txt").read_text(encoding="utf-8")
        if (ARTIFACTS / "jd_text.txt").exists() else
        "Senior AI Engineer. 5-9 years. Production embeddings-based retrieval, vector "
        "databases, learning-to-rank, NDCG/MAP evaluation, strong Python. Pune/Noida."),
    "Senior Frontend Engineer": (
        "Senior Frontend Engineer, 4-7 years. Strong React, TypeScript, Next.js, Redux, "
        "Tailwind CSS, performance and accessibility. Build delightful user-facing products. "
        "Bangalore or remote."),
    "Data Engineer": (
        "Data Engineer, 3-6 years. Build data pipelines with Spark, Kafka, Airflow, dbt, "
        "Snowflake and BigQuery. Strong Python and SQL. ETL at scale. Hyderabad."),
}


def _rank(role_title, jd_text, top_k):
    if not jd_text or not jd_text.strip():
        return pd.DataFrame(), None, "Paste a job description first."
    rows, meta = serve.rank_jd(jd_text, role_title or "Role", top_k=int(top_k))
    df = pd.DataFrame([{
        "rank": r["rank"], "candidate_id": r["candidate_id"],
        "title @ company": f"{r['title']} @ {r['company']}", "years": r["years"],
        "score": r["score"], "reasoning": r["reasoning"]} for r in rows])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    pd.DataFrame([{"candidate_id": r["candidate_id"], "rank": r["rank"],
                   "score": r["score"], "reasoning": r["reasoning"]} for r in rows]
                 ).to_csv(tmp.name, index=False)
    status = (f"Ranked {len(rows)} of {meta['shortlist']} shortlisted "
              f"(honeypots gated; {meta['rank_ms']} ms; ranker: {meta['ranker']}). "
              f"Detected JD focus: {', '.join(meta['must_aspects'])}.")
    return df, tmp.name, status


def _load_preset(name):
    return name.split(" (")[0], PRESETS[name]


with gr.Blocks(title="Candidate Ranker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Candidate Ranker\n"
                "Paste any job description and rank the top candidates from the 100k pool, "
                "with explainable, fact-grounded reasons. Honeypots are filtered automatically.")
    with gr.Row():
        with gr.Column(scale=2):
            preset = gr.Dropdown(list(PRESETS), value=list(PRESETS)[0], label="Start from a template")
            role = gr.Textbox(label="Role title", value="Senior AI Engineer")
            jd = gr.Textbox(label="Job description", lines=12, value=PRESETS[list(PRESETS)[0]])
            with gr.Row():
                topk = gr.Slider(10, 100, value=50, step=10, label="How many candidates")
                btn = gr.Button("Rank candidates", variant="primary")
        with gr.Column(scale=3):
            status = gr.Markdown()
            table = gr.Dataframe(label="Ranked candidates", wrap=True)
            csv = gr.File(label="Download CSV")

    preset.change(_load_preset, preset, [role, jd])
    btn.click(_rank, [role, jd, topk], [table, csv, status])


if __name__ == "__main__":
    print("warming shared candidate index (one-time)...")
    serve._index()
    print("ready.")
    demo.launch()
