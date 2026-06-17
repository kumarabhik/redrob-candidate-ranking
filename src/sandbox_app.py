"""Gradio sandbox app (deploy to HuggingFace Spaces) — submission_spec §10.5.

Upload a JSON list of <=100 candidate records (same schema as sample_candidates.json) or
click the example; the app runs the ranker end-to-end on CPU and returns a ranked table +
downloadable CSV. Satisfies the mandatory "working hosted sandbox" requirement.

Local run:  python src/sandbox_app.py    (then open the printed URL)
Deploy:     see SANDBOX.md
"""
from __future__ import annotations

import json
import tempfile

import gradio as gr
import pandas as pd

from parse import Candidate
from sandbox_demo import rank_sample


def _run(file_obj, pasted_json: str):
    if file_obj is not None:
        raw = json.loads(open(file_obj.name, encoding="utf-8").read())
    elif pasted_json and pasted_json.strip():
        raw = json.loads(pasted_json)
    else:
        return pd.DataFrame(), None, "Provide a JSON file or paste candidate records."
    if isinstance(raw, dict):
        raw = [raw]
    raw = raw[:100]
    records = [Candidate(r) for r in raw]
    rows, n_gated = rank_sample(records, top=100)
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    df.to_csv(tmp.name, index=False)
    msg = f"Ranked {len(rows)} candidates ({n_gated} honeypots gated) of {len(records)}."
    return df, tmp.name, msg


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown("# Redrob Candidate Ranker — sandbox\n"
                "Ranks the top candidates for the *Senior AI Engineer* JD. "
                "Upload <=100 candidate records (schema = `sample_candidates.json`).")
    with gr.Row():
        f = gr.File(label="candidates JSON", file_types=[".json"])
        txt = gr.Textbox(label="...or paste JSON list", lines=6)
    btn = gr.Button("Rank", variant="primary")
    status = gr.Markdown()
    table = gr.Dataframe(label="Ranked candidates")
    out = gr.File(label="Download ranked CSV")
    btn.click(_run, [f, txt], [table, out, status])


if __name__ == "__main__":
    demo.launch()
