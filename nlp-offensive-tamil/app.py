"""Gradio demo + REST API for the Tamil-English offensive language detector.

Standalone:  python app.py [--share]      (expects ./artifacts produced by the notebook)
API usage:   from gradio_client import Client
             Client("<url>").predict("comment text", "MuRIL transformer (fine-tuned)", api_name="/classify")
"""
import json
import os
import sys

import gradio as gr
import pandas as pd

from nlp_core import PRETTY, BaselinePredictor, TransformerPredictor

ENGINES = ["MuRIL transformer (fine-tuned)", "TF-IDF + linear baseline"]


def verdict_markdown(r):
    if r["status"] == "empty_input":
        return "### ✋ Nothing to classify\nType or paste a comment first."
    lab, conf = r["label"], r["confidence"]
    icon = "✅" if lab == "Not_offensive" else ("🌐" if lab == "Not_Tamil" else "⚠️")
    head = f"## {icon} {PRETTY[lab]}"
    if r["status"] == "uncertain":
        head = f"## ❓ Uncertain - needs human review\nBest guess: **{PRETTY[lab]}**"
    elif r["status"] == "out_of_scope_language":
        head = f"## 🌐 Out-of-scope language\nModel's guess: **{PRETTY[lab]}**"
    script = (r.get("script") or {}).get("type", "-")
    sub = f"Confidence **{conf:.0%}** · {r['latency_ms']:.0f} ms · script: *{script}* · window chunks: {r['n_chunks']}"
    notes = "\n".join(f"- {w}" for w in r["warnings"])
    return head + "\n\n" + sub + (f"\n\n**Notes**\n{notes}" if notes else "")


def build_demo(tf_pred, base_pred=None, examples=None):
    engines = {ENGINES[0]: tf_pred}
    if base_pred is not None:
        engines[ENGINES[1]] = base_pred

    def classify(text, engine):
        r = engines[engine].predict(text)
        probs = {PRETTY[k]: v for k, v in r["probs"].items()} if r["probs"] else None
        keys = ["status", "label", "confidence", "latency_ms", "n_chunks", "aggregation", "script",
                "n_tokens", "unk_ratio", "tokens_per_word", "cleaned", "warnings"]
        return verdict_markdown(r), probs, {k: r.get(k) for k in keys}

    def classify_batch(block, engine):
        rows = []
        for line in [l for l in (block or "").splitlines() if l.strip()][:200]:
            r = engines[engine].predict(line)
            rows.append({"comment": line[:120], "verdict": PRETTY.get(r["label"], "-"),
                         "confidence": r["confidence"], "status": r["status"]})
        return pd.DataFrame(rows, columns=["comment", "verdict", "confidence", "status"])

    with gr.Blocks(title="Tamil-English Offensive Comment Detector") as demo:
        gr.Markdown("# 🛡️ Tamil–English Offensive Comment Detector\n"
                    "Detects offensive language in **code-mixed Tamil–English** comments (Tanglish, Tamil script "
                    "or mixed) and identifies the **target** of the offence. It abstains when unsure and flags "
                    "out-of-scope languages.")
        with gr.Tab("Single comment"):
            with gr.Row():
                with gr.Column(scale=3):
                    inp = gr.Textbox(label="Comment", lines=4,
                                     placeholder="e.g. Padam vera level, thalaivar mass entry 🔥")
                    engine = gr.Radio(list(engines), value=ENGINES[0], label="Model")
                    with gr.Row():
                        btn = gr.Button("Classify", variant="primary")
                        clr = gr.Button("Clear")
                with gr.Column(scale=2):
                    verdict = gr.Markdown()
                    probs = gr.Label(label="Class probabilities", num_top_classes=6)
            details = gr.JSON(label="Details: preprocessing, edge-case flags, latency")
            if examples:
                gr.Examples(examples=[[e] for e in examples], inputs=[inp],
                            label="Examples (held-out test comments + edge cases)")
            btn.click(classify, [inp, engine], [verdict, probs, details], api_name="classify")
            inp.submit(classify, [inp, engine], [verdict, probs, details])
            clr.click(lambda: ("", "", None, None), None, [inp, verdict, probs, details])
        with gr.Tab("Batch"):
            binp = gr.Textbox(label="One comment per line (max 200)", lines=10)
            bengine = gr.Radio(list(engines), value=ENGINES[0], label="Model")
            bbtn = gr.Button("Classify all", variant="primary")
            bout = gr.Dataframe(label="Results")
            bbtn.click(classify_batch, [binp, bengine], bout, api_name="classify_batch")
        with gr.Tab("About"):
            gr.Markdown(
                "**Model:** MuRIL fine-tuned on Offenseval Dravidian (Tamil), CC BY 4.0.  \n"
                "**Pipeline:** input guard → normalisation → WordPiece + sliding window → MuRIL → "
                "max-offensive-chunk aggregation → confidence threshold τ → script guard.  \n"
                "**Statuses:** `ok`, `uncertain` (confidence < τ, send to human), "
                "`out_of_scope_language`, `empty_input`.  \n"
                "**Intended use:** assist human moderators; not for fully automatic removal.")
    return demo


if __name__ == "__main__":
    art = os.environ.get("ARTIFACT_DIR", "artifacts")
    tf = TransformerPredictor.from_artifacts(art)
    bl = BaselinePredictor.from_artifacts(art)
    ex_path = os.path.join(art, "demo_examples.json")
    ex = json.load(open(ex_path)) if os.path.exists(ex_path) else None
    build_demo(tf, bl, ex).launch(share="--share" in sys.argv)