"""
app.py — Gradio demo for a Hugging Face Space.

Serves the SAME TFLite model that runs on the Raspberry Pi, with the SAME
confidence contract (Section 6), so the online demo and the field device agree.

The Space needs only a light stack (see requirements-hf.txt): gradio, numpy,
pillow, and tflite-runtime (or tensorflow-cpu as a fallback). It reuses
predict.predict() so there is one inference code path.

Locally:   python app.py       -> http://127.0.0.1:7860
On HF:     the Space runtime imports `demo` / runs this file automatically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from predict import DEFAULT_MODEL, LABEL_MAP, predict

HERE = Path(__file__).resolve().parent

# Surface the honest data-sourcing decision (Section 3 / 10) in the UI if present.
_decision_path = HERE / "data" / "DATASET_DECISION.json"
if _decision_path.exists():
    _decision = json.loads(_decision_path.read_text(encoding="utf-8"))
    _schema_note = (
        f"**Model schema:** `{_decision.get('schema')}` — "
        f"{_decision.get('num_classes')} classes actually trained."
    )
else:
    _schema_note = (
        "**Model schema:** 5-class as specified. See README for the "
        "data-sourcing decision (anthracnose availability)."
    )


def classify(image, threshold):
    if image is None:
        return {"error": "Please upload a cassava leaf image."}, "—"
    tmp = HERE / "_upload.jpg"
    image.convert("RGB").save(tmp)
    result = predict(tmp, DEFAULT_MODEL, threshold)
    # gr.Label wants {label: prob_fraction}
    label_scores = {k: v / 100.0 for k, v in result["all_class_probabilities"].items()}
    verdict = (
        f"### {result['predicted_class']}\n"
        f"**Confidence: {result['confidence']:.2f}%**"
    )
    return label_scores, verdict


with gr.Blocks(title="AgroControl — Cassava Leaf Disease Prediction") as demo:
    gr.Markdown(
        "# 🌿 AgroControl — Cassava Leaf Disease Prediction Model\n"
        "### Built by Akabuike Daniel\n"
        "AI component of an IoT system for cassava disease management  "
        "for a Raspberry Pi 3B deployment. Upload a leaf photo to get a "
        "disease prediction with a calibrated confidence score.\n\n"
        f"{_schema_note}\n\n"
        "> Below the confidence threshold the model honestly returns "
        "*“Uncertain — please retake photo”* instead of guessing."
    )
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Cassava leaf image")
            thr = gr.Slider(0, 100, value=60, step=1, label="Confidence threshold (%)")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            verdict = gr.Markdown(label="Diagnosis")
            scores = gr.Label(num_top_classes=5, label="Class probabilities")
    btn.click(classify, inputs=[img, thr], outputs=[scores, verdict])
    gr.Markdown(
        "Classes: "
        + ", ".join(LABEL_MAP.values())
        + "\n\n*Research/education prototype — not a substitute for agronomic advice.*"
         + "\n\n*Done by DTRINO .*"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
