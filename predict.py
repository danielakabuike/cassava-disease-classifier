"""
predict.py — Raspberry Pi 3B inference (Section 6 & 8).

Loads models/cassava_model.tflite and classifies a single image, returning the
confidence-annotated JSON contract from Section 6:

    {
      "predicted_class": "<name>",
      "confidence": <float 0-100, 2dp>,
      "all_class_probabilities": {class_name: prob_pct, ...}
    }

Below CONFIDENCE_THRESHOLD (default 60%), predicted_class becomes
"Uncertain — please retake photo" instead of a forced label.

Preprocessing (resize→RGB→float32) is minimal because the backbone's own
normalisation is baked into the .tflite graph. This script deliberately has
**zero dependency on the full tensorflow package** — it uses tflite-runtime and
falls back to tf.lite only if the former is missing, with a clear install hint.

Usage on the Pi:
    python predict.py --image leaf.jpg
    python predict.py --image leaf.jpg --threshold 70
    python predict.py --camera            # capture from the Pi camera, then classify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --- label map (single source of truth). Load without importing config so the
# --- Pi doesn't need the training-side modules; fall back to config if present.
HERE = Path(__file__).resolve().parent
try:
    with open(HERE / "label_map.json", "r", encoding="utf-8") as _fh:
        _RAW = json.load(_fh)
    LABEL_MAP = {int(k): v for k, v in _RAW.items()}
except FileNotFoundError:
    LABEL_MAP = {0: "Healthy", 1: "Cassava Mosaic Disease (CMD)",
                 2: "Cassava Bacterial Blight (CBB)",
                 3: "Cassava Brown Streak Disease (CBSD)",
                 4: "Cassava Anthracnose Disease (CAD)"}

# If a pipeline run recorded a data-sourcing decision, its effective label map
# is authoritative — this is what keeps slot 4 honest (e.g. CGM placeholder, not
# anthracnose) so predict.py NEVER reports a class the model wasn't trained on.
_DECISION = HERE / "data" / "DATASET_DECISION.json"
if _DECISION.exists():
    try:
        _elm = json.loads(_DECISION.read_text(encoding="utf-8")).get("effective_label_map")
        if _elm:
            LABEL_MAP = {int(k): v for k, v in _elm.items()}
    except (json.JSONDecodeError, ValueError):
        pass

DEFAULT_MODEL = HERE / "models" / "cassava_model.tflite"
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 60.0
UNCERTAIN_LABEL = "Uncertain — please retake photo"


def _load_interpreter(model_path: Path):
    """Prefer tflite-runtime (light); fall back to tf.lite with a clear hint."""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=str(model_path))
    except ImportError:
        try:
            import tensorflow as tf
            print("[predict] tflite-runtime not found; using full tensorflow. "
                  "On the Pi install the lighter runtime:\n"
                  "  pip install tflite-runtime", file=sys.stderr)
            return tf.lite.Interpreter(model_path=str(model_path))
        except ImportError:
            sys.exit(
                "ERROR: no TFLite runtime available.\n"
                "Install the lightweight runtime on the Raspberry Pi with:\n"
                "  pip install tflite-runtime\n"
                "(or 'pip install tensorflow' on a full workstation).")


def _preprocess(image_path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)          # 0..255; graph normalises
    return np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)


def predict(image_path, model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD):
    image_path, model_path = Path(image_path), Path(model_path)
    if not model_path.exists():
        sys.exit(f"ERROR: model not found at {model_path}. "
                 "Run convert_to_tflite.py first, or copy the .tflite to the Pi.")
    if not image_path.exists():
        sys.exit(f"ERROR: image not found at {image_path}")

    interp = _load_interpreter(model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    x = _preprocess(image_path).astype(inp["dtype"])
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0].astype(np.float64)
    probs = probs / probs.sum() if probs.sum() else probs   # guard

    top = int(np.argmax(probs))
    confidence = round(float(probs[top]) * 100, 2)
    all_probs = {LABEL_MAP.get(i, f"class_{i}"): round(float(p) * 100, 2)
                 for i, p in enumerate(probs)}

    predicted = (LABEL_MAP.get(top, f"class_{top}")
                 if confidence >= threshold else UNCERTAIN_LABEL)
    return {"predicted_class": predicted,
            "confidence": confidence,
            "all_class_probabilities": all_probs}


def _capture_from_camera() -> Path:
    """Capture a frame from the Pi camera. Tries picamera2, then OpenCV."""
    out = HERE / "capture.jpg"
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.start()
        cam.capture_file(str(out))
        cam.stop()
        return out
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(out), frame)
            return out
    except Exception:
        pass
    sys.exit("ERROR: could not capture from camera "
             "(install picamera2 or opencv-python, or pass --image).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path, help="Path to a leaf image.")
    g.add_argument("--camera", action="store_true", help="Capture from Pi camera.")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Confidence %% below which result is 'Uncertain'.")
    args = ap.parse_args()

    image_path = _capture_from_camera() if args.camera else args.image
    result = predict(image_path, args.model, args.threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
