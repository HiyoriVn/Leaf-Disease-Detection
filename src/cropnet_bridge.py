"""Run optional CropNet in its own Python environment from the PyTorch app."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-cropnet" / "Scripts" / "python.exe"
MODEL = ROOT / "checkpoints" / "cropnet" / "saved_model.pb"
MODEL_DATA = ROOT / "checkpoints" / "cropnet" / "variables" / "variables.data-00000-of-00001"
MODEL_INDEX = ROOT / "checkpoints" / "cropnet" / "variables" / "variables.index"
CLASS_NAMES = (
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
    "Unknown",
)


def available() -> bool:
    return all(path.is_file() for path in (PYTHON, MODEL, MODEL_DATA, MODEL_INDEX))


def predict_cropnet(content: bytes) -> dict:
    if not available():
        raise RuntimeError("CropNet environment or SavedModel is missing")
    try:
        process = subprocess.run(
            [str(PYTHON), "-m", "src.cropnet_worker"],
            input=content, capture_output=True, cwd=ROOT, timeout=90, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"CropNet worker failed: {exc}") from exc
    if process.returncode != 0:
        details = process.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError(f"CropNet worker failed: {details[-1] if details else process.returncode}")
    try:
        result = json.loads(process.stdout.decode("utf-8"))
        probs = result["probabilities"]
        if len(probs) != 6 or not 0.9999 <= sum(probs) <= 1.0001:
            raise ValueError("invalid probability distribution")
        if result["predicted_label"] != CLASS_NAMES[result["predicted_index"]]:
            raise ValueError("invalid class mapping")
        order = sorted(range(6), key=lambda i: probs[i], reverse=True)
        result["confidence"] = probs[order[0]]
        result["top3"] = [{"index": i, "label": CLASS_NAMES[i], "probability": probs[i]} for i in order[:3]]
        return result
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Invalid CropNet worker output: {exc}") from exc
