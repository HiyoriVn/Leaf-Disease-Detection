"""Reproduce and diagnose LiSA's 0..255 vs 0..1 input saturation.

Run one image through the SAME TFLite checkpoint using two float32 input
ranges, log all probabilities. This is a debugging comparison, NOT a test-set
accuracy measurement or proof the scaling contract is correct.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.rice_inference import load_rice_config, load_rice_interpreter, predict_rice, preprocess_rice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True,
                        help="Local RGB image (prefer labeled Paddy Doctor test images)")
    parser.add_argument("--out", type=Path, default=Path("reports/rice_pixel_scale_comparison.json"))
    args = parser.parse_args()
    config = load_rice_config()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    raw = args.image.read_bytes()
    with Image.open(BytesIO(raw)) as opened:
        img = opened.convert("RGB")
    interpreter = load_rice_interpreter(config)
    records = {}
    for scale in ("0_1", "0_255"):
        probe = replace(config, pixel_scale=scale)
        inp = preprocess_rice(img, probe)
        start = time.perf_counter()
        prediction = predict_rice(img, interpreter, probe)
        elapsed = (time.perf_counter()-start)*1000
        records[scale] = {
            "top1": prediction["predicted_label"],
            "top1_score": prediction["confidence"],
            "input_min": float(inp.min()),
            "input_max": float(inp.max()),
            "runtime_ms": round(elapsed, 3),
            "probabilities": dict(zip(config.labels, prediction["probabilities"])),
        }
    result = {
        "image": str(args.image),
        "image_sha256": sha256(raw).hexdigest(),
        "model_path": str(config.model_path),
        "model_sha256": sha256(config.model_path.read_bytes()).hexdigest(),
        "preprocessing_verified": config.preprocessing_verified,
        "temperature_file_present": (Path(__file__).resolve().parents[1] / "models" / "rice_temperature.json").exists(),
        "notes": ("If temperature_file_present is true, scores include calibration; "
                  "temporarily move the temperature file for raw Softmax comparison. "
                  "A single image cannot establish correct preprocessing."),
        "results": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
