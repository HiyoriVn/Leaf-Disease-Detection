"""Inspect LiSA TFLite without Streamlit. Does not depend on OOD gate."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from src.rice_inference import load_rice_config, load_rice_interpreter, predict_rice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="Real photo to test (recommended)")
    parser.add_argument("--out", type=Path, default=Path("reports/rice_smoke_local.json"))
    args = parser.parse_args()
    cfg = load_rice_config()
    model = cfg.model_path
    report = {
        "model_file_found": model.is_file(), "model_path": str(model),
        "model_expected_bytes": cfg.expected_model_bytes,
        "preprocessing_verified": cfg.preprocessing_verified,
        "image_used": str(args.image) if args.image else "synthetic RGB test (NOT diagnosis)",
    }
    if not model.is_file():
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    report["model_actual_bytes"] = model.stat().st_size
    report["model_sha256"] = sha256(model.read_bytes()).hexdigest()
    if report["model_actual_bytes"] != cfg.expected_model_bytes:
        raise SystemExit("Model size differs; checkpoint must be checked")
    t0 = time.perf_counter()
    interpreter = load_rice_interpreter(cfg)
    report["model_load_ms"] = round((time.perf_counter()-t0)*1000, 2)
    report["input"] = {
        "shape": list(map(int, interpreter.get_input_details()[0]["shape"])),
        "dtype": str(interpreter.get_input_details()[0]["dtype"]),
    }
    report["output"] = {
        "shape": list(map(int, interpreter.get_output_details()[0]["shape"])),
        "dtype": str(interpreter.get_output_details()[0]["dtype"]),
    }
    if args.image:
        with Image.open(args.image) as im:
            image = im.convert("RGB")
    else:
        image = Image.new("RGB", (640, 480), (40, 140, 40))
    t0 = time.perf_counter()
    result = predict_rice(image, interpreter, cfg)
    report["inference_ms"] = round((time.perf_counter()-t0)*1000, 2)
    report["tflite_invoke_completed"] = True
    report["top1"] = result["predicted_label"]
    report["top1_score"] = result["confidence"]
    report["probability_sum"] = sum(result["probabilities"])
    report["all_classes"] = dict(zip(cfg.labels, result["probabilities"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("NOTE: This smoke test confirms the interpreter executes, NOT the prediction is correct.")


if __name__ == "__main__":
    main()
