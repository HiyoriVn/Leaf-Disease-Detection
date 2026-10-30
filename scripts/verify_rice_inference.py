"""P0: compare candidate LiSA inference preprocessing with saved TEST predictions.

Requires ORIGINAL test images, Keras model, TFLite model and the repository's
test_predictions.csv. Agreement of Keras/TFLite alone is insufficient; matching
saved notebook probabilities is the strongest available local reference.
Run this BEFORE marking preprocessing_verified=true.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "Rice-leaf" / "riceLeafDisease_trainingResult"


def as_prob(scores):
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(s) != 10 or not np.all(np.isfinite(s)):
        raise ValueError("Expected 10 finite output values")
    if np.all(s >= 0) and np.all(s <= 1) and abs(s.sum() - 1.) < 1e-3:
        return s / s.sum()
    t = np.exp(s - s.max())
    return t / t.sum()


def prepare(img, method, scale, tf=None):
    img = ImageOps.exif_transpose(img).convert("RGB")
    if method == "pil_bilinear":
        x = np.asarray(img.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
    elif method == "pil_bicubic":
        x = np.asarray(img.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32)
    elif method == "tf_bilinear":
        if tf is None:
            raise RuntimeError("TensorFlow required for tf_bilinear")
        x = tf.image.resize(np.asarray(img), (224, 224), method="bilinear").numpy()
    else:
        raise ValueError(method)
    if scale == "0_1":
        x = x / 255.0
    return x[np.newaxis, ...].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="Directory containing the 10 class folders of original Paddy train_images")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--report", type=Path, default=Path("reports/rice_contract_local.json"))
    args = ap.parse_args()
    from ai_edge_litert.interpreter import Interpreter
    import tensorflow as tf

    classes = json.loads((ART / "class_names.json").read_text(encoding="utf-8"))
    with (ART / "test_predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = rows[::max(1, len(rows) // max(args.samples, 1))][:args.samples]
    images = []
    for row in rows:
        source = Path(row["image_path"])
        path = args.dataset_root / source.parent.name / source.name
        if not path.is_file():
            raise FileNotFoundError(f"Missing original image: {path}")
        with Image.open(path) as opened:
            images.append(opened.copy())

    try:
        keras_model = tf.keras.models.load_model(ART / "best_model.keras",
                                                  compile=False, safe_mode=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load Keras checkpoint. Custom LiSA layers may need their "
            "original code and custom_objects registration. No verification performed."
        ) from exc
    interpreter = Interpreter(model_path=str(ART / "lisa_paddy_fp32.tflite"))
    interpreter.allocate_tensors()
    ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(ins) != 1 or len(outs) != 1:
        raise RuntimeError("Unexpected TFLite input/output count")
    if tuple(ins[0]["shape"]) != (1, 224, 224, 3) or ins[0]["dtype"] != np.float32:
        raise RuntimeError(f"Unexpected TFLite input: {ins[0]}")
    if int(np.prod(outs[0]["shape"])) != 10:
        raise RuntimeError(f"Unexpected TFLite output: {outs[0]}")
    reference = np.array([[float(row["p_" + name]) for name in classes] for row in rows])
    scores = []
    for method in ("pil_bilinear", "pil_bicubic", "tf_bilinear"):
        for scale in ("0_255", "0_1"):
            ref_err, parity_err, top1_ref, top1_pair = [], [], [], []
            for image, p_ref in zip(images, reference):
                x = prepare(image, method, scale, tf=tf)
                interpreter.set_tensor(ins[0]["index"], x)
                interpreter.invoke()
                pt = as_prob(interpreter.get_tensor(outs[0]["index"]))
                pk = as_prob(keras_model(x, training=False).numpy())
                ref_err.append(np.mean(np.abs(pt - p_ref)))
                parity_err.append(np.max(np.abs(pt - pk)))
                top1_ref.append(int(pt.argmax() == p_ref.argmax()))
                top1_pair.append(int(pt.argmax() == pk.argmax()))
            scores.append({
                "resize": method, "pixel_scale": scale,
                "tflite_vs_notebook_mae": round(float(np.mean(ref_err)), 8),
                "tflite_vs_keras_max_abs_avg": round(float(np.mean(parity_err)), 8),
                "top1_agreement_notebook": round(float(np.mean(top1_ref)), 4),
                "top1_agreement_keras": round(float(np.mean(top1_pair)), 4),
            })
    scores.sort(key=lambda r: (r["tflite_vs_notebook_mae"], -r["top1_agreement_notebook"]))
    best = scores[0]
    verified = (best["tflite_vs_notebook_mae"] <= .005
                and best["top1_agreement_notebook"] >= .98
                and best["tflite_vs_keras_max_abs_avg"] <= .005)
    report = {
        "samples": len(images),
        "model": "LiSA TFLite fp32",
        "candidate_results": scores,
        "candidate_passed_strict_thresholds": bool(verified),
        "notes": "Even a pass should be reviewed against actual notebook transforms; "
                 "test images are used only for post-hoc contract validation, not for training.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not verified:
        raise SystemExit("P0 FAIL: preprocessing not verified; keep preprocessing_verified=false.")


if __name__ == "__main__":
    main()
