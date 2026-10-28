"""Evaluate a model per crop against a labeled directory tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .inference import ModelError, load_manifest, load_model, predict

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def evaluate_crop(root: Path, crop: str, config, model) -> dict:
    crop_dir = root / crop
    if not crop_dir.is_dir():
        raise ModelError(f"Dataset folder missing: {crop_dir}")
    truth: list[int] = []
    predicted: list[int] = []
    unknown = []
    label_to_index = {label: i for i, label in enumerate(config.labels)}
    for label_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
        if label_dir.name not in label_to_index:
            unknown.append(label_dir.name)
            continue
        for image_path in sorted(p for p in label_dir.rglob("*") if p.suffix.lower() in EXTENSIONS):
            with Image.open(image_path) as image:
                prediction = predict(image, config, model, top_k=1)[0]["label"]
            truth.append(label_to_index[label_dir.name])
            predicted.append(label_to_index[prediction])
    if unknown:
        raise ModelError(f"{crop}: dataset class folders not in model labels: {', '.join(unknown)}")
    if not truth:
        raise ModelError(f"{crop}: no labeled images found under {crop_dir}.")
    matrix = confusion_matrix(truth, predicted, labels=np.arange(len(config.labels)))
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(f1_score(truth, predicted, labels=np.arange(len(config.labels)), average="macro", zero_division=0)), "labels": list(config.labels), "confusion_matrix": matrix.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root folder containing rice/, coffee/, cassava/ class folders")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    configs = load_manifest()
    results = {crop: evaluate_crop(args.data, crop, config, load_model(config)) for crop, config in configs.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "evaluation.json"
    destination.write_text(json.dumps({"scope": "local experiment", "dataset_root": str(args.data.resolve()), "results": results}, indent=2), encoding="utf-8")
    print(f"Saved experimental metrics to {destination}")
    for crop, values in results.items():
        print(f"{crop}: n={values['n']} accuracy={values['accuracy']:.4f} macro_f1={values['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
