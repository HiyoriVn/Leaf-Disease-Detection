"""Diagnostic in-source evaluation of the published cassava checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml

from src.cassava_inference import decode_image, load_cassava_model, load_config, predict_image
from src.classification_metrics import classification_metrics


def select_stratified(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return sorted(rows, key=lambda row: row["image_path"])
    by_class = {i: [r for r in rows if int(r["label_index"]) == i] for i in range(5)}
    desired = {i: limit * len(group) / len(rows) for i, group in by_class.items()}
    counts = {i: int(desired[i]) for i in by_class}
    for i in sorted(by_class, key=lambda i: desired[i] - counts[i], reverse=True)[:limit - sum(counts.values())]:
        counts[i] += 1
    rng = random.Random(seed)
    return sorted((r for i, group in by_class.items() for r in rng.sample(group, counts[i])), key=lambda row: row["image_path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "internal_test"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "baseline_cassava")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    dataset_cfg = yaml.safe_load((ROOT / "configs" / "dataset_config.yaml").read_text(encoding="utf-8"))
    data_root = Path(dataset_cfg["data_root"])
    manifest_path = ROOT / "manifests" / "cassava.csv"
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == args.split and not r["exclusion_reason"]]
    selected = select_stratified(rows, args.limit, int(dataset_cfg["seed"]))
    torch.set_num_threads(4)
    config = load_config()
    model = load_cassava_model(config)
    predictions = []
    start = time.perf_counter()
    for row in selected:
        image = decode_image((data_root / row["image_path"]).read_bytes(), config)
        result = predict_image(image, model, config)
        predictions.append({"image_path": row["image_path"], "true_index": int(row["label_index"]),
                            "true_label": row["canonical_label"], "pred_index": result["predicted_index"],
                            "pred_label": result["predicted_label"],
                            **{f"prob_{i}": p for i, p in enumerate(result["probabilities"])}})
    metrics = classification_metrics([r["true_index"] for r in predictions],
                                     [r["pred_index"] for r in predictions], config.class_names)
    metrics.update({"scope": "diagnostic in-source subset; published checkpoint trained on same Kaggle source; potentially contaminated",
                    "split": args.split, "requested_limit": args.limit, "seed": dataset_cfg["seed"],
                    "duration_seconds": round(time.perf_counter() - start, 2),
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "checkpoint_sha256": config.checkpoint_sha256, "class_names": config.class_names})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
