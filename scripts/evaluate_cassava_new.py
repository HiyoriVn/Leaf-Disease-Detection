"""One-time internal-test evaluation of the locked validation-selected checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import timm
import torch
from torch.utils.data import DataLoader
import yaml

from scripts.train_cassava import CassavaDataset, hash_file, read_manifest
from src.cassava_inference import EXPECTED_LABELS
from src.classification_metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "cassava.csv")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"test result already exists; refusing to repeat: {args.output}")
    dataset_config = yaml.safe_load((ROOT / "configs" / "dataset_config.yaml").read_text(encoding="utf-8"))
    root = args.data_root or Path(dataset_config["data_root"])
    rows, counts = read_manifest(args.manifest, root)
    test_rows = [r for r in rows if r["split"] == "internal_test"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint["class_names"]) != EXPECTED_LABELS:
        raise ValueError("checkpoint class order differs from frozen label contract")
    if checkpoint["preprocessing"] != {"resize": 256, "crop": 224, "interpolation": "INTER_AREA",
                                       "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}:
        raise ValueError("checkpoint preprocessing differs from evaluation")
    model = timm.create_model(checkpoint["architecture"], pretrained=False, num_classes=5)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(CassavaDataset(test_rows, root, False, 0), batch_size=16 if device.type == "cuda" else 8,
                        num_workers=2, pin_memory=device.type == "cuda")
    true, pred, predictions = [], [], []
    started = time.perf_counter()
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if tuple(logits.shape) != (len(labels), 5):
                raise ValueError("unexpected output shape")
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            for j, probability in enumerate(probs):
                label = int(labels[j])
                prediction = max(range(5), key=lambda i: probability[i])
                row = test_rows[offset + j]
                predictions.append({"image_path": row["image_path"], "true_index": label,
                                    "pred_index": prediction,
                                    **{f"prob_{i}": probability[i] for i in range(5)}})
                true.append(label)
                pred.append(prediction)
            offset += len(labels)
    metrics = classification_metrics(true, pred, EXPECTED_LABELS)
    metrics.update({"scope": "internal test of newly trained ImageNet-initialized model; same Kaggle source, no external generalization claim",
                    "split": "internal_test", "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "duration_seconds": round(time.perf_counter() - started, 2), "checkpoint_sha256": hash_file(args.checkpoint),
                    "manifest_sha256": hash_file(args.manifest), "best_epoch": checkpoint["best_epoch"],
                    "selection_val_macro_f1": checkpoint["val_macro_f1"], "class_names": EXPECTED_LABELS})
    args.output.mkdir(parents=True)
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
