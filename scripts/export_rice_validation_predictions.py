"""Export PADDY validation predictions for P2 temperature scaling.

Uses the FROZEN val_split.csv from the training artifact. This tool requires
P0 preprocessing verification FIRST, and refuses to apply existing temperature.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageOps
from src.rice_inference import load_rice_config, load_rice_interpreter, predict_rice, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("reports/val_predictions.csv"))
    args = ap.parse_args()

    cfg = load_rice_config()
    if not cfg.preprocessing_verified:
        raise SystemExit("P0 is not verified. Do not export calibration probabilities yet.")
    if (ROOT / "models" / "rice_temperature.json").exists():
        raise SystemExit("Remove or temporarily relocate rice_temperature.json to avoid double calibration.")
    source = ROOT / "Rice-leaf" / "riceLeafDisease_trainingResult" / "val_split.csv"
    with source.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("val_split.csv has no images")
    interpreter = load_rice_interpreter(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = ["image_path", "label_id", "class_name", "predicted_label_id", "predicted_class",
             "confidence"] + ["p_" + label for label in cfg.labels]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for n, row in enumerate(rows, 1):
            original = Path(row["image_path"])
            path = args.dataset_root / original.parent.name / original.name
            if not path.is_file():
                raise FileNotFoundError(f"Missing validation picture: {path}")
            with Image.open(path) as img:
                result = predict_rice(img, interpreter, cfg)
            rank = result["predicted_label"]
            output = {
                "image_path": str(original), "label_id": row["label_id"],
                "class_name": row["class_name"],
                "predicted_label_id": cfg.labels.index(rank),
                "predicted_class": rank,
                "confidence": result["confidence"],
            }
            output.update({"p_" + label: p for label, p in zip(cfg.labels, result["probabilities"])})
            writer.writerow(output)
            if n % 100 == 0:
                print(f"Completed {n}/{len(rows)}")
    print(f"Wrote {len(rows)} validation rows to {args.output}")


if __name__ == "__main__":
    main()
