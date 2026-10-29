"""Join fixed CropNet smoke predictions with both PyTorch models on same images."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cassava_inference import decode_image, load_cassava_model, load_config, predict_image

DATA_ROOT = Path(r"C:\Users\Admin\Downloads\data\extracted")
INPUT = ROOT / "reports" / "cropnet_smoke" / "predictions.csv"
OUTPUT = ROOT / "reports" / "cropnet_smoke" / "comparison.csv"


def main() -> None:
    config = load_config(ROOT / "configs" / "cassava_finetuned_app.json")
    model = load_cassava_model(config)
    with INPUT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    results = []
    for row in rows:
        content = (DATA_ROOT / row["image_path"]).read_bytes()
        prediction = predict_image(decode_image(content, config), model, config)
        results.append({
            "image_path": row["image_path"],
            "local_label": row["local_label"],
            "published_mobilenetv3": row["published_pred_label"],
            "finetuned_mobilenetv3": prediction["predicted_label"],
            "cropnet": row["cropnet_pred_label"],
            "cropnet_unknown_probability": row["prob_5_unknown"],
        })
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} same-image predictions to {OUTPUT}")


if __name__ == "__main__":
    main()
