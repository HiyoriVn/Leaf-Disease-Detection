"""Small, fixed, labeled in-source sample for CropNet prediction inspection."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cropnet_inference import CLASS_NAMES, MODEL_DIR, file_sha256, load_model, predict

DATA_ROOT = Path(r"C:\Users\Admin\Downloads\data\extracted")
BASELINE = ROOT / "reports" / "baseline_cassava_full_val" / "predictions.csv"
OUTPUT = ROOT / "reports" / "cropnet_smoke"


def main() -> None:
    selected: list[dict] = []
    counts = {i: 0 for i in range(5)}
    with BASELINE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            index = int(row["true_index"])
            if counts[index] < 5:
                selected.append(row)
                counts[index] += 1
            if all(n == 5 for n in counts.values()):
                break
    if len(selected) != 25:
        raise ValueError(f"Expected 25 fixed validation examples, got {len(selected)}")
    model = load_model()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        image_path = DATA_ROOT / row["image_path"]
        start = time.perf_counter()
        result = predict(image_path.read_bytes(), model)
        latency_ms = (time.perf_counter() - start) * 1000
        record = {
            "image_path": row["image_path"], "local_label": row["true_label"],
            "published_pred_label": row["pred_label"],
            "cropnet_pred_label": result["predicted_label"],
            "cropnet_pred_index": result["predicted_index"],
            "inference_ms_including_preprocessing": round(latency_ms, 3),
        }
        record.update({f"prob_{i}_{name.lower().replace(' ', '_')}": value for i, (name, value) in enumerate(zip(("cbb", "cbsd", "cgm_green_mite", "cmd", "healthy", "unknown"), result["probabilities"]))})
        records.append(record)
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    files = {str(path.relative_to(MODEL_DIR)).replace('\\', '/'): {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in MODEL_DIR.rglob('*') if path.is_file() and '.complete' not in path.parts}
    summary = {
        "sample": "First five rows per local label from existing 3210-row validation predictions CSV; same-source exploratory smoke only",
        "count": len(records), "class_order": CLASS_NAMES,
        "artifact_files": files,
        "cropnet_prediction_counts": {name: sum(r["cropnet_pred_label"] == name for r in records) for name in CLASS_NAMES},
        "published_agreement_count": sum(r["cropnet_pred_label"] == r["published_pred_label"] for r in records),
        "latency_ms_median_including_preprocessing": sorted(r["inference_ms_including_preprocessing"] for r in records)[len(records)//2],
        "input_shape": [1, 224, 224, 3], "output_shape": [1, 6],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
