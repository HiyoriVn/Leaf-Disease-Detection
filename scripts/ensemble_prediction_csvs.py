"""Average probability CSVs from compatible models/TTA passes and score the result."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_model_predictions import LABELS, score, write_csv
from scripts.evaluate_cassava_validation import save_confusion_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+",
                        help="Nonnegative probability-mean weights; defaults to equal weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="probability_mean_ensemble")
    args = parser.parse_args()
    if len(args.predictions) < 2:
        parser.error("provide at least two prediction CSVs")
    weights = args.weights or [1.0 / len(args.predictions)] * len(args.predictions)
    if len(weights) != len(args.predictions) or any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-8:
        parser.error("weights must match the number of CSVs, be nonnegative, and sum to 1")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    rows_by_model = []
    for path in args.predictions:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows_by_model.append(list(csv.DictReader(stream)))
    ref = rows_by_model[0]
    if len(ref) != 3210:
        raise ValueError("expected 3210 validation rows")
    ref_keys = [(r["image_path"], r["true_index"]) for r in ref]
    for rows in rows_by_model[1:]:
        if [(r["image_path"], r["true_index"]) for r in rows] != ref_keys:
            raise ValueError("prediction inputs do not have the same image/label order")
    for rows in rows_by_model:
        for row in rows:
            if (int(row["true_index"]) not in range(5) or row["true_label"] != LABELS[int(row["true_index"])]
                    or int(row["pred_index"]) not in range(5)
                    or row["pred_label"] != LABELS[int(row["pred_index"])]):
                raise ValueError(f"class mapping mismatch for {row['image_path']}")
            probs = [float(row[f"prob_{k}"]) for k in range(5)]
            if abs(sum(probs) - 1.0) > 2e-5:
                raise ValueError(f"invalid probability vector for {row['image_path']}")
            if int(row["pred_index"]) != max(range(5), key=probs.__getitem__):
                raise ValueError(f"predicted index disagrees with probabilities for {row['image_path']}")
    combined = []
    for i, reference in enumerate(ref):
        probs = [sum(weight * float(rows[i][f"prob_{k}"])
                     for weight, rows in zip(weights, rows_by_model)) for k in range(5)]
        pred = max(range(5), key=probs.__getitem__)
        combined.append({"image_path": reference["image_path"], "true_index": int(reference["true_index"]),
                         "true_label": LABELS[int(reference["true_index"])], "pred_index": pred,
                         "pred_label": LABELS[pred], "confidence": probs[pred],
                         **{f"prob_{k}": probs[k] for k in range(5)}})
    metrics = score(combined)
    correct = sum(int(r["true_index"]) == int(r["pred_index"]) for r in combined)
    metrics.update({"model_id": args.name, "n": len(combined), "accuracy": correct / len(combined),
                    "split": "val", "scope": "fixed Cassava validation; not independent test",
                    "probability_averaging_inputs": [str(p) for p in args.predictions]})
    args.output.mkdir(parents=True)
    write_csv(args.output / "predictions.csv", combined)
    write_csv(args.output / "misclassified.csv",
              [r for r in combined if r["true_index"] != r["pred_index"]], list(combined[0]))
    write_csv(args.output / "classification_report.csv", metrics["per_class"])
    cm_rows = [{"true_class": LABELS[i], **{LABELS[j]: metrics["confusion_matrix"][i][j]
                                             for j in range(5)}} for i in range(5)]
    write_csv(args.output / "confusion_matrix.csv", cm_rows)
    save_confusion_png(metrics["confusion_matrix"], args.output / "confusion_matrix.png",
                       f"{args.name} confusion matrix")
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "run_info.json").write_text(json.dumps({"method": "weighted arithmetic mean of class probabilities",
        "weights": weights,
        "inputs": [str(p) for p in args.predictions], "n": len(combined),
        "scope": "fixed validation split; not independent test"}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
