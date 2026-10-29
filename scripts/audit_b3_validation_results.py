"""Verify a returned EfficientNet-B3 validation ZIP without touching model weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

EXPECTED_CHECKPOINT = "7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b"
EXPECTED_MANIFEST = "570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082"
LABELS = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]
REQUIRED = {
    "metrics.json", "classification_report.csv", "confusion_matrix.csv",
    "predictions.csv", "misclassified.csv", "evaluation.log", "run_info.json",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol


def audit(archive: Path, checkpoint: Path | None = None, manifest: Path | None = None) -> dict:
    with zipfile.ZipFile(archive) as zf:
        names = {Path(n).name for n in zf.namelist() if not n.endswith("/")}
        missing = REQUIRED - names
        if missing:
            raise ValueError(f"archive missing files: {sorted(missing)}")

        def read(name: str) -> str:
            matches = [n for n in zf.namelist() if Path(n).name == name]
            if len(matches) != 1:
                raise ValueError(f"expected one {name}, found {len(matches)}")
            return zf.read(matches[0]).decode("utf-8-sig")

        metrics = json.loads(read("metrics.json"))
        info = json.loads(read("run_info.json"))
        log = read("evaluation.log")
        predictions = list(csv.DictReader(read("predictions.csv").splitlines()))
        misclassified = list(csv.DictReader(read("misclassified.csv").splitlines()))
        report = list(csv.DictReader(read("classification_report.csv").splitlines()))
        matrix_rows = list(csv.DictReader(read("confusion_matrix.csv").splitlines()))

    if info.get("n_val") != 3210 or metrics.get("n") != 3210 or len(predictions) != 3210:
        raise ValueError("validation image count is not 3210 in every artifact")
    if len(misclassified) != 542:
        raise ValueError(f"misclassified.csv has {len(misclassified)} rows, expected 542")
    if info.get("checkpoint_sha256") != EXPECTED_CHECKPOINT or metrics.get("checkpoint_sha256") != EXPECTED_CHECKPOINT:
        raise ValueError("checkpoint SHA-256 does not match the fixed B3 checkpoint")
    if info.get("manifest_sha256") != EXPECTED_MANIFEST or metrics.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise ValueError("manifest SHA-256 does not match the frozen Cassava manifest")
    if checkpoint and digest(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("local checkpoint SHA-256 mismatch")
    if manifest and digest(manifest) != EXPECTED_MANIFEST:
        raise ValueError("local manifest SHA-256 mismatch")
    if info.get("device") != "cuda" or "RTX 3060" not in str(info.get("gpu")):
        raise ValueError("run_info does not identify the expected CUDA RTX 3060 run")
    if info.get("split_counts") != {"train": 14978, "val": 3210, "internal_test": 3209}:
        raise ValueError("split counts do not match the frozen manifest")

    cm = [[0] * 5 for _ in range(5)]
    wrong = []
    for row in predictions:
        t, p = int(row["true_index"]), int(row["pred_index"])
        if not (0 <= t < 5 and 0 <= p < 5):
            raise ValueError("prediction contains class index outside [0,4]")
        probs = [float(row[f"prob_{i}"]) for i in range(5)]
        if not close(sum(probs), 1.0, 2e-5) or p != max(range(5), key=probs.__getitem__):
            raise ValueError(f"invalid probability vector or argmax for {row['image_path']}")
        cm[t][p] += 1
        if t != p:
            wrong.append(row)
    if wrong != misclassified:
        raise ValueError("misclassified.csv does not exactly match predictions.csv")
    if cm != metrics.get("confusion_matrix"):
        raise ValueError("metrics.json confusion matrix differs from predictions.csv")
    if [r.get("true_class") for r in matrix_rows] != LABELS:
        raise ValueError("confusion_matrix.csv row label order mismatch")
    csv_cm = [[int(row[label]) for label in LABELS] for row in matrix_rows]
    if csv_cm != cm:
        raise ValueError("confusion_matrix.csv differs from predictions.csv")
    if [row.get("class_name") for row in report] != LABELS:
        raise ValueError("classification report label order mismatch")

    per_class = []
    for i, label in enumerate(LABELS):
        support = sum(cm[i])
        predicted = sum(cm[j][i] for j in range(5))
        tp = cm[i][i]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = metrics["per_class"][i]
        if row["class_name"] != label or row["support"] != support:
            raise ValueError(f"per-class support mismatch for {label}")
        if not all(close(float(row[k]), v) for k, v in (("precision", precision), ("recall", recall), ("f1", f1))):
            raise ValueError(f"per-class metric mismatch for {label}")
        if not all(close(float(report[i][k]), v) for k, v in (("precision", precision), ("recall", recall), ("f1", f1))):
            raise ValueError(f"classification_report.csv metric mismatch for {label}")
        per_class.append({"class_name": label, "support": support, "precision": precision,
                          "recall": recall, "f1": f1})
    correct = sum(cm[i][i] for i in range(5))
    accuracy = correct / len(predictions)
    macro_f1 = sum(x["f1"] for x in per_class) / 5
    balanced_accuracy = sum(x["recall"] for x in per_class) / 5
    if not close(metrics["accuracy"], accuracy) or not close(metrics["macro_f1"], macro_f1) or not close(metrics["balanced_accuracy"], balanced_accuracy):
        raise ValueError("aggregate metrics do not recompute from predictions")
    if not all(token in log for token in (
        "Progress 201/201 batches (3210/3210 images)",
        f"accuracy={accuracy:.8f}", f"macro_f1={macro_f1:.8f}",
        f"balanced_accuracy={balanced_accuracy:.8f}")):
        raise ValueError("evaluation.log completion/metrics do not agree with artifacts")
    errors = [{"true_label": LABELS[i], "predicted_label": LABELS[j], "count": cm[i][j]}
              for i in range(5) for j in range(5) if i != j and cm[i][j] > 0]
    return {
        "status": "verified",
        "results_archive_sha256": digest(archive),
        "n": len(predictions), "correct": correct, "misclassified": len(wrong),
        "accuracy": accuracy, "macro_f1": macro_f1, "balanced_accuracy": balanced_accuracy,
        "checkpoint_sha256": metrics["checkpoint_sha256"], "manifest_sha256": metrics["manifest_sha256"],
        "selection_epoch": metrics.get("selection_epoch"),
        "checkpoint_val_macro_f1_metadata": metrics.get("checkpoint_val_macro_f1_metadata"),
        "per_class": per_class, "confusion_matrix": cm, "error_pairs": errors,
        "gpu": info["gpu"], "duration_seconds": info["duration_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive, args.checkpoint, args.manifest)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
