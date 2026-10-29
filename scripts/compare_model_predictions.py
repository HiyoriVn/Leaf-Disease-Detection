"""Compare model prediction CSVs on the exact same frozen validation images."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
          "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy"]


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3210:
        raise ValueError(f"{path}: expected 3210 predictions, found {len(rows)}")
    return rows


def score(rows: list[dict]) -> dict:
    cm = [[0] * 5 for _ in range(5)]
    for row in rows:
        true_index, pred_index = int(row["true_index"]), int(row["pred_index"])
        if true_index not in range(5) or pred_index not in range(5):
            raise ValueError(f"invalid class index in prediction row: {row.get('image_path')}")
        if row.get("true_label") != LABELS[true_index] or row.get("pred_label") != LABELS[pred_index]:
            raise ValueError(f"class label/index mismatch in prediction row: {row.get('image_path')}")
        cm[true_index][pred_index] += 1
    per_class = []
    for i, name in enumerate(LABELS):
        tp, support = cm[i][i], sum(cm[i])
        predicted = sum(cm[j][i] for j in range(5))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_name": name, "support": support, "precision": precision,
                          "recall": recall, "f1": f1})
    correct = sum(cm[i][i] for i in range(5))
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "macro_f1": sum(x["f1"] for x in per_class) / 5,
            "balanced_accuracy": sum(x["recall"] for x in per_class) / 5,
            "per_class": per_class, "confusion_matrix": cm}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    fields = fieldnames or (list(rows[0]) if rows else None)
    if not fields:
        raise ValueError(f"CSV field names unavailable for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline predictions.csv")
    parser.add_argument("--v2", type=Path, help="V2 predictions.csv")
    parser.add_argument("--v3", type=Path, help="V3 predictions.csv")
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=CSV",
                        help="Additional candidate prediction CSV (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    sources = {"baseline": load_predictions(args.baseline)}
    for name, path in (("v2", args.v2), ("v3", args.v3)):
        if path:
            sources[name] = load_predictions(path)
    for value in args.candidate:
        if "=" not in value:
            parser.error(f"candidate must be NAME=CSV: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in sources:
            parser.error(f"duplicate or empty candidate name: {name}")
        sources[name] = load_predictions(Path(raw_path))
    if len(sources) < 2:
        parser.error("provide at least one candidate using --v2, --v3 or --candidate")
    reference = [(r["image_path"], r["true_index"]) for r in sources["baseline"]]
    for name, rows in sources.items():
        if [(r["image_path"], r["true_index"]) for r in rows] != reference:
            raise ValueError(f"{name} predictions do not match the exact baseline image/label sequence")
    scores = {name: score(rows) for name, rows in sources.items()}
    class_rows = []
    transition_rows = []
    for i, label in enumerate(LABELS):
        base = scores["baseline"]["per_class"][i]
        row = {"class_name": label, "support": base["support"]}
        for name in sources:
            current = scores[name]["per_class"][i]
            row[f"{name}_precision"] = current["precision"]
            row[f"{name}_recall"] = current["recall"]
            row[f"{name}_f1"] = current["f1"]
        for name in sources:
            if name == "baseline":
                continue
            row[f"{name}_delta_f1_vs_baseline"] = row[f"{name}_f1"] - row["baseline_f1"]
            row[f"{name}_delta_recall_vs_baseline"] = row[f"{name}_recall"] - row["baseline_recall"]
        class_rows.append(row)
        for name in sources:
            if name == "baseline":
                continue
            corrected = regressed = 0
            for b, c in zip(sources["baseline"], sources[name]):
                if int(b["true_index"]) != i:
                    continue
                base_ok = int(b["pred_index"]) == i
                candidate_ok = int(c["pred_index"]) == i
                if not base_ok and candidate_ok:
                    corrected += 1
                elif base_ok and not candidate_ok:
                    regressed += 1
            transition_rows.append({"model": name, "true_class": label,
                                    "baseline_wrong_candidate_correct": corrected,
                                    "baseline_correct_candidate_wrong": regressed,
                                    "net_correct_change": corrected - regressed})
    args.output.mkdir(parents=True)
    (args.output / "comparison_summary.json").write_text(
        json.dumps({"scope": "same 3210 fixed validation rows; not independent test",
                    "models": scores}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output / "per_class_comparison.csv", class_rows)
    write_csv(args.output / "paired_error_transitions.csv", transition_rows)
    print(json.dumps({k: {m: round(v[m], 6) for m in ("accuracy", "macro_f1", "balanced_accuracy")}
                      for k, v in scores.items()}, indent=2))


if __name__ == "__main__":
    main()
