"""P2: fit softmax temperature on VALIDATION predictions, evaluate on TEST.

The repository currently includes test_predictions.csv but not val_predictions.csv.
Do not fit temperature on the TEST split (data leakage). This is independent
of OOD rejection and does not turn softmax confidence into diagnosis certainty.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_csv(path, labels):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No prediction rows")
    p = np.array([[float(r["p_" + label]) for label in labels] for r in rows])
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([int(r["label_id"]) for r in rows])
    if np.any(y < 0) or np.any(y >= len(labels)):
        raise ValueError("Unexpected label IDs")
    return p, y


def temperature_scale(p, t):
    q = np.log(np.clip(p, 1e-12, 1.0)) / t
    q -= q.max(axis=1, keepdims=True)
    q = np.exp(q)
    return q / q.sum(axis=1, keepdims=True)


def ece(p, y, bins=15):
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins+1)[:-1], np.linspace(0, 1, bins+1)[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def metrics(p, y):
    correct = (p.argmax(axis=1) == y)
    one_hot = np.eye(p.shape[1])[y]
    out = {
        "accuracy": float(correct.mean()), "ece_15bins": ece(p, y),
        "mean_nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "multiclass_brier": float(np.sum((p - one_hot) ** 2, axis=1).mean()),
    }
    out["risk_coverage"] = [
        {"threshold": float(thr), "coverage": float((p.max(axis=1) >= thr).mean()),
         "error_rate_on_selected": (float((~correct[p.max(axis=1) >= thr]).mean())
                                    if (p.max(axis=1) >= thr).any() else None)}
        for thr in (.5, .7, .9, .95, .99)
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--test-csv", type=Path, required=True)
    ap.add_argument("--class-json", type=Path, default=Path(
        "Rice-leaf/riceLeafDisease_trainingResult/class_names.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/rice_calibration_local.json"))
    args = ap.parse_args()
    if args.val_csv.resolve() == args.test_csv.resolve():
        raise SystemExit("Refusing to calibrate using the test CSV.")
    labels = json.loads(args.class_json.read_text(encoding="utf-8"))
    pv, yv = read_csv(args.val_csv, labels)
    pt, yt = read_csv(args.test_csv, labels)
    grid = np.exp(np.linspace(np.log(.3), np.log(8.), 250))
    losses = [metrics(temperature_scale(pv, t), yv)["mean_nll"] for t in grid]
    t = float(grid[int(np.argmin(losses))])
    out = {
        "temperature_from_validation_only": t,
        "validation_before": metrics(pv, yv),
        "validation_after": metrics(temperature_scale(pv, t), yv),
        "test_before": metrics(pt, yt),
        "test_after": metrics(temperature_scale(pt, t), yt),
        "warning": "Temperature scaling alone cannot detect arbitrary non-rice images."
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
