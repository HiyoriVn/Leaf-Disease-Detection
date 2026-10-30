"""P1: train an independent RGB image validity gate on rice and non-rice pictures.

The model's purpose is to discriminate rice images vs external input, NOT to
diagnose disease. Train only on representative rice AND non-rice examples.
This small handcrafted feature baseline requires independent evaluation.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from src.rice_quality_gate import rice_gate_features


def images(folder):
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in
             {".jpg", ".jpeg", ".png", ".webp"}]
    if len(paths) < 20:
        raise ValueError(f"Need at least 20 images in {folder}, found {len(paths)}")
    features = []
    for path in sorted(paths):
        try:
            with Image.open(path) as image:
                features.append(rice_gate_features(image))
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            print("Skipped invalid file", path)
    return np.asarray(features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rice-dir", type=Path, required=True)
    ap.add_argument("--other-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("models/rice_gate.json"))
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    positive = images(args.rice_dir)
    negative = images(args.other_dir)
    X = np.concatenate([positive, negative])
    y = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
    clf.fit(scaler.transform(X_train), y_train)
    prob = clf.predict_proba(scaler.transform(X_val))[:, 1]
    # Choose threshold on held-out ID examples; report the induced OOD false accept rate.
    threshold = float(np.quantile(prob[y_val == 1], .05))
    accepted = prob >= threshold
    result = {
        "schema_version": 1, "feature_version": 1,
        "mean": scaler.mean_.tolist(),
        "std": np.where(scaler.scale_ <= 0, 1., scaler.scale_).tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "threshold": float(np.clip(threshold, 1e-6, 1.-1e-6)),
        "validation": {
            "n_rice": int(sum(y_val == 1)), "n_other": int(sum(y_val == 0)),
            "auroc": round(float(roc_auc_score(y_val, prob)), 4),
            "rice_false_reject_rate": round(float(np.mean(~accepted[y_val == 1])), 4),
            "ood_false_accept_rate": round(float(np.mean(accepted[y_val == 0])), 4),
        },
        "notes": "Exploratory handcrafted-feature baseline; not a certified rice detector. "
                 "Use a separate unseen OOD set for final evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validation": result["validation"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
