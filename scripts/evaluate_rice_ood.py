"""P2: evaluate a calibrated Rice gate on untouched held-out image folders.

The rice/non-rice folders MUST be disjoint from the training/calibration set.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.rice_quality_gate import check_rice_image


def collect(path, label, gate_path):
    rows = []
    for file in sorted(path.rglob("*")):
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            with Image.open(file) as image:
                result = check_rice_image(image, gate_path=gate_path)
            rows.append((label, result.status, result.score))
        except OSError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rice-dir", required=True, type=Path)
    ap.add_argument("--other-dir", required=True, type=Path)
    ap.add_argument("--gate", type=Path, default=Path("models/rice_gate.json"))
    args = ap.parse_args()
    rows = collect(args.rice_dir, 1, args.gate) + collect(args.other_dir, 0, args.gate)
    if not rows:
        raise SystemExit("No images")
    pos = [r for r in rows if r[0] == 1]
    neg = [r for r in rows if r[0] == 0]
    if not pos or not neg:
        raise SystemExit("Both rice and other images required")
    out = {
        "rice_count": len(pos), "ood_count": len(neg),
        "rice_accept_rate": sum(r[1] == "accept" for r in pos) / len(pos),
        "rice_reject_rate": sum(r[1] == "reject" for r in pos) / len(pos),
        "ood_false_accept_rate": sum(r[1] == "accept" for r in neg) / len(neg),
        "ood_reject_rate": sum(r[1] == "reject" for r in neg) / len(neg),
        "unverified_rate": sum(r[1] == "unverified" for r in rows) / len(rows),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out["unverified_rate"] > 0:
        raise SystemExit("Gate invalid or missing; do not report as calibrated.")


if __name__ == "__main__":
    main()
