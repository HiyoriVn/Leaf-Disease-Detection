"""Evaluate one Cassava checkpoint on the frozen validation split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw
import timm
import torch
from torch.utils.data import DataLoader
import yaml

from scripts.train_cassava import CassavaDataset, read_manifest
from src.cassava_inference import EXPECTED_LABELS
from src.classification_metrics import classification_metrics

EXPECTED_CHECKPOINTS = {
    "baseline": "7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b",
    "v2": "7398eb98fe0917d330baea90c5c34c6160e2e781aab31dc7f9c73c9c653e2901",
    "v3": "0e76a24e1d979ee29f508b83fe6059f89f7a13a167effcaddc3eacfea0cf669c",
}
EXPECTED_MANIFEST_SHA256 = "570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082"
EXPECTED_PREPROCESSING = {"resize": 343, "crop": 300, "interpolation": "INTER_CUBIC",
                         "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_png(matrix: list[list[int]], path: Path,
                       title: str = "Cassava validation confusion matrix") -> None:
    labels = ["CBB", "CBSD", "Green Mottle", "CMD", "Healthy"]
    cell, left, top = 104, 156, 100
    image = Image.new("RGB", (left + 5 * cell + 28, top + 5 * cell + 88), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), title, fill="black")
    draw.text((left + 2 * cell, 62), "Predicted class", fill="black")
    draw.text((16, top + 2 * cell), "True class", fill="black")
    maximum = max(max(row) for row in matrix) or 1
    for i, label in enumerate(labels):
        draw.text((left + i * cell + 8, top - 22), label, fill="black")
        draw.text((4, top + i * cell + 38), label, fill="black")
        for j, value in enumerate(matrix[i]):
            shade = int(248 - 150 * value / maximum)
            x, y = left + j * cell, top + i * cell
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=(shade, shade + 4, 255), outline="white")
            draw.text((x + 45, y + 43), str(value), fill="black")
    image.save(path)


def predict_probabilities(model, images: torch.Tensor, tta: str) -> torch.Tensor:
    transforms = [(0, False)]
    if tta == "hflip":
        transforms = [(0, False), (0, True)]
    elif tta == "dihedral":
        transforms = [(rotation, flip) for rotation in range(4) for flip in (False, True)]
    probabilities = []
    for rotation, flip in transforms:
        batch = torch.rot90(images, rotation, dims=(-2, -1)) if rotation else images
        if flip:
            batch = torch.flip(batch, dims=(-1,))
        probabilities.append(torch.softmax(model(batch).float(), dim=1))
    return torch.stack(probabilities).mean(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "cassava.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", choices=("baseline", "v2", "v3", "custom"), default="baseline")
    parser.add_argument("--expected-checkpoint-sha256", help="Required for --model custom")
    parser.add_argument("--tta", choices=("none", "hflip", "dihedral"), default="none")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}; choose a new output directory")
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "evaluation.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("b3_validation")
    try:
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("--require-cuda set, but torch.cuda.is_available() is False")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
        else:
            torch.set_num_threads(4)

        checkpoint_sha = sha256(args.checkpoint)
        expected_checkpoint_sha = (args.expected_checkpoint_sha256 if args.model == "custom"
                                   else EXPECTED_CHECKPOINTS[args.model])
        if args.model == "custom" and not expected_checkpoint_sha:
            raise ValueError("--expected-checkpoint-sha256 is required for --model custom")
        if checkpoint_sha != expected_checkpoint_sha:
            raise ValueError(f"checkpoint SHA-256 mismatch: {checkpoint_sha}")
        manifest_sha = sha256(args.manifest)
        if manifest_sha != EXPECTED_MANIFEST_SHA256:
            raise ValueError(f"Cassava manifest SHA-256 mismatch: {manifest_sha}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint.get("architecture"), str):
            raise ValueError("checkpoint architecture missing")
        if tuple(checkpoint.get("class_names", ())) != EXPECTED_LABELS:
            raise ValueError("checkpoint label order mismatch")
        preprocessing = checkpoint.get("preprocessing", {})
        if args.model != "custom":
            for key, value in EXPECTED_PREPROCESSING.items():
                if preprocessing.get(key) != value:
                    raise ValueError(f"checkpoint preprocessing mismatch for {key}")
        else:
            for key in ("resize", "crop", "interpolation", "mean", "std"):
                if key not in preprocessing:
                    raise ValueError(f"custom checkpoint preprocessing missing {key}")
            if (not isinstance(preprocessing["resize"], int) or
                    not isinstance(preprocessing["crop"], int) or
                    preprocessing["resize"] < preprocessing["crop"] or
                    preprocessing["crop"] <= 0 or
                    preprocessing["interpolation"] not in ("INTER_AREA", "INTER_CUBIC")):
                raise ValueError("invalid custom checkpoint preprocessing")
        resize, crop = int(preprocessing.get("resize", 343)), int(preprocessing.get("crop", 300))
        interpolation_name = preprocessing.get("interpolation", "INTER_CUBIC")
        interpolation = getattr(cv2, interpolation_name)
        resize_mode = preprocessing.get("resize_mode", "square")
        if resize_mode not in ("square", "shortest_edge"):
            raise ValueError(f"unsupported checkpoint resize mode: {resize_mode}")

        rows, counts = read_manifest(args.manifest, args.data_root)
        val_rows = [r for r in rows if r["split"] == "val"]
        if len(val_rows) != 3210 or counts != {"train": 14978, "val": 3210, "internal_test": 3209}:
            raise ValueError(f"frozen split counts mismatch: {counts}")
        model = timm.create_model(checkpoint["architecture"], pretrained=False, num_classes=5)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        loader = DataLoader(CassavaDataset(val_rows, args.data_root, False, 0, resize=resize, crop=crop,
                                           interpolation=interpolation, resize_mode=resize_mode),
                            batch_size=args.batch_size, shuffle=False, num_workers=2,
                            pin_memory=device.type == "cuda", persistent_workers=True)
        log.info("Starting validation: model=%s n=%d device=%s batch=%d amp=%s tta=%s",
                 args.model, len(val_rows), device, args.batch_size, device.type == "cuda", args.tta)
        predictions, truth, predicted = [], [], []
        total_loss, offset = 0.0, 0
        started = time.perf_counter()
        batches = len(loader)
        with torch.inference_mode():
            for batch_index, (images, labels) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                labels_device = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    probs = predict_probabilities(model, images, args.tta)
                if tuple(probs.shape) != (len(labels), 5):
                    raise ValueError(f"unexpected output shape: {tuple(probs.shape)}")
                total_loss += float(-torch.log(probs.clamp_min(1e-12)).gather(1, labels_device[:, None]).sum())
                probs_batch = probs.cpu().tolist()
                for j, probs in enumerate(probs_batch):
                    true_index = int(labels[j])
                    pred_index = int(np.argmax(probs))
                    row = val_rows[offset + j]
                    predictions.append({"image_path": row["image_path"], "true_index": true_index,
                                        "true_label": EXPECTED_LABELS[true_index], "pred_index": pred_index,
                                        "pred_label": EXPECTED_LABELS[pred_index], "confidence": probs[pred_index],
                                        **{f"prob_{k}": probs[k] for k in range(5)}})
                    truth.append(true_index)
                    predicted.append(pred_index)
                offset += len(labels)
                if batch_index == 1 or batch_index % 10 == 0 or batch_index == batches:
                    log.info("Progress %d/%d batches (%d/%d images)", batch_index, batches, offset, len(val_rows))
        if offset != 3210 or len(predictions) != 3210:
            raise RuntimeError(f"incomplete evaluation: got {offset} predictions")
        elapsed = time.perf_counter() - started
        metrics = classification_metrics(truth, predicted, EXPECTED_LABELS)
        metrics.update({"model_id": args.model, "tta": args.tta,
                        "split": "val", "scope": "fixed Cassava validation split; not an independent test",
                        "cross_entropy_loss": total_loss / len(truth), "duration_seconds": elapsed,
                        "checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
                        "architecture": checkpoint["architecture"], "input_size": crop,
                        "selection_epoch": checkpoint["best_epoch"],
                        "checkpoint_val_macro_f1_metadata": checkpoint["val_macro_f1"]})
        write_csv(args.output / "predictions.csv", predictions)
        write_csv(args.output / "misclassified.csv", [p for p in predictions if p["true_index"] != p["pred_index"]],
                  list(predictions[0]))
        write_csv(args.output / "classification_report.csv", metrics["per_class"])
        confusion_rows = [{"true_class": EXPECTED_LABELS[i], **{EXPECTED_LABELS[j]: metrics["confusion_matrix"][i][j]
                           for j in range(5)}} for i in range(5)]
        write_csv(args.output / "confusion_matrix.csv", confusion_rows)
        save_confusion_png(metrics["confusion_matrix"], args.output / "confusion_matrix.png",
                           f"{checkpoint['architecture']} validation confusion matrix ({args.tta})")
        (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        info = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__, "timm": timm.__version__, "opencv": cv2.__version__,
                "numpy": np.__version__, "batch_size": args.batch_size, "amp": device.type == "cuda",
                "model_id": args.model, "tta": args.tta,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
                "manifest": str(args.manifest), "manifest_sha256": manifest_sha,
                "data_root": str(args.data_root), "n_val": len(val_rows), "split_counts": counts,
                "preprocessing": {**preprocessing, "resize_mode": resize_mode},
                "duration_seconds": elapsed}
        (args.output / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Finished: accuracy=%.8f macro_f1=%.8f balanced_accuracy=%.8f loss=%.8f duration=%.1fs",
                 metrics["accuracy"], metrics["macro_f1"], metrics["balanced_accuracy"],
                 metrics["cross_entropy_loss"], elapsed)
    except Exception:
        log.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()
