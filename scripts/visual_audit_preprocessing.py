"""Create a small original-vs-training-preprocessing visual audit sheet."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def preprocess(image: Image.Image, size: int, crop: int, mode: str, interpolation: int) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if mode == "square":
        resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
    else:
        h, w = rgb.shape[:2]
        scale = size / min(h, w)
        out_w, out_h = max(size, round(w * scale)), max(size, round(h * scale))
        resized = cv2.resize(rgb, (out_w, out_h), interpolation=interpolation)
    h, w = resized.shape[:2]
    top, left = (h - crop) // 2, (w - crop) // 2
    return Image.fromarray(resized[top:top + crop, left:left + crop])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/cassava.csv"))
    parser.add_argument("--baseline-results-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=343)
    parser.add_argument("--crop", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    val_by_pair: dict[tuple[int, int], str] = {}
    with zipfile.ZipFile(args.baseline_results_zip) as archive:
        with archive.open("predictions.csv") as stream:
            for row in csv.DictReader((line.decode("utf-8-sig") for line in stream)):
                truth, pred = int(row["true_index"]), int(row["pred_index"])
                if truth != pred and (truth, pred) not in val_by_pair:
                    val_by_pair[(truth, pred)] = row["image_path"]
    class_names = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    samples: list[tuple[str, str, str]] = []
    for (truth, pred), path in val_by_pair.items():
        if (truth, pred) in ((0, 4), (4, 0), (2, 3)):
            samples.append(("validation error", f"{class_names[truth]} → {class_names[pred]}", path))
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        train_rows = [r for r in csv.DictReader(stream) if r["split"] == "train" and not r["exclusion_reason"]]
    for class_index in (0, 2, 4):
        row = next(r for r in train_rows if int(r["label_index"]) == class_index)
        samples.append(("training example", class_names[class_index], row["image_path"]))
    if len(samples) != 6:
        raise RuntimeError("could not select expected validation errors and training examples")

    panel = 320
    margin = 24
    header = 54
    card_h = panel + header + margin
    canvas = Image.new("RGB", (margin + 2 * (panel + 16) + margin, margin + len(samples) * card_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_i, (split, label, relative) in enumerate(samples):
        path = args.data_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            original = im.convert("RGB")
            transformed = preprocess(original, args.size, args.crop, "square", cv2.INTER_CUBIC)
            original_thumb = original.copy()
            original_thumb.thumbnail((panel, panel), Image.Resampling.LANCZOS)
        y = margin + row_i * card_h
        draw.text((margin, y), f"{split} | {label} | {relative}", fill="black", font=font)
        oy = y + header
        canvas.paste(original_thumb, (margin, oy))
        canvas.paste(transformed.resize((panel, panel), Image.Resampling.LANCZOS), (margin + panel + 16, oy))
        draw.text((margin, oy + panel + 3), "Original (fit, no crop)", fill="black", font=font)
        draw.text((margin + panel + 16, oy + panel + 3), "343×343 resize → center crop 300×300", fill="black", font=font)
    canvas.save(args.output)
    print(f"Wrote {len(samples)} sample pairs to {args.output}")


if __name__ == "__main__":
    main()
