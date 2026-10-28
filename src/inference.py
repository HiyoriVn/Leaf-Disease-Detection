"""Strict shared loading and inference for TorchScript leaf classifiers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "manifest.json"


class ModelError(RuntimeError):
    """A missing or invalid model artifact or prediction contract."""


@dataclass(frozen=True)
class ModelConfig:
    crop: str
    checkpoint: Path
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    labels: tuple[str, ...]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ModelConfig]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"Cannot read model manifest at {path}: {exc}") from exc
    result: dict[str, ModelConfig] = {}
    for crop in ("rice", "coffee", "cassava"):
        item = raw.get(crop)
        if not isinstance(item, dict):
            raise ModelError(f"Manifest is missing configuration for {crop}.")
        labels = item.get("labels", [])
        size = item.get("image_size", 224)
        mean, std = item.get("mean"), item.get("std")
        if not isinstance(labels, list) or len(set(labels)) != len(labels):
            raise ModelError(f"{crop}: labels must be a list of unique class names.")
        if not isinstance(size, int) or size < 16:
            raise ModelError(f"{crop}: image_size must be an integer >= 16.")
        if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3 or any(float(x) <= 0 for x in std):
            raise ModelError(f"{crop}: mean and std must contain three channels; std must be positive.")
        if item.get("architecture") != "torchscript":
            raise ModelError(f"{crop}: only explicit TorchScript checkpoints are supported by this baseline.")
        checkpoint = (path.parent / item.get("checkpoint", "")).resolve()
        result[crop] = ModelConfig(crop, checkpoint, size, tuple(map(float, mean)), tuple(map(float, std)), tuple(labels))
    return result


def preprocess(image: Image.Image, config: ModelConfig) -> torch.Tensor:
    import torch
    image = ImageOps.exif_transpose(image).convert("RGB").resize((config.image_size, config.image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(config.mean, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(config.std, dtype=tensor.dtype)[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0)


def load_model(config: ModelConfig) -> Any:
    if not config.labels:
        raise ModelError(f"{config.crop}: class labels are not configured in models/manifest.json.")
    if not config.checkpoint.is_file():
        raise ModelError(f"{config.crop}: checkpoint not found: {config.checkpoint}")
    import torch
    try:
        model = torch.jit.load(str(config.checkpoint), map_location="cpu")
        model.eval()
    except Exception as exc:
        raise ModelError(f"{config.crop}: could not load TorchScript checkpoint: {exc}") from exc
    return model


def predict(image: Image.Image, config: ModelConfig, model: Any | None = None, top_k: int = 3) -> list[dict[str, float | str]]:
    import torch
    model = model if model is not None else load_model(config)
    try:
        with torch.inference_mode():
            logits = model(preprocess(image, config))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
            raise ModelError(f"{config.crop}: expected logits shaped [1, number_of_classes].")
        if logits.shape[1] != len(config.labels):
            raise ModelError(f"{config.crop}: model returns {logits.shape[1]} scores but manifest has {len(config.labels)} labels.")
        probabilities = torch.softmax(logits[0], dim=0)
        k = min(max(1, top_k), len(config.labels))
        values, indices = torch.topk(probabilities, k)
        return [{"label": config.labels[int(i)], "confidence": float(v)} for v, i in zip(values, indices)]
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(f"{config.crop}: inference failed: {exc}") from exc
