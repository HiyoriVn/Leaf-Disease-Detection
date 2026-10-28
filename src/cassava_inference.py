"""Verified cassava state_dict adapter, separate from the TorchScript contract."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cassava_app.json"
EXPECTED_LABELS = (
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
)


class CassavaInferenceError(RuntimeError):
    """Bad image, invalid model contract, or inference failure."""


@dataclass(frozen=True)
class CassavaConfig:
    checkpoint: Path
    checkpoint_sha256: str
    architecture: str
    class_names: tuple[str, ...]
    resize: int
    crop: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    max_upload_bytes: int
    max_pixels: int


def load_config(path: Path = CONFIG_PATH) -> CassavaConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        prep = data["preprocessing"]
        labels = tuple(data["class_names"])
        if data["architecture"] != "mobilenetv3_small_100" or data["num_classes"] != 5 or len(labels) != 5:
            raise ValueError("expected MobileNetV3-Small with five classes")
        if labels != EXPECTED_LABELS or prep["interpolation"] != "INTER_AREA":
            raise ValueError("invalid checkpoint label order or interpolation")
        if prep["resize"] != 256 or prep["crop"] != 224:
            raise ValueError("unexpected preprocessing dimensions")
        mean, std = tuple(map(float, prep["mean"])), tuple(map(float, prep["std"]))
        if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
            raise ValueError("mean/std must have three channels and positive std")
        sha = data["checkpoint_sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid checkpoint SHA-256")
        relative_checkpoint = Path(data["checkpoint"])
        if relative_checkpoint.is_absolute() or ".." in relative_checkpoint.parts:
            raise ValueError("checkpoint must be repository-relative")
        if int(data["max_upload_bytes"]) <= 0 or int(data["max_pixels"]) <= 0:
            raise ValueError("image limits must be positive")
        return CassavaConfig(
            checkpoint=(ROOT / relative_checkpoint).resolve(), checkpoint_sha256=sha,
            architecture=data["architecture"], class_names=labels,
            resize=prep["resize"], crop=prep["crop"], mean=mean, std=std,
            max_upload_bytes=int(data["max_upload_bytes"]), max_pixels=int(data["max_pixels"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CassavaInferenceError(f"Không đọc được cấu hình cassava: {exc}") from exc


def decode_image(content: bytes, config: CassavaConfig) -> Image.Image:
    if not content:
        raise CassavaInferenceError("Chưa có ảnh để phân tích.")
    if len(content) > config.max_upload_bytes:
        raise CassavaInferenceError("Ảnh vượt giới hạn 20 MB. Vui lòng chọn ảnh nhỏ hơn.")
    try:
        with Image.open(BytesIO(content)) as image:
            if image.width * image.height > config.max_pixels:
                raise CassavaInferenceError("Ảnh có độ phân giải quá lớn. Giới hạn là 24 triệu pixel.")
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise CassavaInferenceError("Không đọc được ảnh. Hãy dùng JPEG, PNG hoặc WebP hợp lệ.") from exc


def preprocess_image(image: Image.Image, config: CassavaConfig) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized = cv2.resize(rgb, (config.resize, config.resize), interpolation=cv2.INTER_AREA)
    offset = (config.resize - config.crop) // 2
    cropped = resized[offset:offset + config.crop, offset:offset + config.crop, :] / 255.0
    normalized = (cropped - np.asarray(config.mean, dtype=np.float32)) / np.asarray(config.std, dtype=np.float32)
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cassava_model(config: CassavaConfig) -> Any:
    if not config.checkpoint.is_file():
        raise CassavaInferenceError(f"Không tìm thấy checkpoint: {config.checkpoint}")
    if _sha256(config.checkpoint) != config.checkpoint_sha256:
        raise CassavaInferenceError("SHA-256 của checkpoint không khớp bản đã kiểm chứng.")
    try:
        model = timm.create_model(config.architecture, pretrained=False, num_classes=len(config.class_names))
        state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        weights = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        model.load_state_dict(weights, strict=True)
        model.eval()
        return model
    except Exception as exc:
        raise CassavaInferenceError(f"Không nạp được checkpoint cassava: {exc}") from exc


def predict_image(image: Image.Image, model: Any, config: CassavaConfig) -> dict:
    tensor = preprocess_image(image, config)
    try:
        with torch.inference_mode():
            logits = model(tensor)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, len(config.class_names)):
            raise CassavaInferenceError(f"Đầu ra model không hợp lệ: {getattr(logits, 'shape', None)}")
        if not torch.isfinite(logits).all():
            raise CassavaInferenceError("Model trả về giá trị không hữu hạn.")
        probabilities = torch.softmax(logits[0], dim=0)
        order = torch.argsort(probabilities, descending=True).tolist()
        top3 = [{"index": i, "label": config.class_names[i], "probability": float(probabilities[i])} for i in order[:3]]
        return {"predicted_index": order[0], "predicted_label": config.class_names[order[0]],
                "confidence": float(probabilities[order[0]]),
                "probabilities": [float(p) for p in probabilities], "top3": top3,
                "input_shape": list(tensor.shape), "output_shape": list(logits.shape)}
    except CassavaInferenceError:
        raise
    except Exception as exc:
        raise CassavaInferenceError(f"Không thể phân tích ảnh: {exc}") from exc
