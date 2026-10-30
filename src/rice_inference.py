"""Independent LiSA/TFLite inference adapter for RGB rice-leaf images.

The resize/pixel-scale settings are *provisional* until compared to the
training notebook and Keras model. Do not treat predictions as validated.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rice_app.json"
EXPECTED_CLASSES = (
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "blast", "brown_spot", "dead_heart", "downy_mildew", "hispa",
    "normal", "tungro",
)


class RiceInferenceError(RuntimeError):
    """Invalid contract, image, or TFLite inference."""


@dataclass(frozen=True)
class RiceConfig:
    model_path: Path
    labels: tuple[str, ...]
    image_size: tuple[int, int]
    pixel_scale: str
    preprocessing_verified: bool
    max_pixels: int
    expected_model_bytes: int


def load_rice_config(path: Path = CONFIG_PATH) -> RiceConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        def safe_relative(value: str) -> Path:
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("Expected repository-relative path")
            return ROOT / candidate
        model_path = safe_relative(raw["model"])
        label_path = safe_relative(raw["labels"])
        labels = tuple(json.loads(label_path.read_text(encoding="utf-8")))
        if labels != EXPECTED_CLASSES:
            raise ValueError("Rice label order does not match training artifact")
        size = tuple(raw["image_size"])
        if size != (224, 224):
            raise ValueError("Expected 224x224 LiSA inputs")
        scale = raw["pixel_scale"]
        if scale not in ("0_255", "0_1"):
            raise ValueError("pixel_scale must be 0_255 or 0_1")
        max_pixels = int(raw["max_pixels"])
        expected_bytes = int(raw["expected_model_bytes"])
        if max_pixels <= 0 or expected_bytes <= 0:
            raise ValueError("Invalid image/model size limits")
        return RiceConfig(model_path, labels, size, scale,
                          bool(raw["preprocessing_verified"]),
                          max_pixels, expected_bytes)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RiceInferenceError(f"Không đọc được cấu hình LiSA: {exc}") from exc


def preprocess_rice(image: Image.Image, config: RiceConfig) -> np.ndarray:
    """RGB arbitrary-size image to NHWC float32, using explicit scale config."""
    if image.width * image.height > config.max_pixels:
        raise RiceInferenceError("Ảnh có quá nhiều pixel; hãy chọn ảnh nhỏ hơn")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize(config.image_size, resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    if config.pixel_scale == "0_1":
        pixels /= 255.0
    return pixels[None, ...]


def load_rice_interpreter(config: RiceConfig):
    """Load a fresh interpreter per request (no shared mutable tensor state)."""
    if not config.model_path.is_file():
        raise RiceInferenceError(f"Không tìm thấy model: {config.model_path}")
    if config.model_path.stat().st_size != config.expected_model_bytes:
        raise RiceInferenceError("Dung lượng model khác artifact trên nhánh hehe")
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise RiceInferenceError(
            "Thiếu ai-edge-litert. Cài bằng: pip install -r requirements-rice.txt"
        ) from exc
    try:
        interpreter = Interpreter(model_path=str(config.model_path))
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details(), interpreter.get_output_details()
        if len(inp) != 1 or len(out) != 1:
            raise ValueError("Expected one input and one output tensor")
        if tuple(inp[0]["shape"]) != (1, 224, 224, 3) or inp[0]["dtype"] != np.float32:
            raise ValueError(f"Unexpected model input shape/dtype: {inp[0]}")
        if int(np.prod(out[0]["shape"])) != len(config.labels) or out[0]["dtype"] != np.float32:
            raise ValueError(f"Unexpected model output shape/dtype: {out[0]}")
        return interpreter
    except Exception as exc:
        raise RiceInferenceError(f"Không tải được model LiSA: {exc}") from exc


def predict_rice(image: Image.Image, interpreter, config: RiceConfig) -> dict:
    tensor = preprocess_rice(image, config)
    try:
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()
        scores = np.asarray(interpreter.get_tensor(out["index"]), dtype=np.float64).reshape(-1)
        if len(scores) != len(config.labels) or not np.all(np.isfinite(scores)):
            raise ValueError("Invalid scores from TFLite")
        # Avoid applying softmax twice when model already returns probabilities.
        if np.all(scores >= 0) and np.all(scores <= 1) and np.isclose(scores.sum(), 1, atol=1e-4):
            probabilities = scores / scores.sum()
        else:
            exp = np.exp(scores - scores.max())
            probabilities = exp / exp.sum()
        # Optional temperature learned on a validation split, never on TEST.
        temperature_path = ROOT / "models" / "rice_temperature.json"
        temperature_applied = False
        if temperature_path.is_file():
            try:
                calibration = json.loads(temperature_path.read_text(encoding="utf-8"))
                if calibration.get("schema_version") != 1:
                    raise ValueError("Unsupported calibration schema")
                temperature = float(calibration["temperature"])
                if not np.isfinite(temperature) or temperature <= 0:
                    raise ValueError("Invalid temperature")
                logp = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
                exps = np.exp(logp - logp.max())
                probabilities = exps / exps.sum()
                temperature_applied = True
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RiceInferenceError(f"Cấu hình hiệu chỉnh LiSA không hợp lệ: {exc}") from exc
        order = np.argsort(-probabilities)
        top3 = [{"label": config.labels[int(i)], "probability": float(probabilities[i])}
                for i in order[:3]]
        return {"predicted_label": top3[0]["label"],
                "confidence": top3[0]["probability"],
                "top3": top3, "probabilities": probabilities.tolist(),
                "preprocessing_verified": config.preprocessing_verified,
                "temperature_calibrated": temperature_applied}
    except RiceInferenceError:
        raise
    except Exception as exc:
        raise RiceInferenceError(f"Lỗi suy luận LiSA: {exc}") from exc
