"""Inference helpers for the coffee leaf disease ResNet18 checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class CoffeeInferenceError(RuntimeError):
    """Raised when the coffee checkpoint cannot be loaded or used."""


def load_coffee_model(model_path: Path) -> tuple[nn.Module, list[str]]:
    """Load and validate a CPU checkpoint; cache this result in the app."""
    if not model_path.is_file():
        raise CoffeeInferenceError(f"Không tìm thấy model cà phê: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        class_names = checkpoint["class_names"]
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(class_names, list) or not class_names or not all(
            isinstance(name, str) and name for name in class_names
        ):
            raise ValueError("class_names không hợp lệ")

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(state_dict)
        model.eval()
        return model, class_names
    except Exception as exc:
        raise CoffeeInferenceError(f"Không thể tải model cà phê: {exc}") from exc


_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict_coffee(
    image: Image.Image, model: nn.Module, class_names: list[str]
) -> tuple[str, float, list[dict[str, float | str]]]:
    """Return predicted class, confidence percent, and top-three probabilities."""
    if not class_names:
        raise CoffeeInferenceError("Model cà phê không có danh sách lớp")
    try:
        input_tensor = _PREPROCESS(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]
        if probabilities.numel() != len(class_names):
            raise ValueError("Số đầu ra của model không khớp số lớp")

        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
        top3 = [
            {"label": class_names[index.item()], "probability": probability.item()}
            for probability, index in zip(top_probs, top_indices)
        ]
        return top3[0]["label"], float(top3[0]["probability"]) * 100, top3
    except Exception as exc:
        raise CoffeeInferenceError(f"Không thể dự đoán ảnh cà phê: {exc}") from exc
