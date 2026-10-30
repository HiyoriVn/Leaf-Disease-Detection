import hashlib

import pytest
import torch
from PIL import Image

from src.coffee_inference import (
    CoffeeInferenceError,
    load_coffee_model,
    predict_coffee,
)


class FixedModel(torch.nn.Module):
    def forward(self, inputs):
        assert inputs.shape == (1, 3, 224, 224)
        return torch.tensor([[0.0, 4.0, 2.0, 1.0]])


def test_coffee_prediction_keeps_checkpoint_class_order_and_top3():
    classes = ["Cerscospora", "Healthy", "Leaf rust", "Phoma"]

    label, confidence_percent, top3 = predict_coffee(
        Image.new("RGB", (70, 90)), FixedModel(), classes
    )

    assert label == "Healthy"
    assert confidence_percent == pytest.approx(top3[0]["probability"] * 100)
    assert [item["label"] for item in top3] == ["Healthy", "Leaf rust", "Phoma"]
    assert sum(item["probability"] for item in top3) < 1.0
    assert all(0.0 <= item["probability"] <= 1.0 for item in top3)


def test_coffee_loader_checks_checkpoint_hash_and_mapping(tmp_path):
    model = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "coffee.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["one", "two"],
        },
        checkpoint_path,
    )
    expected_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    with pytest.raises(CoffeeInferenceError, match="SHA-256"):
        load_coffee_model(checkpoint_path, "0" * 64)
    with pytest.raises(CoffeeInferenceError, match="class_names"):
        load_coffee_model(checkpoint_path, expected_hash, ["two", "one"])


def test_coffee_loader_reports_missing_checkpoint(tmp_path):
    with pytest.raises(CoffeeInferenceError, match="Không tìm thấy"):
        load_coffee_model(tmp_path / "missing.pth")


@pytest.mark.parametrize(
    "logits",
    [torch.zeros((1, 3)), torch.full((1, 4), float("nan"))],
)
def test_coffee_rejects_bad_model_outputs(logits):
    class BadModel(torch.nn.Module):
        def forward(self, inputs):
            return logits

    classes = ["Cerscospora", "Healthy", "Leaf rust", "Phoma"]
    with pytest.raises(CoffeeInferenceError):
        predict_coffee(Image.new("RGB", (70, 90)), BadModel(), classes)
