from dataclasses import replace
from io import BytesIO
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.cassava_inference import (
    CassavaInferenceError, decode_image, load_cassava_model, load_config,
    predict_image, preprocess_image, _sha256,
)


def image_bytes(mode="RGB"):
    image = Image.new(mode, (64, 48), 128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_and_rgb_tensor_shape():
    config = load_config()
    image = decode_image(image_bytes("L"), config)
    tensor = preprocess_image(image, config)
    assert image.mode == "L"
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    expected = (128 / 255.0 - np.array(config.mean)) / np.array(config.std)
    np.testing.assert_allclose(tensor[0, :, 0, 0].numpy(), expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_invalid_image_is_rejected(payload):
    with pytest.raises(CassavaInferenceError):
        decode_image(payload, load_config())


def test_large_upload_is_rejected():
    config = replace(load_config(), max_upload_bytes=10)
    with pytest.raises(CassavaInferenceError, match="20 MB"):
        decode_image(image_bytes(), config)


def test_probability_and_class_mapping():
    class FixedModel(torch.nn.Module):
        def forward(self, x):
            assert x.shape == (1, 3, 224, 224)
            return torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])

    config = load_config()
    result = predict_image(decode_image(image_bytes(), config), FixedModel(), config)
    assert result["output_shape"] == [1, 5]
    assert result["predicted_index"] == 4
    assert result["predicted_label"] == "Healthy"
    assert [x["index"] for x in result["top3"]] == [4, 3, 2]
    assert sum(result["probabilities"]) == pytest.approx(1.0, abs=1e-6)


def test_invalid_output_shape_is_rejected():
    class BadModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(1, 4)

    config = load_config()
    with pytest.raises(CassavaInferenceError, match="Đầu ra"):
        predict_image(decode_image(image_bytes(), config), BadModel(), config)


def test_strict_state_dict_rejects_incompatible_weights(tmp_path):
    path = tmp_path / "wrong.pth"
    torch.save({"model_state": {}}, path)
    config = replace(load_config(), checkpoint=path, checkpoint_sha256=_sha256(path))
    with pytest.raises(CassavaInferenceError, match="Không nạp được checkpoint"):
        load_cassava_model(config)


def test_real_checkpoint_matches_phase2_smoke_predictions():
    config = load_config()
    data_root = Path(yaml.safe_load(Path("configs/dataset_config.yaml").read_text(encoding="utf-8"))["data_root"])
    csv_path = Path("reports/smoke_cassava/predictions.csv")
    if not config.checkpoint.is_file() or not csv_path.is_file() or not data_root.is_dir():
        pytest.skip("Local checkpoint, Phase 2 smoke CSV, or images unavailable")
    model = load_cassava_model(config)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    for row in rows:
        image = decode_image((data_root / row["image_path"]).read_bytes(), config)
        result = predict_image(image, model, config)
        assert result["predicted_label"] == row["predicted_label"]
        assert result["confidence"] == pytest.approx(float(row["confidence"]), abs=1e-5)
