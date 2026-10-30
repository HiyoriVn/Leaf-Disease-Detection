"""Unit tests that do not require downloading or executing the real TFLite binary."""
import numpy as np
from PIL import Image

from src.rice_inference import load_rice_config, preprocess_rice, predict_rice


class FakeInterpreter:
    def __init__(self):
        self.input = None

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 224, 224, 3]), "dtype": np.float32}]

    def get_output_details(self):
        return [{"index": 1, "shape": np.array([1, 10]), "dtype": np.float32}]

    def set_tensor(self, index, tensor):
        assert index == 0
        self.input = tensor

    def invoke(self):
        assert self.input is not None

    def get_tensor(self, index):
        assert index == 1
        return np.array([[0.01, 0.01, 0.01, 0.9, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]], dtype=np.float32)


def test_rice_config_matches_training_labels():
    config = load_rice_config()
    assert config.labels[0] == "bacterial_leaf_blight"
    assert config.labels[3] == "blast"
    assert config.labels[-1] == "tungro"
    assert len(config.labels) == 10
    assert not config.preprocessing_verified


def test_preprocess_accepts_arbitrary_rgb_sizes():
    config = load_rice_config()
    for width, height in ((640, 480), (1920, 1080), (137, 497)):
        image = Image.new("RGB", (width, height), (128, 64, 32))
        out = preprocess_rice(image, config)
        assert out.shape == (1, 224, 224, 3)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out[0, 20, 20], [128, 64, 32])


def test_predict_uses_original_class_order():
    config = load_rice_config()
    interpreter = FakeInterpreter()
    image = Image.new("RGB", (300, 500), (60, 110, 80))
    result = predict_rice(image, interpreter, config)
    assert result["predicted_label"] == "blast"
    assert len(result["top3"]) == 3
    assert abs(sum(result["probabilities"]) - 1) < 1e-6
