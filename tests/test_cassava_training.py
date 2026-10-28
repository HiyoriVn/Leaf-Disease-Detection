from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import timm
import yaml

from scripts.train_cassava import CassavaDataset, read_manifest
from src.cassava_inference import load_config, preprocess_image
from src.classification_metrics import classification_metrics


def test_metrics_known_confusion_matrix():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["A", "B"])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.75
    assert result["macro_f1"] == 0.7333333333333334


def test_frozen_splits_and_evaluation_preprocessing():
    data_root = Path(yaml.safe_load(Path("configs/dataset_config.yaml").read_text(encoding="utf-8"))["data_root"])
    rows, counts = read_manifest(Path("manifests/cassava.csv"), data_root)
    assert counts == {"train": 14978, "val": 3210, "internal_test": 3209}
    row = next(r for r in rows if r["split"] == "val")
    tensor, label = CassavaDataset([row], data_root, False, 1)[0]
    with Image.open(data_root / row["image_path"]) as image:
        expected = preprocess_image(image, load_config())[0]
    np.testing.assert_allclose(tensor.numpy(), expected.numpy(), rtol=0, atol=0)
    assert label == int(row["label_index"])


def test_imagenet_weights_strictly_match_1000_class_backbone():
    weights = Path("checkpoints/imagenet_mobilenetv3_small_100.safetensors")
    if not weights.is_file():
        return
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.reset_classifier(5)
    assert model.classifier.out_features == 5
