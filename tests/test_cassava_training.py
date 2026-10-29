from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from safetensors.torch import load_file
import timm
import yaml

from scripts.train_cassava import CassavaDataset, read_manifest, set_trainable
from src.cassava_inference import load_config, preprocess_image
from src.classification_metrics import classification_metrics
from scripts.evaluate_cassava_validation import EXPECTED_CHECKPOINTS, predict_probabilities
import torch


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


def test_efficientnet_b3_preprocessing_and_weights():
    config = yaml.safe_load(Path("configs/cassava_efficientnet_b3.yaml").read_text(encoding="utf-8"))
    data_root = Path(yaml.safe_load(Path("configs/dataset_config.yaml").read_text(encoding="utf-8"))["data_root"])
    rows, _ = read_manifest(Path("manifests/cassava.csv"), data_root)
    row = next(r for r in rows if r["split"] == "val")
    image, label = CassavaDataset([row], data_root, False, config["seed"],
                                  resize=config["resize"], crop=config["input_size"],
                                  interpolation=cv2.INTER_CUBIC)[0]
    assert image.shape == (3, 300, 300)
    assert label == int(row["label_index"])
    weights = Path(config["pretrained_file"])
    if weights.is_file():
        model = timm.create_model(config["architecture"], pretrained=False, num_classes=1000)
        model.load_state_dict(load_file(str(weights)), strict=True)
        model.reset_classifier(5)
        assert model.classifier.out_features == 5
        set_trainable(model, "head")
        assert any(p.requires_grad for p in model.classifier.parameters())
        assert not any(p.requires_grad for p in model.blocks[0].parameters())
        set_trainable(model, "last")
        assert any(p.requires_grad for p in model.blocks[-1].parameters())
        assert not any(p.requires_grad for p in model.blocks[0].parameters())
        set_trainable(model, "all")
        assert all(p.requires_grad for p in model.parameters())


def test_symptom_preserving_preprocess_and_aspect_resize():
    data_root = Path(yaml.safe_load(Path("configs/dataset_config.yaml").read_text(encoding="utf-8"))["data_root"])
    rows, _ = read_manifest(Path("manifests/cassava.csv"), data_root)
    row = next(r for r in rows if r["split"] == "train")
    dataset = CassavaDataset([row], data_root, True, 3, resize=343, crop=300,
                             interpolation=cv2.INTER_CUBIC, resize_mode="shortest_edge",
                             augmentation_profile="symptom_preserving", train_crop="random")
    tensor, label = dataset[0]
    assert tensor.shape == (3, 300, 300)
    assert np.isfinite(tensor.numpy()).all()
    assert label == int(row["label_index"])


def test_efficientnet_b4_factory_and_trainability():
    model = timm.create_model("efficientnet_b4.ra2_in1k", pretrained=False, num_classes=1000)
    model.reset_classifier(5)
    set_trainable(model, "last")
    assert any(p.requires_grad for p in model.classifier.parameters())
    assert any(p.requires_grad for p in model.blocks[-1].parameters())
    assert not any(p.requires_grad for p in model.blocks[0].parameters())


def test_phase3e_model_hash_registry_and_tta_probability_contract():
    assert EXPECTED_CHECKPOINTS == {
        "baseline": "7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b",
        "v2": "7398eb98fe0917d330baea90c5c34c6160e2e781aab31dc7f9c73c9c653e2901",
        "v3": "0e76a24e1d979ee29f508b83fe6059f89f7a13a167effcaddc3eacfea0cf669c",
    }

    class Dummy(torch.nn.Module):
        def forward(self, x):
            return torch.stack((x.mean((1, 2, 3)), -x.mean((1, 2, 3)),
                                x[:, 0].mean((1, 2)), x[:, 1].mean((1, 2)),
                                x[:, 2].mean((1, 2))), dim=1)

    x = torch.rand(2, 3, 20, 20)
    for tta in ("none", "hflip", "dihedral"):
        probs = predict_probabilities(Dummy(), x, tta)
        assert probs.shape == (2, 5)
        torch.testing.assert_close(probs.sum(1), torch.ones(2))
