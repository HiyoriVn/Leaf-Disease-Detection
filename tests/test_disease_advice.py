"""Advice sections remain one-to-one with saved model class labels."""
from pathlib import Path
from src.disease_advice import get_advice

ROOT = Path(__file__).resolve().parents[1]


def test_every_rice_label_has_advice():
    import json
    labels = json.loads((ROOT / "Rice-leaf/riceLeafDisease_trainingResult/class_names.json").read_text(encoding="utf-8"))
    for label in labels:
        advice = get_advice("rice", label)
        assert advice and "Nguồn" not in advice[:20]


def test_coffee_labels():
    for label in ("Cerscospora", "Healthy", "Leaf rust", "Phoma"):
        assert get_advice("coffee", label)


def test_cassava_labels():
    for label in ("Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
                  "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy"):
        assert get_advice("cassava", label)


def test_unknown_label_no_advice():
    assert get_advice("rice", "not_a_valid_disease") is None
    assert get_advice("not_a_crop", "Healthy") is None
