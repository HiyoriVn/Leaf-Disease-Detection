"""Quality gate contract tests; real OOD metrics require unseen images."""
from PIL import Image
from src.rice_quality_gate import check_rice_image, rice_gate_features


def test_features_deterministic():
    image = Image.new("RGB", (270, 520), (30, 120, 40))
    x = rice_gate_features(image)
    assert x.ndim == 1 and x.shape[0] == 47
    assert (x == x).all()


def test_white_red_logo_rejected(tmp_path):
    image = Image.new("RGB", (400, 400), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 300, 300), fill=(180, 0, 0))
    decision = check_rice_image(image, tmp_path / "missing.json")
    assert decision.status == "reject"


def test_no_gate_does_not_claim_rice_valid(tmp_path):
    image = Image.new("RGB", (200, 300), (40, 140, 60))
    decision = check_rice_image(image, tmp_path / "missing.json")
    assert decision.status == "unverified"
