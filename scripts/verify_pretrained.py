"""Verify local ImageNet safetensors against a training config before transfer/run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import timm
import torch
import yaml
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    weight_path = ROOT / config["pretrained_file"]
    actual_hash = sha256(weight_path)
    if actual_hash != config["pretrained_sha256"]:
        raise ValueError(f"SHA-256 mismatch: {weight_path}")
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(str(weight_path)), strict=True)
    model.reset_classifier(5)
    model.eval()
    with torch.inference_mode():
        shape = list(model(torch.zeros(1, 3, int(config["input_size"]), int(config["input_size"]))).shape)
    if shape != [1, 5]:
        raise ValueError(f"Unexpected output shape: {shape}")
    print(json.dumps({"architecture": config["architecture"], "weights": str(weight_path),
                      "bytes": weight_path.stat().st_size, "sha256": actual_hash,
                      "input_shape": [1, 3, int(config["input_size"]), int(config["input_size"])],
                      "output_shape": shape, "torch": torch.__version__, "timm": timm.__version__}))


if __name__ == "__main__":
    main()
