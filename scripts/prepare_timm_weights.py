"""Download/cache ImageNet timm weights as safetensors and print their SHA-256."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import timm
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"output exists; refusing to overwrite weights: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = timm.create_model(args.architecture, pretrained=True, num_classes=1000)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(args.output))
    hasher = hashlib.sha256()
    with args.output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    print(f"architecture={args.architecture}")
    print(f"pretrained_cfg={model.pretrained_cfg}")
    print(f"output={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
