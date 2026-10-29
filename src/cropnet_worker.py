"""One-shot binary-stdin worker for the Streamlit app's separate TF environment."""
from __future__ import annotations

import json
import sys

from src.cropnet_inference import load_model, predict


def main() -> None:
    content = sys.stdin.buffer.read()
    if not content or len(content) > 20_000_000:
        raise ValueError("Image is empty or exceeds 20 MB")
    result = predict(content, load_model())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
