"""Isolated TensorFlow CropNet inference; run with .venv-cropnet only."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "checkpoints" / "cropnet"
EXPECTED_SHA256 = {
    "saved_model.pb": "aac5d946061721786ff2cd3e8e49515d1cf660f9982a10266b519cd679410992",
    "variables/variables.data-00000-of-00001": "dd6919af91cf984853e8c57a30a8bd08adbad99d424fda57474e871904a7a1e0",
    "variables/variables.index": "54749e7c840c85fb08409ec1911128675e743498735e08f12f50ee8d27bd9199",
}
CLASS_NAMES = (
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
    "Unknown",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model():
    for relative, expected in EXPECTED_SHA256.items():
        path = MODEL_DIR / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Missing or unexpected CropNet artifact: {relative}")
    model = tf.saved_model.load(str(MODEL_DIR))
    if not any(
        tuple(f.structured_input_signature[0][0].shape) == (None, 224, 224, 3)
        and tuple(f.structured_outputs.shape) == (None, 6)
        for f in model.__call__.concrete_functions
    ):
        raise ValueError("Unexpected CropNet input/output signature")
    return model


def preprocess(content: bytes) -> tf.Tensor:
    # Match the official tutorial: RGB, float / 255, then bilinear resize.
    with Image.open(BytesIO(content)) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = tf.cast(tf.convert_to_tensor(rgb), tf.float32) / 255.0
    return tf.expand_dims(tf.image.resize(tensor, (224, 224)), 0)


def predict(content: bytes, model) -> dict:
    image = preprocess(content)
    output = model(image, training=False)
    values = np.asarray(output, dtype=np.float64)
    if values.shape != (1, 6) or not np.isfinite(values).all():
        raise ValueError(f"Unexpected CropNet output: {values.shape}")
    probabilities = values[0]
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise ValueError("CropNet output is not a six-class probability distribution")
    index = int(np.argmax(probabilities))
    return {
        "predicted_index": index,
        "predicted_label": CLASS_NAMES[index],
        "probabilities": probabilities.tolist(),
        "input_shape": list(image.shape),
        "output_shape": list(values.shape),
    }
