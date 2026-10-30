import hashlib
import json

import pytest

from src.model_registry import ModelRegistryError, available_models, load_model_registry


def make_registry(tmp_path, checkpoint_bytes=b"weights"):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    config = tmp_path / "cassava.json"
    config.write_text(
        json.dumps({"checkpoint": "weights.pth", "checkpoint_sha256": digest}),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "cassava-test",
                        "crop": "cassava",
                        "label": "Test model",
                        "enabled": True,
                        "config": "cassava.json",
                        "sha256": digest,
                    },
                    {
                        "id": "coffee-missing",
                        "crop": "coffee",
                        "label": "Missing model",
                        "enabled": True,
                        "checkpoint": "absent.pth",
                        "sha256": "0" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_registry_returns_only_existing_hash_verified_models(tmp_path):
    registry = make_registry(tmp_path)

    models = available_models(registry, tmp_path, "cassava")

    assert [model["id"] for model in models] == ["cassava-test"]
    assert models[0]["checkpoint_path"] == tmp_path / "weights.pth"


def test_registry_rejects_modified_checkpoint(tmp_path):
    registry = make_registry(tmp_path)
    (tmp_path / "weights.pth").write_bytes(b"different weights")

    assert available_models(registry, tmp_path, "cassava") == []


def test_registry_rejects_duplicate_ids(tmp_path):
    registry = make_registry(tmp_path)
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["models"].append(dict(data["models"][0]))
    registry.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelRegistryError, match="duplicate model id"):
        load_model_registry(registry)
