import json

import pytest

from src.inference import ModelError, load_manifest


def test_manifest_names_all_three_crops():
    configs = load_manifest()
    assert set(configs) == {"rice", "coffee", "cassava"}
    assert all(config.image_size == 224 for config in configs.values())


def test_manifest_rejects_incompatible_architecture(tmp_path):
    source = json.loads((load_manifest.__globals__["MANIFEST_PATH"]).read_text(encoding="utf-8"))
    source["rice"]["architecture"] = "unknown"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ModelError, match="only explicit TorchScript"):
        load_manifest(manifest)


def test_unconfigured_label_mapping_is_reported():
    config = load_manifest()["rice"]
    from src.inference import load_model

    with pytest.raises(ModelError, match="class labels are not configured"):
        load_model(config)
