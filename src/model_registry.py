"""Small, explicit registry for plant models supported by the demo app."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ModelRegistryError(ValueError):
    """Raised when the model registry is malformed or inconsistent."""


def load_model_registry(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["models"]
        if data.get("schema_version") != 1 or not isinstance(entries, list):
            raise ValueError("unsupported schema_version or models list")
        seen: set[str] = set()
        for entry in entries:
            model_id = entry["id"]
            if not isinstance(model_id, str) or not model_id or model_id in seen:
                raise ValueError(f"invalid or duplicate model id: {model_id!r}")
            seen.add(model_id)
            if entry.get("crop") not in {"cassava", "coffee", "rice"}:
                raise ValueError(f"unsupported crop for {model_id}")
            sha = entry.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"invalid SHA-256 for {model_id}")
        return entries
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelRegistryError(f"Cannot read model registry {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_models(
    registry_path: Path,
    repository_root: Path,
    crop: str,
    coffee_model_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return enabled models whose local checkpoint exists and matches its hash."""
    available: list[dict[str, Any]] = []
    for entry in load_model_registry(registry_path):
        if entry["crop"] != crop or not entry.get("enabled", True):
            continue

        if entry.get("config"):
            config_path = (repository_root / entry["config"]).resolve()
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                checkpoint_path = (repository_root / config["checkpoint"]).resolve()
                expected_sha = config["checkpoint_sha256"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if expected_sha != entry["sha256"]:
                continue
            entry["config_path"] = config_path
        else:
            model_path = coffee_model_path or (
                os.environ.get(entry.get("path_env", "")) if entry.get("path_env") else None
            )
            checkpoint_path = Path(model_path) if model_path else repository_root / entry["checkpoint"]

        if checkpoint_path.is_file() and _sha256(checkpoint_path) == entry["sha256"]:
            entry["checkpoint_path"] = checkpoint_path.resolve()
            available.append(entry)
    return available
