"""Fail-closed OOD gate for LiSA. Calibrated parameters are trained separately.

An obvious-graphic rule rejects some synthetic non-plant pictures (e.g. red
logos on white backgrounds). It is NOT a general leaf detector. Without a
validated binary gate the status stays 'unverified', never 'accept'.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "models" / "rice_gate.json"


@dataclass(frozen=True)
class GateDecision:
    status: str  # accept | reject | unverified
    reason: str
    score: float | None = None


def rice_gate_features(image: Image.Image) -> np.ndarray:
    """Deterministic, low-cost color/texture features for optional calibrated gate."""
    rgb = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    flat = a.reshape(-1, 3)
    h = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    white = np.all(a > .90, axis=2)
    red = (r > 1.35 * g) & (r > 1.35 * b) & (r > .24)
    green = (g > 1.12 * r) & (g > 1.08 * b) & (g > .19)
    grad = np.abs(np.diff(a.mean(axis=2), axis=0)).mean()
    grad2 = np.abs(np.diff(a.mean(axis=2), axis=1)).mean()
    features = list(flat.mean(axis=0)) + list(flat.std(axis=0))
    features += list(np.percentile(flat, [10, 50, 90], axis=0).ravel())
    features += [white.mean(), red.mean(), green.mean(), (a.mean(axis=2) < .11).mean(),
                 float(grad), float(grad2), float(h[..., 1].mean()),
                 float(h[..., 1].std())]
    for c in range(3):
        features.extend(np.histogram(flat[:, c], bins=8, range=(0., 1.), density=False)[0] / flat.shape[0])
    return np.asarray(features, dtype=np.float64)


def check_rice_image(image: Image.Image, gate_path: Path = GATE_PATH) -> GateDecision:
    f = rice_gate_features(image)
    # Obvious logo/graphic heuristic: reject, not an ID/OOD acceptance rule.
    white_fraction, red_fraction, green_fraction = f[15:18]
    if white_fraction > 0.58 and red_fraction > 0.012 and green_fraction < 0.03:
        return GateDecision("reject", "Hình có đặc trưng đồ họa/logo trên nền trắng.")
    if not gate_path.is_file():
        return GateDecision("unverified", "Chưa có rice_gate.json được hiệu chỉnh.")
    try:
        data = json.loads(gate_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("feature_version") != 1:
            raise ValueError("Unsupported gate schema")
        center = np.asarray(data["mean"], dtype=np.float64)
        std = np.asarray(data["std"], dtype=np.float64)
        coef = np.asarray(data["coef"], dtype=np.float64)
        if f.shape != center.shape or f.shape != std.shape or f.shape != coef.shape:
            raise ValueError("Feature size mismatch")
        if np.any(std <= 0) or not np.all(np.isfinite(coef)):
            raise ValueError("Invalid gate calibration")
        z = np.clip(float(np.dot((f - center) / std, coef) + data["intercept"]), -60, 60)
        score = float(1.0 / (1.0 + np.exp(-z)))
        threshold = float(data["threshold"])
        if not (0.0 < threshold < 1.0):
            raise ValueError("Invalid threshold")
        if score < threshold:
            return GateDecision("reject", "Bộ kiểm tra nhận thấy ảnh không đủ giống ảnh lá lúa.", score)
        return GateDecision("accept", "Ảnh vượt qua bộ lọc lá lúa đã hiệu chỉnh.", score)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return GateDecision("unverified", "Bộ lọc ảnh chưa khả dụng hoặc có cấu hình không hợp lệ.")
