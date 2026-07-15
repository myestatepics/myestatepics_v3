from __future__ import annotations

"""Bounded RC4.2 photographic finish applied after the frozen HDR renderer."""

import cv2
import numpy as np

from .scene import Scene


def _smooth01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(rgb.astype(np.float32), 0.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055).astype(np.float32)


def _union(scene: Scene, names: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, np.float32)
    for name in names:
        result = np.maximum(result, scene.masks.get(name, np.zeros(shape, np.float32)))
    return np.clip(result, 0.0, 1.0)


def apply_lightroom_look(rgb: np.ndarray, scene: Scene) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Apply luminance-only texture, black, and highlight refinements.

    Existing feathered semantic masks control eligibility. RGB ratios are
    retained exactly in linear light, so this stage cannot rotate hue or add
    saturation. All corrections are bounded and contain no exposure planner.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("photographic finish expects an HxWx3 RGB image")

    linear = _srgb_to_linear(rgb)
    y = (0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]).astype(np.float32)
    original_y = y.copy()
    shape = y.shape

    material = _union(scene, ("floor", "cabinet", "table", "rug", "door"), shape)
    exclusions = _union(scene, ("wall", "ceiling", "window", "curtain", "bed"), shape)
    material *= 1.0 - exclusions
    material *= _smooth01((y - 0.025) / 0.10) * _smooth01((0.88 - y) / 0.22)

    # Guided bases retain strong edges. Fine and broad residuals approximate a
    # restrained Texture/Clarity finish without sharpening or halo-producing
    # unsharp masks.
    fine_base = cv2.ximgproc.guidedFilter(y, y, radius=5, eps=5e-5)
    broad_base = cv2.ximgproc.guidedFilter(y, y, radius=18, eps=1.5e-3)
    fine_detail = y - fine_base
    broad_detail = fine_base - broad_base
    detail_delta = material * (0.10 * fine_detail + 0.045 * broad_detail)
    detail_delta = np.clip(detail_delta, -0.018, 0.018)
    y = np.maximum(y + detail_delta, 0.0)

    # A 2.5% dark-tone anchor, continuously faded before the midtones. The
    # multiplicative form cannot erase detail or turn photographed black gray.
    black_targets = _union(scene, ("cabinet", "floor", "window", "table"), shape)
    black_gate = _smooth01((0.34 - y) / 0.28)
    black_weight = black_targets * black_gate
    y *= 1.0 - 0.025 * black_weight

    curtain = scene.masks.get("curtain", np.zeros(shape, np.float32))
    window = scene.masks.get("window", np.zeros(shape, np.float32)) * (1.0 - curtain)
    window_gate = _smooth01((y - 0.62) / 0.28)
    window_weight = window * window_gate
    y *= 1.0 - (1.0 - 2.0 ** -0.15) * window_weight

    # Recessed/fixture cores receive a smooth shoulder while surrounding glow
    # is retained. The maximum reduction is deliberately smaller than windows.
    lamp = np.maximum(
        scene.masks.get("lamp", np.zeros(shape, np.float32)),
        scene.masks.get("ceiling", np.zeros(shape, np.float32)),
    ) * (1.0 - curtain)
    lamp_gate = _smooth01((y - 0.70) / 0.25)
    lamp_weight = lamp * lamp_gate
    y *= 1.0 - 0.035 * lamp_weight

    scale = y / np.maximum(original_y, 1e-8)
    # Preserve linear RGB ratios and prevent a later channel clip from changing
    # hue. Highlight reductions naturally remain below this bound.
    gamut_scale = 1.0 / np.maximum(np.max(linear, axis=2), 1.0)
    scale = np.minimum(scale, gamut_scale)
    output = _linear_to_srgb(linear * scale[..., None])

    changed = np.abs(y - original_y)
    return np.clip(output, 0.0, 1.0), {
        "applied": True,
        "luminance_only": True,
        "material_microcontrast_strength": 0.10,
        "material_clarity_strength": 0.045,
        "black_anchor_percent": 2.5,
        "window_rolloff_stops": 0.15,
        "lamp_core_rolloff_percent": 3.5,
        "material_coverage_percent": float(np.mean(material > 0.05) * 100.0),
        "window_coverage_percent": float(np.mean(window_weight > 0.05) * 100.0),
        "lamp_coverage_percent": float(np.mean(lamp_weight > 0.05) * 100.0),
        "mean_absolute_luminance_change": float(np.mean(changed)),
        "p99_absolute_luminance_change": float(np.percentile(changed, 99)),
    }
