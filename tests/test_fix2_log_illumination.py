from __future__ import annotations

import numpy as np

from engine.tonal import (
    _apply_illumination_gain,
    _estimate_log_illumination,
    _luminance,
)


def test_multiplicative_preserves_rgb_ratios():
    h, w = 96, 128
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = [80, 52, 31]  # synthetic wood, safely below clipping

    y = _luminance(rgb)
    illumination, reflectance, _ = _estimate_log_illumination(
        rgb=rgb,
        luminance=y,
        radius=32,
    )

    log_gain = np.full((h, w), np.log(1.45), dtype=np.float32)
    out, _ = _apply_illumination_gain(
        rgb=rgb,
        illumination=illumination,
        reflectance=reflectance,
        log_gain=log_gain,
    )

    before = rgb[20, 20].astype(np.float64)
    after = out[20, 20].astype(np.float64)

    before_ratios = np.array([
        before[0] / before[1],
        before[1] / before[2],
        before[0] / before[2],
    ])
    after_ratios = np.array([
        after[0] / after[1],
        after[1] / after[2],
        after[0] / after[2],
    ])

    relative_error = np.abs(after_ratios / before_ratios - 1.0)
    assert float(np.max(relative_error)) < 0.01


def test_intrinsic_reconstruction_without_gain_is_nearly_identical():
    rng = np.random.default_rng(123)
    rgb = rng.integers(24, 180, size=(80, 120, 3), dtype=np.uint8)

    y = _luminance(rgb)
    illumination, reflectance, _ = _estimate_log_illumination(
        rgb=rgb,
        luminance=y,
        radius=24,
    )
    out, _ = _apply_illumination_gain(
        rgb=rgb,
        illumination=illumination,
        reflectance=reflectance,
        log_gain=np.zeros_like(y, dtype=np.float32),
    )

    max_diff = int(np.max(np.abs(out.astype(np.int16) - rgb.astype(np.int16))))
    assert max_diff <= 1
