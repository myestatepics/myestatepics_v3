
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


def treat_window_zones(
    rgb: np.ndarray,
    scene: Scene,
) -> tuple[np.ndarray, dict]:
    window = (
        scene.masks["window"]
        * (1.0 - scene.masks["mirror"])
        * (1.0 - 0.75 * scene.masks["curtain"])
    )
    curtain = scene.masks["curtain"]

    window_percent = float(np.mean(window > 0.5) * 100.0)
    curtain_percent = float(np.mean(curtain > 0.5) * 100.0)
    mirror_excluded = float(
        np.mean(
            (scene.masks["window"] > 0.5)
            & (scene.masks["mirror"] > 0.5)
        )
        * 100.0
    )

    if window_percent < 0.2:
        return rgb.copy(), {
            "status": "no_windows",
            "window_percent": window_percent,
            "curtain_percent": curtain_percent,
            "mirror_excluded_percent": mirror_excluded,
            "mean_L_reduction_window": 0.0,
            "window_strength": 0.0,
        }

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    before_l = lab[..., 0] / 255.0
    l = before_l.copy()

    window_pixels = before_l[window > 0.5]
    p50 = float(np.median(window_pixels))
    p95 = float(np.percentile(window_pixels, 95))

    recoverable_range = max(0.0, p95 - p50)
    strength = float(
        np.clip(
            0.48 + recoverable_range * 0.35,
            0.48,
            0.58,
        )
    )

    l -= (
        window
        * np.clip(l - 0.68, 0.0, 0.32)
        * strength
    )

    l -= (
        curtain
        * np.clip(l - 0.86, 0.0, 0.14)
        * 0.24
    )

    lab[..., 0] = np.clip(l * 255.0, 0, 255)

    neutral = (0.10 * window)[..., None]
    lab[..., 1:3] = (
        lab[..., 1:3] * (1.0 - neutral)
        + 128.0 * neutral
    )

    out = cv2.cvtColor(
        np.clip(lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    )

    reduction = before_l - l

    return out, {
        "status": "treated",
        "window_percent": window_percent,
        "curtain_percent": curtain_percent,
        "mirror_excluded_percent": mirror_excluded,
        "mean_L_reduction_window": (
            float(np.mean(reduction[window > 0.5]))
            if np.any(window > 0.5)
            else 0.0
        ),
        "window_strength": strength,
        "window_p50_before": p50,
        "window_p95_before": p95,
    }
