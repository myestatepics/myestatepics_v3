
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def restore_protected_chroma(
    original: np.ndarray,
    corrected: np.ndarray,
    scene: Scene,
) -> tuple[np.ndarray, dict]:
    """
    Restore only a small fraction of original chroma.
    Floors are now included in protected materials.
    """
    strength_value = 0.22

    original_f = original.astype(np.float32) / 255.0
    y = (
        0.2126 * original_f[..., 0]
        + 0.7152 * original_f[..., 1]
        + 0.0722 * original_f[..., 2]
    )

    guard = _smoothstep(0.05, 0.10, y)
    protected = np.maximum(
        scene.masks["protected"],
        scene.masks["floor"],
    )
    protection = protected * guard

    src = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    dst = cv2.cvtColor(
        corrected,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    before = dst[..., 1:3].copy()
    strength = (strength_value * protection)[..., None]

    dst[..., 1:3] = (
        dst[..., 1:3] * (1.0 - strength)
        + src[..., 1:3] * strength
    )

    restored = np.sqrt(
        np.sum(
            (dst[..., 1:3] - before) ** 2,
            axis=2,
        )
    )

    out = cv2.cvtColor(
        np.clip(dst, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    )

    return out, {
        "protected_percent": float(
            np.mean(protection > 0.5) * 100.0
        ),
        "restore_strength": strength_value,
        "floor_included": True,
        "mean_chroma_restored": (
            float(np.mean(restored[protection > 0.1]))
            if np.any(protection > 0.1)
            else 0.0
        ),
    }
