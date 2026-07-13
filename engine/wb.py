
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


def _rgb_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def _luminance(f: np.ndarray) -> np.ndarray:
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _valid_reference(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f = _rgb_float(rgb)
    y = _luminance(f)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    return (mask > 0.5) & (y > 0.08) & (y < 0.95) & (chroma < 20.0)


def _gains_from_pixels(f: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    count = int(np.count_nonzero(valid))
    if count == 0:
        return np.ones(3, dtype=np.float32), 0, np.ones(3, dtype=np.float32)
    medians = np.median(f[valid], axis=0)
    target = float(np.mean(medians))
    gains = target / np.maximum(medians, 1e-5)
    return gains.astype(np.float32), count, medians.astype(np.float32)


def global_pass1_wb(rgb: np.ndarray, scene: Scene) -> tuple[np.ndarray, dict]:
    """
    v3.2 Fix 1 (pass 1 of two-pass WB): remove the bulk of any global cast
    BEFORE neutral-reference validation, so a strong cast cannot disqualify
    the ceiling reference (the circular-validation bug).
    Shades-of-gray (Minkowski p=6) over the structure mask, limits [0.72, 1.35].
    Skipped (identity) when computed gains are all within 1.0 +/- 0.03.
    """
    f = _rgb_float(rgb)
    y = _luminance(f)
    structure = scene.masks.get("structure")
    if structure is not None and np.count_nonzero(structure > 0.5) >= 2000:
        sel = (structure > 0.5) & (y > 0.04) & (y < 0.95)
    else:
        sel = (y > 0.04) & (y < 0.95)
    samples = f[sel]
    if samples.shape[0] < 500:
        return rgb.copy(), {"pass1_applied": False, "pass1_gains": [1.0, 1.0, 1.0],
                            "pass1_reason": "insufficient_pixels"}
    illum = np.power(np.mean(np.power(np.clip(samples, 1e-6, 1.0), 6.0), axis=0), 1.0 / 6.0)
    target = float(np.mean(illum))
    gains = np.clip(target / np.maximum(illum, 1e-5), 0.72, 1.35).astype(np.float32)
    if np.all(np.abs(gains - 1.0) <= 0.03):
        return rgb.copy(), {"pass1_applied": False, "pass1_gains": [float(g) for g in gains],
                            "pass1_reason": "within_noop_band"}
    out = np.clip(f * gains[None, None, :], 0.0, 1.0)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), {
        "pass1_applied": True,
        "pass1_gains": [float(g) for g in gains],
        "pass1_reason": "cast_detected",
    }


def semantic_white_balance(
    rgb: np.ndarray,
    scene: Scene,
    tones: dict,
) -> tuple[np.ndarray, dict]:
    f = _rgb_float(rgb)
    h, w = rgb.shape[:2]
    reference_used = "shades_of_gray"
    confidence = "LOW"
    half_agreement = None

    valid = np.zeros((h, w), dtype=bool)
    if tones["ceiling"]["ceiling_is_neutral_reference"]:
        ceiling_valid = _valid_reference(rgb, scene.masks["ceiling"])
        if np.count_nonzero(ceiling_valid) >= 2000:
            valid = ceiling_valid
            reference_used = "ceiling"

            left = ceiling_valid.copy()
            left[:, w // 2:] = False
            right = ceiling_valid.copy()
            right[:, :w // 2] = False
            gl, cl, _ = _gains_from_pixels(f, left)
            gr, cr, _ = _gains_from_pixels(f, right)
            agreement = float(np.max(np.abs(gl - gr))) if cl >= 800 and cr >= 800 else 999.0
            half_agreement = agreement
            if (
                cl >= 800 and cr >= 800
                and agreement <= 0.06
                and scene.coverage.get("ceiling", 0.0) >= 4.0
            ):
                confidence = "HIGH"
            else:
                confidence = "MEDIUM"

    if not np.any(valid):
        upper_wall = scene.masks["wall"].copy()
        upper_wall[int(h * 0.40):] = 0.0
        light_wall = tones["wall_class_map"] == 3
        upper_valid = _valid_reference(rgb, upper_wall * light_wall.astype(np.float32))
        if np.count_nonzero(upper_valid) >= 2000:
            valid = upper_valid
            reference_used = "upper_wall"
            confidence = "MEDIUM"

    if np.any(valid):
        raw_gains, pixel_count, medians = _gains_from_pixels(f, valid)
    else:
        structure = scene.masks["structure"] > 0.5
        sample_mask = structure
        if np.count_nonzero(sample_mask) < 2000:
            sample_mask = np.ones((h, w), dtype=bool)
        y = _luminance(f)
        sample_mask &= (y > 0.04) & (y < 0.95)
        samples = f[sample_mask]
        illum = np.power(np.mean(np.power(np.clip(samples, 1e-6, 1.0), 6.0), axis=0), 1.0 / 6.0)
        target = float(np.mean(illum))
        raw_gains = target / np.maximum(illum, 1e-5)
        pixel_count = int(samples.shape[0])
        medians = illum.astype(np.float32)

    limits = (0.75, 1.32) if confidence == "HIGH" else (0.82, 1.22)
    gains = np.clip(raw_gains, limits[0], limits[1]).astype(np.float32)
    corrected = np.clip(f * gains[None, None, :], 0.0, 1.0)

    return np.clip(corrected * 255.0 + 0.5, 0, 255).astype(np.uint8), {
        "reference_used": reference_used,
        "confidence": confidence,
        "half_agreement": half_agreement,
        "reference_pixel_count": int(pixel_count),
        "reference_medians": [float(x) for x in medians],
        "gains": [float(x) for x in gains],
        "limits_used": [float(limits[0]), float(limits[1])],
    }


def global_safe_white_balance(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    f = _rgb_float(rgb)
    y = _luminance(f)
    valid = (y > 0.04) & (y < 0.95)
    samples = f[valid]
    illum = np.power(np.mean(np.power(np.clip(samples, 1e-6, 1.0), 6.0), axis=0), 1.0 / 6.0)
    target = float(np.mean(illum))
    gains = np.clip(target / np.maximum(illum, 1e-5), 0.85, 1.15).astype(np.float32)
    out = np.clip(f * gains[None, None, :], 0.0, 1.0)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), {
        "reference_used": "global_safe_shades_of_gray",
        "confidence": "LOW",
        "gains": [float(x) for x in gains],
        "limits_used": [0.85, 1.15],
        "reference_pixel_count": int(samples.shape[0]),
    }
