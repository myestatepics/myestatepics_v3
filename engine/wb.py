from __future__ import annotations

"""Conservative, single-pass white balance for the MyEstatePics MVP.

The previous implementation could apply a global pass and then a semantic pass.
That made channel gains compound and allowed a questionable semantic reference to
change an image that had already been corrected.  This module has one primary
entry point, :func:`conservative_white_balance`, and keeps the old public
functions only as compatibility wrappers.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .scene import Scene


@dataclass(frozen=True)
class WhiteBalanceConfig:
    """Safety limits for the single-pass white-balance engine."""

    noop_gain_delta: float = 0.025
    high_confidence_limit: float = 0.12
    medium_confidence_limit: float = 0.10
    low_confidence_limit: float = 0.08
    minimum_reference_pixels: int = 1200
    minimum_half_pixels: int = 400
    half_agreement_limit: float = 0.055


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    f = np.clip(linear, 0.0, 1.0)
    srgb = np.where(f <= 0.0031308, 12.92 * f, 1.055 * np.power(f, 1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _luminance(linear: np.ndarray) -> np.ndarray:
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


def _candidate_pixels(rgb: np.ndarray, base_mask: np.ndarray | None) -> np.ndarray:
    """Return safe reference candidates without assuming that every wall is white."""

    linear = _srgb_to_linear(rgb)
    y = _luminance(linear)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    # Exclude deep shadows, near-clipped highlights, and strongly saturated objects.
    valid = (y > 0.035) & (y < 0.88) & (saturation < 0.42)
    if base_mask is not None:
        valid &= base_mask > 0.5
    return valid


def _robust_illuminant(linear: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    count = int(np.count_nonzero(valid))
    if count == 0:
        return np.ones(3, dtype=np.float32), 0

    samples = linear[valid]
    # Remove channel outliers before estimating the illuminant.
    lo = np.percentile(samples, 10.0, axis=0)
    hi = np.percentile(samples, 90.0, axis=0)
    trimmed = samples[np.all((samples >= lo) & (samples <= hi), axis=1)]
    if trimmed.shape[0] >= max(200, count // 5):
        samples = trimmed

    # Shades-of-gray is more stable than gray-world in mixed-light interiors.
    p = 6.0
    illum = np.power(np.mean(np.power(np.clip(samples, 1e-6, 1.0), p), axis=0), 1.0 / p)
    return illum.astype(np.float32), int(samples.shape[0])


def _raw_gains(illuminant: np.ndarray) -> np.ndarray:
    # Geometric mean avoids privileging a single channel and keeps exposure
    # approximately stable after chromatic adaptation.
    target = float(np.exp(np.mean(np.log(np.maximum(illuminant, 1e-6)))))
    return (target / np.maximum(illuminant, 1e-6)).astype(np.float32)


def _half_agreement(
    linear: np.ndarray,
    valid: np.ndarray,
    minimum_half_pixels: int,
) -> float | None:
    h, w = valid.shape
    left = valid.copy()
    left[:, w // 2 :] = False
    right = valid.copy()
    right[:, : w // 2] = False

    left_illum, left_count = _robust_illuminant(linear, left)
    right_illum, right_count = _robust_illuminant(linear, right)
    if left_count < minimum_half_pixels or right_count < minimum_half_pixels:
        return None

    left_gains = _raw_gains(left_illum)
    right_gains = _raw_gains(right_illum)
    return float(np.max(np.abs(left_gains - right_gains)))


def _reference_choice(
    rgb: np.ndarray,
    scene: Scene | None,
    config: WhiteBalanceConfig,
) -> tuple[np.ndarray, str, str, float | None]:
    h, w = rgb.shape[:2]
    linear = _srgb_to_linear(rgb)

    if scene is not None:
        ceiling = scene.masks.get("ceiling")
        if ceiling is not None:
            valid = _candidate_pixels(rgb, ceiling)
            agreement = _half_agreement(linear, valid, config.minimum_half_pixels)
            count = int(np.count_nonzero(valid))
            coverage = float(scene.coverage.get("ceiling", 0.0))
            if count >= config.minimum_reference_pixels and coverage >= 3.0:
                # A ceiling whose two halves imply materially different gains
                # is mixed-lit (often warm fixtures versus window light), not
                # a trustworthy global neutral reference.
                if agreement is not None and agreement <= config.half_agreement_limit:
                    return valid, "ceiling", "HIGH", agreement

        wall = scene.masks.get("wall")
        if wall is not None:
            upper_wall = wall.copy()
            upper_wall[int(h * 0.45) :] = 0.0
            valid = _candidate_pixels(rgb, upper_wall)
            agreement = _half_agreement(linear, valid, config.minimum_half_pixels)
            if (
                int(np.count_nonzero(valid)) >= config.minimum_reference_pixels
                and (agreement is None or agreement <= config.half_agreement_limit * 1.35)
            ):
                return valid, "upper_wall", "MEDIUM", agreement

        structure = scene.masks.get("structure")
        if structure is not None:
            valid = _candidate_pixels(rgb, structure)
            if int(np.count_nonzero(valid)) >= config.minimum_reference_pixels:
                return valid, "structure", "LOW", None

    valid = _candidate_pixels(rgb, None)
    if int(np.count_nonzero(valid)) < config.minimum_reference_pixels:
        # Last-resort fallback still excludes only clipped endpoints.
        y = _luminance(linear)
        valid = (y > 0.025) & (y < 0.92)
    return valid, "global_shades_of_gray", "LOW", None


def conservative_white_balance(
    rgb: np.ndarray,
    scene: Scene | None = None,
    *,
    config: WhiteBalanceConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one bounded white-balance correction.

    The correction is computed and applied in linear-light RGB.  Semantic masks
    choose a possible neutral reference; they never receive local color edits.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 RGB array")

    cfg = config or WhiteBalanceConfig()
    linear = _srgb_to_linear(rgb)
    valid, reference, confidence, agreement = _reference_choice(rgb, scene, cfg)
    illuminant, pixel_count = _robust_illuminant(linear, valid)

    if pixel_count < 50 or not np.all(np.isfinite(illuminant)):
        return rgb.copy(), {
            "applied": False,
            "reason": "insufficient_reference_pixels",
            "reference_used": reference,
            "confidence": "LOW",
            "reference_pixel_count": int(pixel_count),
            "half_agreement": agreement,
            "raw_gains": [1.0, 1.0, 1.0],
            "gains": [1.0, 1.0, 1.0],
            "gain_limit": 0.0,
        }

    raw = _raw_gains(illuminant)
    limit = {
        "HIGH": cfg.high_confidence_limit,
        "MEDIUM": cfg.medium_confidence_limit,
        "LOW": cfg.low_confidence_limit,
    }[confidence]
    gains = np.clip(raw, 1.0 - limit, 1.0 + limit).astype(np.float32)

    if float(np.max(np.abs(gains - 1.0))) <= cfg.noop_gain_delta:
        return rgb.copy(), {
            "applied": False,
            "reason": "within_noop_band",
            "reference_used": reference,
            "confidence": confidence,
            "reference_pixel_count": int(pixel_count),
            "half_agreement": agreement,
            "illuminant": [float(x) for x in illuminant],
            "raw_gains": [float(x) for x in raw],
            "gains": [float(x) for x in gains],
            "gain_limit": float(limit),
        }

    corrected = np.clip(linear * gains[None, None, :], 0.0, 1.0)
    out = _linear_to_srgb(corrected)
    return out, {
        "applied": True,
        "reason": "bounded_cast_correction",
        "reference_used": reference,
        "confidence": confidence,
        "reference_pixel_count": int(pixel_count),
        "half_agreement": agreement,
        "illuminant": [float(x) for x in illuminant],
        "raw_gains": [float(x) for x in raw],
        "gains": [float(x) for x in gains],
        "gain_limit": float(limit),
    }


def gentle_wall_chroma_consistency(
    rgb: np.ndarray,
    scene: Scene,
    *,
    strength: float = 0.28,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce verified mixed-light chroma divergence on neutral wall paint.

    Luminance is never changed. The already-feathered wall confidence and a
    broad weighted color field prevent hard edges; naturally colored walls
    are excluded so accent paint is not neutralized.
    """
    wall = scene.masks.get("wall")
    if wall is None or np.count_nonzero(wall > 0.5) < 2400:
        return rgb.copy(), {"applied": False, "reason": "insufficient_wall"}

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[..., 0] / 255.0
    ab = lab[..., 1:3] - 128.0
    chroma = np.sqrt(np.sum(ab * ab, axis=2))
    valid = (wall > 0.55) & (l > 0.15) & (l < 0.92) & (chroma < 22.0)
    if np.count_nonzero(valid) < 2400:
        return rgb.copy(), {"applied": False, "reason": "no_neutral_wall_family"}

    target = np.median(ab[valid], axis=0).astype(np.float32)
    weight = np.clip(wall, 0.0, 1.0) * valid.astype(np.float32)
    h, w = wall.shape
    scale = min(1.0, 640.0 / max(h, w))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small_weight = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA)
    small_ab = cv2.resize(ab, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = max(12.0, max(sh, sw) / 9.0)
    denominator = cv2.GaussianBlur(small_weight, (0, 0), sigma) + 1e-5
    local_a = cv2.GaussianBlur(small_ab[..., 0] * small_weight, (0, 0), sigma) / denominator
    local_b = cv2.GaussianBlur(small_ab[..., 1] * small_weight, (0, 0), sigma) / denominator
    local_small = np.stack([local_a, local_b], axis=2)
    local = cv2.resize(local_small, (w, h), interpolation=cv2.INTER_LINEAR)
    divergence = np.sqrt(np.sum((local - target[None, None, :]) ** 2, axis=2))
    p90 = float(np.percentile(divergence[valid], 90))
    if p90 < 3.0:
        return rgb.copy(), {
            "applied": False,
            "reason": "wall_chroma_consistent",
            "divergence_p90": p90,
        }

    gate = np.clip((divergence - 2.0) / 7.0, 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    adaptive_strength = min(0.42, strength + max(0.0, p90 - 4.0) * 0.020)
    amount = adaptive_strength * np.clip(wall, 0.0, 1.0) * gate * valid.astype(np.float32)
    corrected_ab = ab - (local - target[None, None, :]) * amount[..., None]
    lab[..., 1:3] = corrected_ab + 128.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out, {
        "applied": True,
        "strength": adaptive_strength,
        "divergence_p90": p90,
        "target_ab": [float(target[0]), float(target[1])],
        "affected_percent": float(np.mean(amount > 0.01) * 100.0),
    }


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------

def global_pass1_wb(rgb: np.ndarray, scene: Scene) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility name for the single primary white-balance pass."""

    out, log = conservative_white_balance(rgb, scene)
    return out, {
        **log,
        "pass1_applied": bool(log["applied"]),
        "pass1_gains": list(log["gains"]),
        "pass1_reason": str(log["reason"]),
        "compatibility_wrapper": "global_pass1_wb",
    }


def semantic_white_balance(
    rgb: np.ndarray,
    scene: Scene,
    tones: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deprecated second-pass hook.

    White balance is already completed by ``global_pass1_wb``.  Returning the
    input unchanged prevents compounded gains while preserving old callers until
    the Phase 3 pipeline cleanup removes this hook.
    """

    del scene, tones
    return rgb.copy(), {
        "applied": False,
        "reason": "single_pass_already_completed",
        "reference_used": "primary_pass",
        "confidence": "N/A",
        "reference_pixel_count": 0,
        "half_agreement": None,
        "raw_gains": [1.0, 1.0, 1.0],
        "gains": [1.0, 1.0, 1.0],
        "gain_limit": 0.0,
        "compatibility_wrapper": "semantic_white_balance",
    }


def global_safe_white_balance(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Single-pass global fallback for scenes without trustworthy masks."""

    out, log = conservative_white_balance(rgb, None)
    return out, {**log, "compatibility_wrapper": "global_safe_white_balance"}
