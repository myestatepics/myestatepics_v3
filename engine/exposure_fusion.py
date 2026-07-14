from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExposureFusionConfig:
    """Configuration for the conservative MVP exposure-fusion engine."""

    target_median: float = 0.50
    max_bright_ev: float = 1.20
    max_dark_ev: float = 0.70
    shadow_anchor_start: float = 0.025
    shadow_anchor_end: float = 0.11
    highlight_guard_start: float = 0.72
    highlight_guard_end: float = 0.96
    contrast_weight: float = 1.0
    saturation_weight: float = 0.35
    exposure_weight: float = 1.0
    final_strength: float = 0.86


@dataclass(frozen=True)
class ExposureMetrics:
    input_median: float
    output_median: float
    input_mean: float
    output_mean: float
    input_shadow_clip_pct: float
    output_shadow_clip_pct: float
    input_highlight_clip_pct: float
    output_highlight_clip_pct: float
    bright_ev: float
    dark_ev: float
    virtual_exposure_count: int


def _validate_rgb(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a numpy array")
    if rgb.dtype != np.uint8:
        raise TypeError("rgb must have dtype uint8")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if rgb.size == 0:
        raise ValueError("rgb must not be empty")


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance_float(rgb_float: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb_float[..., 0]
        + 0.7152 * rgb_float[..., 1]
        + 0.0722 * rgb_float[..., 2]
    )


def _robust_luminance_stats(rgb: np.ndarray) -> tuple[float, float, float, float]:
    f = rgb.astype(np.float32) / 255.0
    y = _luminance_float(f)
    valid = y[(y > 0.01) & (y < 0.99)]
    if valid.size < 256:
        valid = y.reshape(-1)
    return (
        float(np.median(valid)),
        float(np.mean(valid)),
        float(np.mean(y <= (4.0 / 255.0)) * 100.0),
        float(np.mean(y >= (251.0 / 255.0)) * 100.0),
    )


def _bounded_ev_plan(median: float, cfg: ExposureFusionConfig) -> tuple[float, float]:
    """Choose restrained virtual exposure offsets from image luminance."""
    median = max(median, 1e-4)
    required_ev = float(np.log2(max(cfg.target_median, 1e-4) / median))

    # The bright frame does the useful lifting. It is deliberately capped.
    bright_ev = float(np.clip(required_ev, 0.25, cfg.max_bright_ev))

    # The dark frame exists mainly to protect windows/highlights. Do not make
    # it stronger than needed for already-dark rooms.
    highlight_need = _smoothstep(0.62, 0.90, np.array([median], np.float32))[0]
    dark_ev = float(np.clip(0.30 + 0.40 * highlight_need, 0.25, cfg.max_dark_ev))
    return bright_ev, dark_ev


def _ratio_preserving_shoulder(rgb_float: np.ndarray) -> np.ndarray:
    """Compress values above 1.0 by scaling all channels equally per pixel."""
    peak = np.max(rgb_float, axis=2, keepdims=True)
    scale = np.where(peak > 1.0, 1.0 / np.maximum(peak, 1e-6), 1.0)
    return np.clip(rgb_float * scale, 0.0, 1.0)


def _virtual_exposure(rgb: np.ndarray, ev: float, cfg: ExposureFusionConfig) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    y = _luminance_float(f)
    gain = float(2.0**ev)

    if ev > 0.0:
        # Preserve true blacks and progressively reduce lift near highlights.
        black_anchor = _smoothstep(cfg.shadow_anchor_start, cfg.shadow_anchor_end, y)
        highlight_guard = 1.0 - _smoothstep(
            cfg.highlight_guard_start,
            cfg.highlight_guard_end,
            y,
        )
        local_gain = 1.0 + (gain - 1.0) * black_anchor * highlight_guard
    else:
        # Darkening is global and ratio preserving; its purpose is highlight
        # detail for Mertens fusion, not local tone mapping.
        local_gain = np.full_like(y, gain, dtype=np.float32)

    out = f * local_gain[..., None]
    out = _ratio_preserving_shoulder(out)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def generate_virtual_exposures(
    rgb: np.ndarray,
    config: ExposureFusionConfig | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Return dark, original and bright virtual exposures plus diagnostics."""
    _validate_rgb(rgb)
    cfg = config or ExposureFusionConfig()
    median, mean, shadow_clip, highlight_clip = _robust_luminance_stats(rgb)
    bright_ev, dark_ev = _bounded_ev_plan(median, cfg)

    exposures = [
        _virtual_exposure(rgb, -dark_ev, cfg),
        rgb.copy(),
        _virtual_exposure(rgb, bright_ev, cfg),
    ]
    return exposures, {
        "input_median": median,
        "input_mean": mean,
        "input_shadow_clip_pct": shadow_clip,
        "input_highlight_clip_pct": highlight_clip,
        "bright_ev": bright_ev,
        "dark_ev": dark_ev,
        "config": asdict(cfg),
    }


def _blend_with_original(
    original: np.ndarray,
    fused: np.ndarray,
    cfg: ExposureFusionConfig,
) -> np.ndarray:
    original_f = original.astype(np.float32) / 255.0
    fused_f = fused.astype(np.float32) / 255.0
    y = _luminance_float(original_f)

    # The fused result is strongest in shadows/midtones, weaker at the two
    # extremes. This prevents black-wall lifting and protects bright windows.
    black_anchor = _smoothstep(cfg.shadow_anchor_start, cfg.shadow_anchor_end, y)
    highlight_guard = 1.0 - _smoothstep(
        cfg.highlight_guard_start,
        cfg.highlight_guard_end,
        y,
    )
    strength = cfg.final_strength * black_anchor * highlight_guard
    out = original_f * (1.0 - strength[..., None]) + fused_f * strength[..., None]
    out = _ratio_preserving_shoulder(out)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def exposure_fusion(
    rgb: np.ndarray,
    config: ExposureFusionConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Conservatively brighten an RGB uint8 image using exposure fusion.

    The method creates bounded virtual exposures from the real photograph,
    blends them using OpenCV's Mertens exposure fusion, and then mixes the
    fused luminance back into the original with black and highlight guards.
    It does not use semantic masks and does not independently alter channels.
    """
    _validate_rgb(rgb)
    cfg = config or ExposureFusionConfig()
    exposures, plan = generate_virtual_exposures(rgb, cfg)

    merger = cv2.createMergeMertens(
        contrast_weight=float(cfg.contrast_weight),
        saturation_weight=float(cfg.saturation_weight),
        exposure_weight=float(cfg.exposure_weight),
    )
    # OpenCV accepts uint8 BGR/RGB arrays equally for weight computation; all
    # channels are treated symmetrically. Output is float32 in approximately
    # [0, 1], though tiny excursions are possible.
    fused_float = merger.process(exposures)
    fused = np.clip(fused_float, 0.0, 1.0)
    fused_u8 = np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8)
    output = _blend_with_original(rgb, fused_u8, cfg)

    out_median, out_mean, out_shadow_clip, out_highlight_clip = _robust_luminance_stats(output)
    metrics = ExposureMetrics(
        input_median=plan["input_median"],
        output_median=out_median,
        input_mean=plan["input_mean"],
        output_mean=out_mean,
        input_shadow_clip_pct=plan["input_shadow_clip_pct"],
        output_shadow_clip_pct=out_shadow_clip,
        input_highlight_clip_pct=plan["input_highlight_clip_pct"],
        output_highlight_clip_pct=out_highlight_clip,
        bright_ev=plan["bright_ev"],
        dark_ev=plan["dark_ev"],
        virtual_exposure_count=len(exposures),
    )

    return output, {
        "engine": "bounded_mertens_exposure_fusion_v1",
        "metrics": asdict(metrics),
        "config": asdict(cfg),
    }
