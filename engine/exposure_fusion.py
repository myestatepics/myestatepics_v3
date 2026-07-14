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
    local_contrast_strength: float = 0.10
    local_contrast_clip_limit: float = 1.6
    local_contrast_grid: int = 8


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
    input_p25: float
    adaptive_profile: str


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


def _robust_luminance_stats(rgb: np.ndarray) -> tuple[float, float, float, float, float]:
    f = rgb.astype(np.float32) / 255.0
    y = _luminance_float(f)
    valid = y[(y > 0.01) & (y < 0.99)]
    if valid.size < 256:
        valid = y.reshape(-1)
    return (
        float(np.median(valid)),
        float(np.mean(valid)),
        float(np.percentile(valid, 25.0)),
        float(np.mean(y <= (4.0 / 255.0)) * 100.0),
        float(np.mean(y >= (251.0 / 255.0)) * 100.0),
    )


def adaptive_exposure_config(
    rgb: np.ndarray,
    settings: dict | None = None,
) -> tuple[ExposureFusionConfig, dict]:
    """Build a restrained scene-adaptive exposure configuration.

    Dark interiors receive a stronger median target and blend strength. Bright
    rooms remain close to the conservative defaults. The decision uses only
    global luminance statistics, keeping segmentation out of the correction.
    """
    _validate_rgb(rgb)
    base = dict(settings or {})
    median, mean, p25, shadow_clip, highlight_clip = _robust_luminance_stats(rgb)

    if median < 0.24 or p25 < 0.10:
        profile = "dark_interior"
        defaults = {
            "target_median": 0.54,
            "max_bright_ev": 1.45,
            "final_strength": 0.94,
            "local_contrast_strength": 0.12,
        }
    elif median < 0.36 or p25 < 0.17:
        profile = "dim_interior"
        defaults = {
            "target_median": 0.52,
            "max_bright_ev": 1.30,
            "final_strength": 0.90,
            "local_contrast_strength": 0.10,
        }
    else:
        profile = "balanced_interior"
        defaults = {
            "target_median": 0.49,
            "max_bright_ev": 1.10,
            "final_strength": 0.82,
            "local_contrast_strength": 0.08,
        }

    defaults.update(base)
    cfg = ExposureFusionConfig(**defaults)
    return cfg, {
        "adaptive_profile": profile,
        "input_median": median,
        "input_mean": mean,
        "input_p25": p25,
        "input_shadow_clip_pct": shadow_clip,
        "input_highlight_clip_pct": highlight_clip,
    }


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
    median, mean, p25, shadow_clip, highlight_clip = _robust_luminance_stats(rgb)
    bright_ev, dark_ev = _bounded_ev_plan(median, cfg)

    exposures = [
        _virtual_exposure(rgb, -dark_ev, cfg),
        rgb.copy(),
        _virtual_exposure(rgb, bright_ev, cfg),
    ]
    return exposures, {
        "input_median": median,
        "input_mean": mean,
        "input_p25": p25,
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


def _gentle_luminance_contrast(
    rgb: np.ndarray,
    original: np.ndarray,
    cfg: ExposureFusionConfig,
) -> np.ndarray:
    """Add restrained local contrast in LAB luminance only.

    The effect is faded out in deep blacks and highlights, and chroma channels
    are left untouched. This improves cabinet/island separation without HDR
    halos or material hue changes.
    """
    if cfg.local_contrast_strength <= 0.0:
        return rgb.copy()

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_u8 = np.clip(lab[..., 0], 0, 255).astype(np.uint8)
    grid = max(2, int(cfg.local_contrast_grid))
    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.local_contrast_clip_limit),
        tileGridSize=(grid, grid),
    )
    enhanced_l = clahe.apply(l_u8).astype(np.float32)

    original_f = original.astype(np.float32) / 255.0
    y = _luminance_float(original_f)
    black_guard = _smoothstep(0.04, 0.13, y)
    highlight_guard = 1.0 - _smoothstep(0.72, 0.94, y)
    amount = cfg.local_contrast_strength * black_guard * highlight_guard
    lab[..., 0] = lab[..., 0] * (1.0 - amount) + enhanced_l * amount
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def exposure_fusion(
    rgb: np.ndarray,
    config: ExposureFusionConfig | None = None,
    adaptive_settings: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Conservatively brighten an RGB uint8 image using exposure fusion.

    The method creates bounded virtual exposures from the real photograph,
    blends them using OpenCV's Mertens exposure fusion, and then mixes the
    fused luminance back into the original with black and highlight guards.
    It does not use semantic masks and does not independently alter channels.
    """
    _validate_rgb(rgb)
    if config is None:
        cfg, adaptive_log = adaptive_exposure_config(rgb, adaptive_settings)
    else:
        cfg = config
        _, adaptive_log = adaptive_exposure_config(rgb, adaptive_settings)
        adaptive_log["adaptive_profile"] = "explicit_config"
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
    output = _gentle_luminance_contrast(output, rgb, cfg)

    out_median, out_mean, out_p25, out_shadow_clip, out_highlight_clip = _robust_luminance_stats(output)
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
        input_p25=plan["input_p25"],
        adaptive_profile=adaptive_log["adaptive_profile"],
    )

    return output, {
        "engine": "adaptive_bounded_mertens_exposure_fusion_v2",
        "metrics": asdict(metrics),
        "config": asdict(cfg),
        "adaptive": adaptive_log,
        "output_p25": out_p25,
    }
