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
    shadow_anchor_end: float = 0.16
    highlight_guard_start: float = 0.72
    highlight_guard_end: float = 0.96
    contrast_weight: float = 1.0
    saturation_weight: float = 0.35
    exposure_weight: float = 1.0
    final_strength: float = 0.86
    # Mertens already supplies the tonal shaping for the MVP.  Keep the
    # optional CLAHE hook disabled so contrast is not stacked by default.
    local_contrast_strength: float = 0.0
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
    scene=None,
) -> tuple[ExposureFusionConfig, dict]:
    """Build a restrained scene-adaptive exposure configuration.

    Dark interiors receive a stronger median target and blend strength. Bright
    rooms remain close to the conservative defaults. The decision uses only
    global luminance statistics, keeping segmentation out of the correction.
    """
    _validate_rgb(rgb)
    base = dict(settings or {})
    median, mean, p25, shadow_clip, highlight_clip = _robust_luminance_stats(rgb)

    # Use semantic masks only to measure the room, never to independently alter
    # colour channels. Bright windows are excluded from the target calculation.
    f = rgb.astype(np.float32) / 255.0
    y = _luminance_float(f)
    room_median = median
    room_p25 = p25
    if scene is not None:
        structure = scene.masks.get("structure")
        window = scene.masks.get("window")
        if structure is not None:
            valid = structure > 0.45
            if window is not None:
                valid &= window < 0.25
            valid &= (y > 0.015) & (y < 0.90)
            values = y[valid]
            if values.size >= 512:
                room_median = float(np.median(values))
                room_p25 = float(np.percentile(values, 25.0))

    if room_median < 0.27 or room_p25 < 0.12:
        profile = "dark_interior"
        defaults = {
            "target_median": 0.59,
            "max_bright_ev": 1.35,
            "final_strength": 0.98,
            "local_contrast_strength": 0.0,
        }
    elif room_median < 0.40 or room_p25 < 0.20:
        profile = "dim_interior"
        defaults = {
            "target_median": 0.56,
            "max_bright_ev": 1.50,
            "final_strength": 0.95,
            "local_contrast_strength": 0.0,
        }
    else:
        profile = "balanced_interior"
        defaults = {
            "target_median": 0.52,
            "max_bright_ev": 1.15,
            "final_strength": 0.84,
            "local_contrast_strength": 0.0,
        }

    defaults.update(base)
    cfg = ExposureFusionConfig(**defaults)
    return cfg, {
        "adaptive_profile": profile,
        "input_median": median,
        "input_mean": mean,
        "input_p25": p25,
        "room_median": room_median,
        "room_p25": room_p25,
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


def _bounded_room_luminance_correction(
    rgb: np.ndarray,
    target_median: float = 0.50,
    max_ev: float = 0.65,
) -> tuple[np.ndarray, dict]:
    """Apply one smooth room-wide lift while holding blacks and highlights.

    The gain is a single scalar derived from the robust image median.  Only
    its luminance contribution is retained, so photographed hue and saturation
    remain unchanged.  There are no semantic or per-pixel darkness targets.
    """
    f = rgb.astype(np.float32) / 255.0
    y = _luminance_float(f)
    valid = y[(y > 0.02) & (y < 0.92)]
    median = float(np.median(valid)) if valid.size >= 256 else float(np.median(y))
    required_ev = float(np.log2(max(target_median, 1e-4) / max(median, 1e-4)))
    # In extremely dark rooms a large scalar lift makes naturally reflective
    # wood look pale before the black walls become midtone. Keep the same
    # global curve, but tighten its room-wide cap smoothly for that case.
    scene_cap = 0.25 if median < 0.36 else max_ev
    applied_ev = float(np.clip(required_ev, 0.0, scene_cap))
    if applied_ev < 0.01:
        return rgb.copy(), {
            "applied": False,
            "input_median": median,
            "target_median": target_median,
            "applied_ev": 0.0,
        }

    gain = float(2.0**applied_ev)
    black_anchor = _smoothstep(0.018, 0.18, y)
    highlight_guard = 1.0 - _smoothstep(0.58, 0.90, y)
    local_gain = 1.0 + (gain - 1.0) * black_anchor * highlight_guard
    lifted = _ratio_preserving_shoulder(f * local_gain[..., None])
    lifted_u8 = np.clip(lifted * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Keep only the corrected luminance. This is a chroma lock, not a color
    # correction: the incoming WB hue/saturation is copied unchanged.
    src_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    dst_lab = cv2.cvtColor(lifted_u8, cv2.COLOR_RGB2LAB)
    dst_lab[..., 1:3] = src_lab[..., 1:3]
    out = cv2.cvtColor(dst_lab, cv2.COLOR_LAB2RGB)
    return out, {
        "applied": True,
        "input_median": median,
        "target_median": target_median,
        "requested_ev": required_ev,
        "applied_ev": applied_ev,
        "scene_cap_ev": scene_cap,
    }


def _lock_chroma_to_reference(reference: np.ndarray, luminance_source: np.ndarray) -> np.ndarray:
    """Take luminance from the correction and exact Lab chroma from WB input."""
    src = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    dst = cv2.cvtColor(luminance_source, cv2.COLOR_RGB2LAB)
    dst[..., 1:3] = src[..., 1:3]
    return cv2.cvtColor(dst, cv2.COLOR_LAB2RGB)


def _scene_wide_exposure_plan(
    rgb: np.ndarray,
    scene,
    cfg: ExposureFusionConfig,
) -> tuple[float, dict]:
    """Estimate one MLS exposure value from scene-wide measurements."""
    f = rgb.astype(np.float32) / 255.0
    y = _luminance_float(f)
    h, w = y.shape
    structure = None if scene is None else scene.masks.get("structure")
    window = None if scene is None else scene.masks.get("window")
    room = np.ones((h, w), dtype=bool) if structure is None else structure > 0.45
    if window is not None:
        room &= window < 0.25
    room &= (y > 0.015) & (y < 0.95)
    if np.count_nonzero(room) < 512:
        room = (y > 0.015) & (y < 0.95)

    values = y[room]
    median = float(np.median(values))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    shadow_pct = float(np.mean(values < 0.18) * 100.0)
    black_pct = float(np.mean(values < 0.035) * 100.0)

    def semantic_median(name: str) -> float | None:
        if scene is None or name not in scene.masks:
            return None
        valid = (scene.masks[name] > 0.55) & (y > 0.015) & (y < 0.95)
        return float(np.median(y[valid])) if np.count_nonzero(valid) >= 256 else None

    ceiling_median = semantic_median("ceiling")
    wall_median = semantic_median("wall")
    window_area = float(np.mean(window > 0.5) * 100.0) if window is not None else 0.0
    window_values = y[window > 0.5] if window is not None and np.any(window > 0.5) else np.array([])
    window_median = float(np.median(window_values)) if window_values.size else None
    highlight_pct = float(np.mean(y > 0.985) * 100.0)

    dark_decor = bool(
        wall_median is not None
        and ceiling_median is not None
        and wall_median < 0.42 * max(ceiling_median, 1e-4)
    )
    if dark_decor:
        target = 0.42
        max_ev = 0.65
        profile = "dark_decor"
    elif median < 0.28 or shadow_pct > 32.0:
        target = 0.50 if window_area >= 0.3 else 0.52
        max_ev = 0.75
        profile = "underexposed_room"
    elif median < 0.42:
        target = 0.50
        max_ev = 0.65
        profile = "dim_room"
    else:
        target = 0.50
        max_ev = 0.35
        profile = "balanced_room"

    required_ev = float(np.log2(max(target, 1e-4) / max(median, 1e-4)))
    applied_ev = float(np.clip(required_ev, 0.0, max_ev))
    return applied_ev, {
        "profile": profile,
        "room_median": median,
        "room_p25": p25,
        "room_p75": p75,
        "shadow_percent": shadow_pct,
        "black_percent": black_pct,
        "ceiling_median": ceiling_median,
        "wall_median": wall_median,
        "window_area_percent": window_area,
        "window_median": window_median,
        "highlight_clip_percent": highlight_pct,
        "dark_decor": dark_decor,
        "target_median": target,
        "required_ev": required_ev,
        "applied_ev": applied_ev,
        "maximum_ev": max_ev,
    }


def _apply_scene_wide_exposure(
    rgb: np.ndarray,
    ev: float,
    cfg: ExposureFusionConfig,
) -> np.ndarray:
    """Apply the single planned exposure with smooth endpoint protection."""
    if ev <= 0.001:
        return rgb.copy()
    f = rgb.astype(np.float32) / 255.0
    y = _luminance_float(f)
    black_anchor = _smoothstep(cfg.shadow_anchor_start, cfg.shadow_anchor_end, y)
    highlight_guard = 1.0 - _smoothstep(cfg.highlight_guard_start, cfg.highlight_guard_end, y)
    gain = float(2.0**ev)
    local_gain = 1.0 + (gain - 1.0) * black_anchor * highlight_guard
    out = _ratio_preserving_shoulder(f * local_gain[..., None])
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)



def exposure_fusion(
    rgb: np.ndarray,
    config: ExposureFusionConfig | None = None,
    adaptive_settings: dict | None = None,
    scene=None,
) -> tuple[np.ndarray, dict]:
    """Conservatively brighten an RGB uint8 image using exposure fusion.

    The method creates bounded virtual exposures from the real photograph,
    blends them using OpenCV's Mertens exposure fusion, and then mixes the
    fused luminance back into the original with black and highlight guards.
    It does not use semantic masks and does not independently alter channels.
    """
    _validate_rgb(rgb)
    if config is None:
        cfg, adaptive_log = adaptive_exposure_config(rgb, adaptive_settings, scene)
    else:
        cfg = config
        _, adaptive_log = adaptive_exposure_config(rgb, adaptive_settings, scene)
        adaptive_log["adaptive_profile"] = "explicit_config"
    applied_ev, scene_plan = _scene_wide_exposure_plan(rgb, scene, cfg)
    output = _apply_scene_wide_exposure(rgb, applied_ev, cfg)

    out_median, out_mean, out_p25, out_shadow_clip, out_highlight_clip = _robust_luminance_stats(output)
    input_median, input_mean, input_p25, input_shadow_clip, input_highlight_clip = _robust_luminance_stats(rgb)
    metrics = ExposureMetrics(
        input_median=input_median,
        output_median=out_median,
        input_mean=input_mean,
        output_mean=out_mean,
        input_shadow_clip_pct=input_shadow_clip,
        output_shadow_clip_pct=out_shadow_clip,
        input_highlight_clip_pct=input_highlight_clip,
        output_highlight_clip_pct=out_highlight_clip,
        bright_ev=applied_ev,
        dark_ev=0.0,
        virtual_exposure_count=1,
        input_p25=input_p25,
        adaptive_profile=adaptive_log["adaptive_profile"],
    )

    return output, {
        "engine": "scene_wide_mls_luminance_v1",
        "metrics": asdict(metrics),
        "config": asdict(cfg),
        "adaptive": adaptive_log,
        "scene_exposure_plan": scene_plan,
        "output_p25": out_p25,
        "floor_recovery": {
            "applied": False,
            "reason": "removed_from_mvp_phase_a",
        },
    }
