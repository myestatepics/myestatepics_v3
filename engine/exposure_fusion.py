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
    floor_target_median: float = 0.42
    floor_max_lift: float = 0.18
    floor_recovery_strength: float = 0.90
    semantic_exposure_enabled: bool = True
    wall_target_median: float = 0.64
    ceiling_target_median: float = 0.74
    semantic_floor_target_median: float = 0.50
    wall_max_lift: float = 0.16
    ceiling_max_lift: float = 0.14
    semantic_floor_max_lift: float = 0.22
    semantic_strength: float = 0.82
    semantic_feather_divisor: float = 160.0


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
    floor_median = None
    floor_coverage = 0.0
    if scene is not None:
        structure = scene.masks.get("structure")
        window = scene.masks.get("window")
        floor = scene.masks.get("floor")
        if structure is not None:
            valid = structure > 0.45
            if window is not None:
                valid &= window < 0.25
            valid &= (y > 0.015) & (y < 0.90)
            values = y[valid]
            if values.size >= 512:
                room_median = float(np.median(values))
                room_p25 = float(np.percentile(values, 25.0))
        if floor is not None:
            floor_valid = (floor > 0.50) & (y > 0.015) & (y < 0.90)
            floor_values = y[floor_valid]
            floor_coverage = float(np.mean(floor > 0.50) * 100.0)
            if floor_values.size >= 256:
                floor_median = float(np.median(floor_values))

    if room_median < 0.27 or room_p25 < 0.12:
        profile = "dark_interior"
        defaults = {
            "target_median": 0.59,
            "max_bright_ev": 1.70,
            "final_strength": 0.98,
            "local_contrast_strength": 0.10,
        }
    elif room_median < 0.40 or room_p25 < 0.20:
        profile = "dim_interior"
        defaults = {
            "target_median": 0.56,
            "max_bright_ev": 1.50,
            "final_strength": 0.95,
            "local_contrast_strength": 0.09,
        }
    else:
        profile = "balanced_interior"
        defaults = {
            "target_median": 0.52,
            "max_bright_ev": 1.15,
            "final_strength": 0.84,
            "local_contrast_strength": 0.07,
        }

    # A visibly dark floor is a strong signal that the MLS interior still
    # needs more opening, but cap the adjustment so bright rooms stay natural.
    if floor_median is not None and floor_coverage >= 2.0:
        floor_deficit = max(0.0, 0.34 - floor_median)
        defaults["target_median"] = min(0.61, defaults["target_median"] + min(0.035, floor_deficit * 0.18))
        defaults["final_strength"] = min(0.99, defaults["final_strength"] + min(0.025, floor_deficit * 0.12))

    defaults.update(base)
    cfg = ExposureFusionConfig(**defaults)
    return cfg, {
        "adaptive_profile": profile,
        "input_median": median,
        "input_mean": mean,
        "input_p25": p25,
        "room_median": room_median,
        "room_p25": room_p25,
        "floor_median": floor_median,
        "floor_coverage_pct": floor_coverage,
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



def recover_floor_shadows(
    reference: np.ndarray,
    corrected: np.ndarray,
    scene,
    config: ExposureFusionConfig,
) -> tuple[np.ndarray, dict]:
    """Open dark floor detail in LAB luminance without changing floor hue."""
    if scene is None or "floor" not in scene.masks:
        return corrected.copy(), {"applied": False, "reason": "no_floor_mask"}

    floor = np.clip(scene.masks["floor"].astype(np.float32), 0.0, 1.0)
    confident = floor > 0.50
    if np.count_nonzero(confident) < 256:
        return corrected.copy(), {"applied": False, "reason": "insufficient_floor_pixels"}

    ref_f = reference.astype(np.float32) / 255.0
    ref_y = _luminance_float(ref_f)
    floor_values = ref_y[confident & (ref_y > 0.015) & (ref_y < 0.90)]
    if floor_values.size < 256:
        return corrected.copy(), {"applied": False, "reason": "invalid_floor_pixels"}

    floor_median = float(np.median(floor_values))
    requested = max(0.0, float(config.floor_target_median) - floor_median)
    lift = min(float(config.floor_max_lift), requested)
    if lift < 0.008:
        return corrected.copy(), {
            "applied": False,
            "reason": "floor_already_bright",
            "input_floor_median": floor_median,
        }

    # Feather mask edges while keeping the confident interior effective.
    sigma = max(2.0, min(reference.shape[:2]) / 180.0)
    feather = cv2.GaussianBlur(floor, (0, 0), sigmaX=sigma, sigmaY=sigma)
    feather = np.clip(feather, 0.0, 1.0)
    visibility = _smoothstep(0.025, 0.13, ref_y)
    # Darker floor pixels receive more recovery, but photographed black remains anchored.
    darkness = 1.0 - _smoothstep(floor_median, min(0.70, floor_median + 0.30), ref_y)
    amount = float(config.floor_recovery_strength) * feather * visibility * (0.45 + 0.55 * darkness)

    lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(np.float32)
    current_l = lab[..., 0] / 255.0
    target_l = np.minimum(1.0, current_l + lift)
    lab[..., 0] = 255.0 * (current_l * (1.0 - amount) + target_l * amount)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out_y = _luminance_float(out.astype(np.float32) / 255.0)
    return out, {
        "applied": True,
        "input_floor_median": floor_median,
        "output_floor_median": float(np.median(out_y[confident])),
        "requested_lift": requested,
        "applied_lift_cap": lift,
        "floor_target_median": float(config.floor_target_median),
        "floor_coverage_pct": float(np.mean(confident) * 100.0),
    }


def _region_median(y: np.ndarray, mask: np.ndarray, minimum_pixels: int = 256) -> float | None:
    valid = (mask > 0.50) & (y > 0.015) & (y < 0.92)
    values = y[valid]
    if values.size < minimum_pixels:
        return None
    return float(np.median(values))


def _feather_mask(mask: np.ndarray, shape: tuple[int, int], divisor: float) -> np.ndarray:
    sigma = max(2.0, min(shape) / max(divisor, 1.0))
    blurred = cv2.GaussianBlur(np.clip(mask.astype(np.float32), 0.0, 1.0), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def apply_semantic_exposure(
    reference: np.ndarray,
    corrected: np.ndarray,
    scene,
    config: ExposureFusionConfig,
) -> tuple[np.ndarray, dict]:
    """Apply bounded luminance-only exposure targets by semantic region.

    This pass does not alter RGB channels independently. It lifts ceilings,
    walls and floors according to their own measured luminance, while
    excluding windows and feathering all mask boundaries.
    """
    if not config.semantic_exposure_enabled or scene is None:
        return corrected.copy(), {"applied": False, "reason": "disabled_or_no_scene"}

    h, w = reference.shape[:2]
    ref_y = _luminance_float(reference.astype(np.float32) / 255.0)
    lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(np.float32)
    current_l = lab[..., 0] / 255.0

    window = np.clip(scene.masks.get("window", np.zeros((h, w), np.float32)), 0.0, 1.0)
    regions = {
        "ceiling": ("ceiling", config.ceiling_target_median, config.ceiling_max_lift),
        "wall": ("wall", config.wall_target_median, config.wall_max_lift),
        "floor": ("floor", config.semantic_floor_target_median, config.semantic_floor_max_lift),
    }

    total_amount = np.zeros((h, w), np.float32)
    total_delta = np.zeros((h, w), np.float32)
    logs: dict[str, dict] = {}

    for region_name, (mask_name, target, max_lift) in regions.items():
        raw = np.clip(scene.masks.get(mask_name, np.zeros((h, w), np.float32)), 0.0, 1.0)
        raw = raw * (1.0 - window)
        median = _region_median(ref_y, raw)
        if median is None:
            logs[region_name] = {"applied": False, "reason": "insufficient_pixels"}
            continue

        requested = max(0.0, float(target) - median)
        lift = min(float(max_lift), requested)
        if lift < 0.006:
            logs[region_name] = {
                "applied": False,
                "reason": "already_bright",
                "input_median": median,
                "target_median": float(target),
            }
            continue

        feather = _feather_mask(raw, (h, w), config.semantic_feather_divisor)
        visibility = _smoothstep(0.02, 0.12, ref_y)
        highlight_guard = 1.0 - _smoothstep(0.72, 0.94, ref_y)
        amount = float(config.semantic_strength) * feather * visibility * highlight_guard

        # Darker pixels inside each material receive more lift, but the entire
        # region moves toward its own target to avoid a patchy result.
        darkness = 1.0 - _smoothstep(median, min(0.90, median + 0.28), ref_y)
        region_amount = amount * (0.55 + 0.45 * darkness)
        total_amount = np.maximum(total_amount, region_amount)
        total_delta += lift * region_amount

        logs[region_name] = {
            "applied": True,
            "input_median": median,
            "target_median": float(target),
            "requested_lift": requested,
            "applied_lift_cap": lift,
            "coverage_pct": float(np.mean(raw > 0.50) * 100.0),
        }

    if not np.any(total_amount > 0.001):
        return corrected.copy(), {"applied": False, "reason": "no_regions_needed_lift", "regions": logs}

    target_l = np.clip(current_l + total_delta, 0.0, 1.0)
    lab[..., 0] = 255.0 * (current_l * (1.0 - total_amount) + target_l * total_amount)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    out_y = _luminance_float(out.astype(np.float32) / 255.0)
    for region_name, (mask_name, _, _) in regions.items():
        if logs.get(region_name, {}).get("applied"):
            raw = np.clip(scene.masks.get(mask_name, np.zeros((h, w), np.float32)), 0.0, 1.0) * (1.0 - window)
            median = _region_median(out_y, raw)
            logs[region_name]["output_median"] = median

    return out, {
        "applied": True,
        "engine": "semantic_luminance_targets_v1",
        "regions": logs,
        "mean_active_strength": float(np.mean(total_amount[total_amount > 0.01])) if np.any(total_amount > 0.01) else 0.0,
    }

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
    output, semantic_log = apply_semantic_exposure(rgb, output, scene, cfg)
    output, floor_log = recover_floor_shadows(rgb, output, scene, cfg)

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
        "engine": "semantic_scene_aware_exposure_fusion_v4",
        "metrics": asdict(metrics),
        "config": asdict(cfg),
        "adaptive": adaptive_log,
        "output_p25": out_p25,
        "semantic_exposure": semantic_log,
        "floor_recovery": floor_log,
    }
