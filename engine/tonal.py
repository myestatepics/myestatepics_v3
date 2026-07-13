from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


_EPS = 1e-4


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return (
        0.2126 * f[..., 0]
        + 0.7152 * f[..., 1]
        + 0.0722 * f[..., 2]
    ).astype(np.float32)


def _guided_smooth(
    guide_rgb: np.ndarray,
    field: np.ndarray,
    radius: int,
    eps: float = 1e-3,
) -> tuple[np.ndarray, str]:
    guide = _luminance(guide_rgb)

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        smoothed = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=field.astype(np.float32),
            radius=max(8, int(radius)),
            eps=float(eps),
        )
        return smoothed.astype(np.float32), "guided_filter"

    # Safe fallback when opencv-contrib is unavailable.
    smoothed = cv2.bilateralFilter(
        field.astype(np.float32),
        d=9,
        sigmaColor=0.08,
        sigmaSpace=max(16, int(radius)),
    )
    return smoothed.astype(np.float32), "bilateral_fallback"


def _component_constant_log_gain(
    luminance: np.ndarray,
    mask: np.ndarray,
    target_map: np.ndarray,
    min_component_fraction: float,
    max_gain: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build a component-wise constant log-gain field.

    This is intentionally NOT a per-pixel target-minus-current calculation.
    Every large connected material component receives one robust gain estimate,
    which prevents bright patches inside uniformly painted dark walls.
    """
    h, w = luminance.shape
    area = h * w
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    field = np.zeros_like(luminance, dtype=np.float32)
    logs: list[dict] = []

    for component_id in range(1, count):
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        if component_area < max(64, int(area * min_component_fraction)):
            continue

        component = labels == component_id
        median_l = float(np.median(luminance[component]))
        median_target = float(np.median(target_map[component]))

        if median_target <= median_l + 1e-4:
            gain = 1.0
        else:
            gain = float(
                np.clip(
                    median_target / max(median_l, 0.03),
                    1.0,
                    max_gain,
                )
            )

        log_gain = float(np.log(gain))
        field[component] = log_gain

        logs.append({
            "component": int(component_id),
            "area_percent": float(component_area / area * 100.0),
            "median_before": median_l,
            "median_target": median_target,
            "gain": gain,
            "log_gain": log_gain,
        })

    return field, logs


def _apply_multiplicative_gain(
    rgb: np.ndarray,
    log_gain: np.ndarray,
) -> np.ndarray:
    """
    Apply illumination in linear RGB using one scalar gain per pixel.

    Multiplying all three channels by the same gain preserves RGB ratios,
    material hue, and saturation far better than additive LAB lifting.
    """
    srgb = rgb.astype(np.float32) / 255.0

    # sRGB -> approximate linear light.
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )

    gain = np.exp(log_gain).astype(np.float32)
    corrected = np.clip(linear * gain[..., None], 0.0, 1.0)

    # Linear light -> sRGB.
    out = np.where(
        corrected <= 0.0031308,
        corrected * 12.92,
        1.055 * np.power(corrected, 1.0 / 2.4) - 0.055,
    )
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def adaptive_per_class_exposure(
    rgb: np.ndarray,
    scene: Scene,
    tones: dict,
    analysis: dict,
) -> tuple[np.ndarray, dict]:
    """
    Checkpoint A: physically motivated illumination correction.

    Design rule:
        Correct the light, never the materials.

    Changes from the previous engine:
    - No additive LAB exposure lift.
    - No chroma compensation boost.
    - Wall components receive uniform multiplicative illumination gain.
    - Ceiling receives multiplicative illumination gain.
    - Floors/cabinets/furniture/rugs receive inherited room light only.
    - Hard protected masks prevent Gaussian/guided-field leakage into materials.
    """
    y = _luminance(rgb)
    h, w = y.shape
    max_side = max(h, w)

    wall_soft = np.clip(scene.masks["wall"], 0.0, 1.0)
    ceiling_soft = np.clip(scene.masks["ceiling"], 0.0, 1.0)
    door_soft = np.clip(scene.masks["door"], 0.0, 1.0)
    floor_soft = np.clip(scene.masks["floor"], 0.0, 1.0)
    protected_soft = np.clip(scene.masks["protected"], 0.0, 1.0)

    # A hard material lock is required. Feathered masks alone allowed exposure
    # leakage and caused the historic z-17 floor shift.
    floor_hard = (floor_soft > 0.20).astype(np.float32)
    protected_hard = (protected_soft > 0.20).astype(np.float32)
    material_lock = np.maximum(floor_hard, protected_hard)

    exclusion = np.maximum.reduce([
        np.clip(scene.masks["window"], 0.0, 1.0),
        np.clip(scene.masks["curtain"], 0.0, 1.0),
        np.clip(scene.masks["mirror"], 0.0, 1.0),
    ])
    exclusion_hard = (exclusion > 0.15).astype(np.float32)

    source_log_gain = np.zeros_like(y, dtype=np.float32)
    per_class: dict[str, object] = {}

    # WALLS — one gain per connected wall component.
    wall_target_map = np.asarray(
        tones["wall_target_map"],
        dtype=np.float32,
    )
    wall_field, wall_components = _component_constant_log_gain(
        luminance=y,
        mask=wall_soft,
        target_map=wall_target_map,
        min_component_fraction=0.01,
        max_gain=1.55,
    )
    source_log_gain = np.maximum(source_log_gain, wall_field)
    per_class["wall_components"] = wall_components

    # CEILING — one robust gain for the full ceiling.
    ceiling_pixels = ceiling_soft > 0.5
    if np.any(ceiling_pixels):
        ceiling_before = float(np.median(y[ceiling_pixels]))
        ceiling_target = float(tones["ceiling"]["target"])
        ceiling_gain = float(
            np.clip(
                ceiling_target / max(ceiling_before, 0.08),
                1.0,
                2.10,
            )
        )
        ceiling_log_gain = float(np.log(ceiling_gain))
        source_log_gain = np.maximum(
            source_log_gain,
            ceiling_log_gain * (ceiling_soft > 0.35).astype(np.float32),
        )
        per_class["ceiling"] = {
            "median_before": ceiling_before,
            "target": ceiling_target,
            "gain": ceiling_gain,
        }
    else:
        per_class["ceiling"] = {
            "median_before": None,
            "target": None,
            "gain": 1.0,
        }

    # Doors get only a restrained multiplicative correction.
    door_pixels = door_soft > 0.5
    if np.any(door_pixels):
        door_before = float(np.median(y[door_pixels]))
        door_target = float(tones["door"]["target"])
        door_gain = float(
            np.clip(
                door_target / max(door_before, 0.08),
                1.0,
                1.25,
            )
        )
        source_log_gain = np.maximum(
            source_log_gain,
            float(np.log(door_gain))
            * (door_soft > 0.35).astype(np.float32),
        )
        per_class["door"] = {
            "median_before": door_before,
            "target": door_target,
            "gain": door_gain,
        }
    else:
        per_class["door"] = {
            "median_before": None,
            "target": None,
            "gain": 1.0,
        }

    # Edge-aware propagation of room illumination.
    propagation_radius = max(16, round(max_side / 100))
    propagated, propagation_mode = _guided_smooth(
        guide_rgb=rgb,
        field=source_log_gain,
        radius=propagation_radius,
        eps=8e-4,
    )
    propagated = np.clip(propagated, 0.0, np.log(2.10))

    # Keep direct structure correction on wall/ceiling/door only.
    direct_structure = np.maximum.reduce([
        (wall_soft > 0.20).astype(np.float32),
        (ceiling_soft > 0.20).astype(np.float32),
        (door_soft > 0.20).astype(np.float32),
    ])
    direct_log_gain = propagated * direct_structure

    # Materials inherit only weak ambient illumination and never direct lift.
    inherit_factor = float(
        np.clip(analysis.get("material_inherit_factor", 0.08), 0.0, 0.12)
    )
    inherited_log_gain = propagated * inherit_factor * material_lock

    # Hard caps prevent floor/cabinet/furniture over-brightening.
    max_material_gain = 1.08
    inherited_log_gain = np.minimum(
        inherited_log_gain,
        np.log(max_material_gain),
    )

    final_log_gain = np.maximum(
        direct_log_gain * (1.0 - material_lock),
        inherited_log_gain,
    )
    final_log_gain *= 1.0 - exclusion_hard

    # Protect true blacks and bright highlights without changing hue.
    black_anchor = _smoothstep(0.018, 0.10, y)
    highlight_guard = 1.0 - _smoothstep(0.72, 0.95, y)
    final_log_gain *= black_anchor * highlight_guard

    out = _apply_multiplicative_gain(rgb, final_log_gain)
    out_y = _luminance(out)

    wall_gain_values = np.exp(final_log_gain[wall_soft > 0.5])
    floor_gain_values = np.exp(final_log_gain[floor_hard > 0.5])
    material_gain_values = np.exp(final_log_gain[material_lock > 0.5])

    return out, {
        "route": "SEMANTIC_CHECKPOINT_A",
        "method": "multiplicative_linear_rgb_log_illumination",
        "per_class": per_class,
        "propagation_mode": propagation_mode,
        "propagation_radius": int(propagation_radius),
        "material_inherit_factor": inherit_factor,
        "max_material_gain": max_material_gain,
        "chroma_compensation": "retired",
        "wall_gain_mean": (
            float(np.mean(wall_gain_values))
            if wall_gain_values.size
            else 1.0
        ),
        "wall_gain_std": (
            float(np.std(wall_gain_values))
            if wall_gain_values.size
            else 0.0
        ),
        "floor_gain_mean": (
            float(np.mean(floor_gain_values))
            if floor_gain_values.size
            else 1.0
        ),
        "floor_gain_max": (
            float(np.max(floor_gain_values))
            if floor_gain_values.size
            else 1.0
        ),
        "protected_gain_mean": (
            float(np.mean(material_gain_values))
            if material_gain_values.size
            else 1.0
        ),
        "protected_gain_max": (
            float(np.max(material_gain_values))
            if material_gain_values.size
            else 1.0
        ),
        "floor_luminance_before": (
            float(np.mean(y[floor_hard > 0.5]))
            if np.any(floor_hard > 0.5)
            else None
        ),
        "floor_luminance_after": (
            float(np.mean(out_y[floor_hard > 0.5]))
            if np.any(floor_hard > 0.5)
            else None
        ),
    }


def global_safe_exposure(
    rgb: np.ndarray,
    analysis: dict,
) -> tuple[np.ndarray, dict]:
    """
    Conservative fallback route.

    Kept deliberately restrained during Checkpoint A. It now uses a scalar
    multiplicative illumination gain instead of additive LAB/chroma boosting.
    """
    y = _luminance(rgb)
    local = cv2.GaussianBlur(y, (0, 0), 36.0)

    dark_region = 1.0 - _smoothstep(0.22, 0.58, local)
    shadow_pixels = 1.0 - _smoothstep(0.16, 0.56, y)
    mid_pixels = np.clip(
        1.0 - np.abs(y - 0.46) / 0.34,
        0.0,
        1.0,
    )

    shadow_lift = float(analysis["shadow_lift"]) * 0.35
    midtone_lift = float(analysis["midtone_lift"]) * 0.35

    requested = (
        shadow_lift * (0.55 * shadow_pixels + 0.45 * dark_region)
        + midtone_lift * (0.55 * mid_pixels + 0.45 * dark_region)
    )

    # Convert the old requested lift into a bounded multiplicative gain.
    gain = 1.0 + np.clip(requested * 1.10, 0.0, 0.22)
    log_gain = np.log(gain)

    black_anchor = _smoothstep(0.018, 0.10, y)
    highlight_guard = 1.0 - _smoothstep(0.72, 0.95, y)
    log_gain *= black_anchor * highlight_guard

    out = _apply_multiplicative_gain(rgb, log_gain)

    return out, {
        "route": "GLOBAL_SAFE_CHECKPOINT_A",
        "method": "multiplicative_linear_rgb",
        "shadow_lift_input": shadow_lift,
        "midtone_lift_input": midtone_lift,
        "gain_mean": float(np.mean(np.exp(log_gain))),
        "gain_max": float(np.max(np.exp(log_gain))),
        "chroma_compensation": "retired",
    }
