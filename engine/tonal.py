
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lab_l(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab, lab[..., 0] / 255.0


def _edge_aware_smooth(
    rgb: np.ndarray,
    lift_map: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, str]:
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        smoothed = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=lift_map.astype(np.float32),
            radius=max(8, int(radius)),
            eps=1e-3,
        )
        return np.clip(smoothed, 0.0, 1.0), "guided_filter"

    smoothed = cv2.bilateralFilter(
        lift_map.astype(np.float32),
        d=9,
        sigmaColor=0.08,
        sigmaSpace=max(16, int(radius)),
    )
    return np.clip(smoothed, 0.0, 1.0), "bilateral_fallback"


def adaptive_per_class_exposure(
    rgb: np.ndarray,
    scene: Scene,
    tones: dict,
    analysis: dict,
) -> tuple[np.ndarray, dict]:
    """
    Phase 3 rules:
    - Walls and ceilings drive room brightness.
    - Floors, cabinets, furniture, rugs, mirrors, art and plants are protected.
    - Protected materials receive only a small inherited ambient lift.
    - Edge-aware smoothing prevents lift from bleeding across dark wall boundaries.
    """
    lab, l = _lab_l(rgb)
    h, w = l.shape
    structure_lift = np.zeros_like(l, dtype=np.float32)
    per_class: dict[str, dict] = {}

    # WALLS: adaptive per-pixel targets from tone_classify.
    wall_mask = scene.masks["wall"]
    wall_target = tones["wall_target_map"]
    wall_lift = np.clip(wall_target - l, 0.0, 0.35) * wall_mask
    structure_lift = np.maximum(structure_lift, wall_lift)

    per_class["wall"] = {
        "median_before": (
            float(np.median(l[wall_mask > 0.5]))
            if np.any(wall_mask > 0.5)
            else None
        ),
        "mean_lift": (
            float(np.mean(wall_lift[wall_mask > 0.5]))
            if np.any(wall_mask > 0.5)
            else 0.0
        ),
    }

    # CEILING: still allowed to drive brightness.
    ceiling_mask = scene.masks["ceiling"]
    if np.any(ceiling_mask > 0.5):
        ceiling_median = float(np.median(l[ceiling_mask > 0.5]))
        ceiling_target = float(tones["ceiling"]["target"])
        ceiling_lift = np.clip(
            ceiling_target - l,
            0.0,
            0.24,
        ) * ceiling_mask
        structure_lift = np.maximum(structure_lift, ceiling_lift)
        per_class["ceiling"] = {
            "median_before": ceiling_median,
            "target": ceiling_target,
            "mean_lift": float(np.mean(ceiling_lift[ceiling_mask > 0.5])),
        }
    else:
        per_class["ceiling"] = {
            "median_before": None,
            "target": 0.0,
            "mean_lift": 0.0,
        }

    # DOOR: tiny direct lift only.
    door_mask = scene.masks["door"]
    if np.any(door_mask > 0.5):
        door_median = float(np.median(l[door_mask > 0.5]))
        door_target = float(tones["door"]["target"])
        door_lift = np.clip(door_target - l, 0.0, 0.08) * door_mask
        structure_lift = np.maximum(structure_lift, door_lift)
        per_class["door"] = {
            "median_before": door_median,
            "target": door_target,
            "mean_lift": float(np.mean(door_lift[door_mask > 0.5])),
        }
    else:
        per_class["door"] = {
            "median_before": None,
            "target": 0.0,
            "mean_lift": 0.0,
        }

    # FLOOR: no independent target in Phase 3.
    floor_mask = scene.masks["floor"]
    per_class["floor"] = {
        "mode": "protected_no_direct_target",
        "median_before": (
            float(np.median(l[floor_mask > 0.5]))
            if np.any(floor_mask > 0.5)
            else None
        ),
        "direct_lift": 0.0,
    }

    radius = max(16, round(max(h, w) / 110))
    smoothed_structure, propagation_mode = _edge_aware_smooth(
        rgb,
        structure_lift,
        radius,
    )

    # Broad room illumination field, but still edge-aware.
    broad_radius = max(24, round(max(h, w) / 55))
    illumination_field, field_mode = _edge_aware_smooth(
        rgb,
        smoothed_structure,
        broad_radius,
    )

    protected = scene.masks["protected"]
    floor = scene.masks["floor"]
    protected_all = np.maximum(protected, floor)

    # Only 8% inherited ambient lift for materials.
    material_inherit_factor = float(
        analysis.get("material_inherit_factor", 0.08)
    )
    inherited_material_lift = (
        material_inherit_factor
        * illumination_field
        * protected_all
    )

    final_lift = np.maximum(
        smoothed_structure * (1.0 - protected_all),
        inherited_material_lift,
    )

    exclusion = np.maximum.reduce([
        scene.masks["window"],
        scene.masks["curtain"],
        scene.masks["mirror"],
    ])
    final_lift *= 1.0 - np.clip(exclusion, 0.0, 1.0)

    black_anchor = _smoothstep(0.025, 0.13, l)
    highlight_guard = 1.0 - _smoothstep(0.68, 0.92, l)
    out_l = np.clip(
        l + final_lift * black_anchor * highlight_guard,
        0.0,
        1.0,
    )

    actual_lift = out_l - l
    chroma_boost = 1.0 + np.clip(
        actual_lift * 0.35 / 0.30,
        0.0,
        0.10,
    )

    lab[..., 0] = out_l * 255.0
    lab[..., 1] = 128.0 + (lab[..., 1] - 128.0) * chroma_boost
    lab[..., 2] = 128.0 + (lab[..., 2] - 128.0) * chroma_boost

    out = cv2.cvtColor(
        np.clip(lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    )

    return out, {
        "route": "SEMANTIC_PHASE3",
        "per_class": per_class,
        "propagation_mode": propagation_mode,
        "field_mode": field_mode,
        "propagation_radius": int(radius),
        "field_radius": int(broad_radius),
        "material_inherit_factor": material_inherit_factor,
        "mean_material_inherited_lift": (
            float(np.mean(inherited_material_lift[protected_all > 0.5]))
            if np.any(protected_all > 0.5)
            else 0.0
        ),
        "max_floor_lift": (
            float(np.max(actual_lift[floor > 0.5]))
            if np.any(floor > 0.5)
            else 0.0
        ),
        "chroma_boost_max": float(np.max(chroma_boost)),
    }


def global_safe_exposure(
    rgb: np.ndarray,
    analysis: dict,
) -> tuple[np.ndarray, dict]:
    lab, l = _lab_l(rgb)
    local = cv2.GaussianBlur(l, (0, 0), 36.0)

    dark_region = 1.0 - _smoothstep(0.22, 0.58, local)
    shadow_pixels = 1.0 - _smoothstep(0.16, 0.56, l)
    mid_pixels = np.clip(
        1.0 - np.abs(l - 0.46) / 0.34,
        0.0,
        1.0,
    )

    black_anchor = _smoothstep(0.025, 0.13, l)
    highlight_guard = 1.0 - _smoothstep(0.68, 0.92, l)

    shadow_lift = float(analysis["shadow_lift"]) * 0.50
    midtone_lift = float(analysis["midtone_lift"]) * 0.50

    lift = (
        shadow_lift * (0.55 * shadow_pixels + 0.45 * dark_region)
        + midtone_lift * (0.55 * mid_pixels + 0.45 * dark_region)
    ) * black_anchor * highlight_guard

    out_l = np.clip(l + lift, 0.0, 1.0)
    actual_lift = out_l - l
    chroma_boost = 1.0 + np.clip(
        actual_lift * 0.35 / 0.30,
        0.0,
        0.10,
    )

    lab[..., 0] = out_l * 255.0
    lab[..., 1] = 128.0 + (lab[..., 1] - 128.0) * chroma_boost
    lab[..., 2] = 128.0 + (lab[..., 2] - 128.0) * chroma_boost

    out = cv2.cvtColor(
        np.clip(lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    )

    return out, {
        "route": "GLOBAL_SAFE",
        "shadow_lift": shadow_lift,
        "midtone_lift": midtone_lift,
        "chroma_boost_max": float(np.max(chroma_boost)),
    }
