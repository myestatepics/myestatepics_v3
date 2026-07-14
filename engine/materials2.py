from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .scene import Scene


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luma(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _mask(scene: Scene, name: str, shape: tuple[int, int]) -> np.ndarray:
    value = scene.masks.get(name)
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return np.clip(value.astype(np.float32), 0.0, 1.0)


def apply_material_guardrails(
    reference: np.ndarray,
    corrected: np.ndarray,
    scene: Scene,
    settings: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Protect photographed materials after global exposure correction.

    ``reference`` should be the white-balanced image. The guardrail never
    reintroduces the original camera cast. It limits excessive brightening of
    intentionally dark surfaces and restores bounded material chroma without
    changing geometry or creating local enhancement targets.
    """
    if reference.shape != corrected.shape:
        raise ValueError("reference and corrected images must have the same shape")
    if reference.dtype != np.uint8 or corrected.dtype != np.uint8:
        raise TypeError("reference and corrected images must be uint8 RGB arrays")

    cfg = (settings or {}).get("material_guardrails", settings or {})
    dark_wall_max_lift = float(cfg.get("dark_wall_max_lift", 0.055))
    dark_material_max_lift = float(cfg.get("dark_material_max_lift", 0.075))
    chroma_restore = float(cfg.get("chroma_restore_strength", 0.50))
    ceiling_chroma_limit = float(cfg.get("ceiling_chroma_limit", 3.0))

    h, w = reference.shape[:2]
    shape = (h, w)
    wall = _mask(scene, "wall", shape)
    floor = _mask(scene, "floor", shape)
    ceiling = _mask(scene, "ceiling", shape)
    cabinet = _mask(scene, "cabinet", shape)
    furnishings = _mask(scene, "furnishings", shape)
    protected = _mask(scene, "protected", shape)

    ref_y = _luma(reference)
    out_y = _luma(corrected)

    # Confidence ramps: darkest surfaces get the strongest luminance hold.
    very_dark = 1.0 - _smoothstep(0.12, 0.38, ref_y)
    dark = 1.0 - _smoothstep(0.16, 0.42, ref_y)
    wall_hold = wall * very_dark
    nonfloor_material = np.maximum.reduce([cabinet, furnishings, protected])
    material_hold = nonfloor_material * dark
    # Floors follow the global room exposure. There is deliberately no
    # floor-specific luminance cap, ratio cap, or chroma re-anchoring.
    floor_hold = np.zeros_like(floor)

    active_hold = np.maximum.reduce([wall_hold, material_hold, floor_hold])
    lift_limit = np.full(shape, dark_material_max_lift, dtype=np.float32)
    lift_limit = lift_limit * (1.0 - wall_hold) + dark_wall_max_lift * wall_hold
    allowed = ref_y + lift_limit
    excess = np.maximum(out_y - allowed, 0.0)
    # Semantic masks are already edge-refined. Apply the limit fully in the
    # confident interior of a mask and feather only uncertain boundary pixels.
    # Keep the hold continuous through mask confidence.  The previous branch
    # snapped to full strength at 0.5 and could print a hard semantic edge.
    hold_strength = _smoothstep(0.05, 0.55, active_hold)
    target_y = out_y - excess * hold_strength

    corrected_f = corrected.astype(np.float32) / 255.0
    ratio = target_y / np.maximum(out_y, 1e-5)
    luminance_limited = np.clip(corrected_f * ratio[..., None], 0.0, 1.0)
    luminance_limited_u8 = np.clip(luminance_limited * 255.0 + 0.5, 0, 255).astype(np.uint8)

    src_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(luminance_limited_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    before_ab = dst_lab[..., 1:3].copy()

    # Do not force chroma into near-black pixels where hue is unstable.
    visible_guard = _smoothstep(0.045, 0.12, ref_y)
    base_restore_mask = np.maximum.reduce([protected, cabinet, furnishings])
    restore_strength = chroma_restore * base_restore_mask * visible_guard
    dst_lab[..., 1:3] = (
        dst_lab[..., 1:3] * (1.0 - restore_strength[..., None])
        + src_lab[..., 1:3] * restore_strength[..., None]
    )

    # Neutral ceilings must not acquire new pink/green/orange contamination.
    src_ab = src_lab[..., 1:3] - 128.0
    dst_ab = dst_lab[..., 1:3] - 128.0
    src_chroma = np.sqrt(np.sum(src_ab * src_ab, axis=2))
    dst_chroma = np.sqrt(np.sum(dst_ab * dst_ab, axis=2))
    neutral_ceiling = ceiling * (src_chroma < 12.0).astype(np.float32)
    ceiling_limit = src_chroma + ceiling_chroma_limit
    scale = np.minimum(1.0, ceiling_limit / np.maximum(dst_chroma, 1e-5))
    scale = 1.0 - neutral_ceiling + neutral_ceiling * scale
    dst_lab[..., 1:3] = 128.0 + (dst_lab[..., 1:3] - 128.0) * scale[..., None]

    out = cv2.cvtColor(np.clip(dst_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    chroma_delta = np.sqrt(np.sum((dst_lab[..., 1:3] - before_ab) ** 2, axis=2))
    held = active_hold > 0.10

    return out, {
        "engine": "material_dark_surface_guardrails_v2",
        "dark_wall_max_lift": dark_wall_max_lift,
        "dark_material_max_lift": dark_material_max_lift,
        "chroma_restore_strength": chroma_restore,
        "held_area_percent": float(np.mean(held) * 100.0),
        "mean_luminance_reduction_held": float(np.mean(excess[held])) if np.any(held) else 0.0,
        "max_luminance_reduction": float(np.max(excess * active_hold)),
        "mean_chroma_adjustment": float(np.mean(chroma_delta[restore_strength > 0.05]))
        if np.any(restore_strength > 0.05) else 0.0,
        "neutral_ceiling_percent": float(np.mean(neutral_ceiling > 0.5) * 100.0),
        "floor_included": False,
        "floor_mode": "global_luminance_only_no_local_guardrail",
    }


def restore_protected_chroma(
    original: np.ndarray,
    corrected: np.ndarray,
    scene: Scene,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility wrapper for legacy callers.

    The MVP uses :func:`apply_material_guardrails` directly. Existing callers
    retain their old function name but receive the safer bounded behavior.
    """
    return apply_material_guardrails(original, corrected, scene, None)
