
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _edge_aware_smooth(
    rgb: np.ndarray,
    gain_map: np.ndarray,
    radius: int,
    max_gain: float = 2.5,
) -> tuple[np.ndarray, str]:
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        smoothed = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=gain_map.astype(np.float32),
            radius=max(8, int(radius)),
            eps=1e-3,
        )
        return np.clip(smoothed, 0.0, max_gain), "guided_filter"

    smoothed = cv2.bilateralFilter(
        gain_map.astype(np.float32),
        d=9,
        sigmaColor=0.08,
        sigmaSpace=max(16, int(radius)),
    )
    return np.clip(smoothed, 0.0, max_gain), "bilateral_fallback"


def _apply_log_gain(rgb: np.ndarray, log_gain: np.ndarray) -> np.ndarray:
    """
    v3.2 Fix 2 (Phase 3 Q4/Q5, implemented for real):
    exposure is a multiplicative gain on luminance, applied as a per-pixel
    ratio to RGB. R:G:B ratios are untouched by construction, so material
    color (wood hue, saturation, grain relationships) is preserved with no
    chroma-compensation hack. Reflectance detail is preserved because every
    pixel in a region is scaled by the same smooth illumination gain.
    """
    f = rgb.astype(np.float32) / 255.0
    ratio = np.exp(log_gain.astype(np.float32))[..., None]
    out = f * ratio
    # Soft highlight shoulder: compress only what would clip, per pixel,
    # preserving ratios by scaling the whole pixel.
    peak = out.max(axis=2, keepdims=True)
    over = peak > 1.0
    if np.any(over):
        scale = np.where(over, 1.0 / np.maximum(peak, 1e-6), 1.0)
        out = out * scale
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _component_log_gain(
    y: np.ndarray,
    mask: np.ndarray,
    target: float,
    max_linear_lift: float,
    max_ratio: float = 1.9,
) -> tuple[np.ndarray, float, float]:
    """
    v3.2 Fix 3: constant lift per component computed from the component
    MEDIAN (spec behavior), never per-pixel push toward the target
    (which is a tone compressor that irons out natural wall shading).
    Returned as a log-luminance gain painted uniformly over the mask.
    """
    sel = mask > 0.5
    if not np.any(sel):
        return np.zeros_like(y, dtype=np.float32), 0.0, 0.0
    median = float(np.median(y[sel]))
    if median <= 1e-4:
        return np.zeros_like(y, dtype=np.float32), median, 0.0
    desired = min(max(target, median), median + max_linear_lift)
    # v3.3: cap the RATIO, not only the linear lift. A "+0.12" on a black
    # surface is a 3.5x multiplier that turns shadow noise and faint tints
    # into visible color. Dark surfaces get gentle multipliers by design.
    log_gain_value = float(np.log(max(desired, 1e-4) / median))
    log_gain_value = min(max(0.0, log_gain_value), float(np.log(max_ratio)))
    gain = np.zeros_like(y, dtype=np.float32)
    gain[sel] = log_gain_value
    return gain, median, log_gain_value


def _adaptive_ratio_cap(
    median: float,
    base_cap: float,
    *,
    surface: str = "wall",
) -> float:
    """Return a luminance-aware exposure cap.

    Near-black surfaces remain tightly protected because large multipliers
    reveal chroma noise and destroy intentional black paint. Medium-dark
    painted surfaces are allowed substantially more reach so dim rooms can
    converge toward MLS brightness instead of stopping at a fixed cap.
    """
    if median <= 0.0:
        return float(base_cap)

    if surface == "ceiling":
        points = (
            (0.08, 1.45),
            (0.18, 1.95),
            (0.32, 2.45),
            (0.48, 2.75),
        )
    else:
        points = (
            (0.08, 1.30),
            (0.18, 1.70),
            (0.32, 2.20),
            (0.48, 2.55),
        )

    adaptive = points[-1][1]
    for threshold, cap in points:
        if median <= threshold:
            adaptive = cap
            break
    return float(max(base_cap, adaptive))


def _convergence_log_gain(
    y_before: np.ndarray,
    y_after: np.ndarray,
    target: float,
    reference_mask: np.ndarray,
    max_ratio: float,
) -> tuple[np.ndarray, dict]:
    """Create one bounded corrective pass from achieved luminance.

    This is deliberately a single pass, not iterative tone mapping. It closes
    the residual caused by edge smoothing and safety guards while retaining
    black and highlight protection.
    """
    sel = reference_mask.astype(bool)
    if np.count_nonzero(sel) < 256:
        return np.zeros_like(y_before, np.float32), {
            "applied": False,
            "reason": "insufficient_reference",
        }

    achieved = float(np.median(y_after[sel]))
    if achieved <= 1e-4 or achieved >= target * 0.97:
        return np.zeros_like(y_before, np.float32), {
            "applied": False,
            "achieved": achieved,
            "target": float(target),
        }

    ratio = float(np.clip(target / achieved, 1.0, max_ratio))
    residual = float(np.log(ratio))

    black_anchor = _smoothstep(0.035, 0.16, y_before)
    highlight_guard = 1.0 - _smoothstep(0.76, 0.96, y_before)
    shadow_mid_weight = 1.0 - 0.35 * _smoothstep(0.48, 0.78, y_before)
    gain = residual * black_anchor * highlight_guard * shadow_mid_weight

    return gain.astype(np.float32), {
        "applied": True,
        "achieved": achieved,
        "target": float(target),
        "ratio": ratio,
    }


def _damp_chroma_growth(
    original: np.ndarray,
    lifted: np.ndarray,
    log_gain: np.ndarray,
    y_original: np.ndarray,
    material_mask: np.ndarray | None = None,
    neutralize_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    v3.3: multiplicative lifting preserves R:G:B ratios, which also scales
    up the visible saturation of dark tints and shadow noise. Like a colorist
    lifting shadows, we allow saturation to grow only ~35% of the way the
    lift would take it (15% for near-black pixels, where color is mostly
    noise). Hue is untouched: a and b are damped by the same factor.
    """
    ratio = np.exp(np.clip(log_gain, 0.0, None))
    grew = ratio > 1.02
    if not np.any(grew):
        return lifted
    src = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB).astype(np.float32)
    c_src = np.sqrt((src[..., 1] - 128.0) ** 2 + (src[..., 2] - 128.0) ** 2)
    c_dst = np.sqrt((dst[..., 1] - 128.0) ** 2 + (dst[..., 2] - 128.0) ** 2)
    # Above L 0.10: saturation may grow ~35% of the lift. Below L 0.10 the
    # colorist rule inverts: lifted blacks are gently DESATURATED so black
    # paint keeps reading as black instead of blooming pink/red from noise.
    growth_share = np.where(y_original < 0.10, -0.50, 0.35).astype(np.float32)
    if material_mask is not None:
        # Materials (wood, stone, fabric): photographed saturation HOLDS as
        # luminance rises -- near-zero growth, or floors run "fire-red".
        growth_share = np.where(material_mask > 0.5, 0.10, growth_share)
    if neutralize_mask is not None:
        # Ceilings are semantically near-neutral paint. Any tint they carry
        # is color bounce = illumination; brightening must CLEAN it, not
        # amplify it: chroma shrinks in proportion to the lift.
        growth_share = np.where(neutralize_mask > 0.5, -0.60, growth_share)
    allowed = c_src * np.clip(1.0 + growth_share * (ratio - 1.0), 0.40, None) + 1e-3
    damp = np.clip(allowed / np.maximum(c_dst, 1e-3), 0.0, 1.0)
    damp = np.where(grew, damp, 1.0)
    dst[..., 1:3] = 128.0 + (dst[..., 1:3] - 128.0) * damp[..., None]
    converted = cv2.cvtColor(np.clip(dst, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    # Only rewrite pixels that were actually damped; everything else keeps the
    # exact lifted RGB (avoids LAB round-trip quantization on untouched pixels).
    needs = (damp < 0.995)[..., None]
    return np.where(needs, converted, lifted)



def _lock_lab_chroma(original: np.ndarray, lifted: np.ndarray) -> np.ndarray:
    """Preserve photographed Lab chroma while accepting lifted luminance.

    GLOBAL_SAFE has no reliable semantic material masks. Reusing the lifted
    L channel with the original a/b channels prevents exposure from warming
    floors, cabinets, walls, or ceilings. Geometry is unchanged.
    """
    src = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst[..., 1] = src[..., 1]
    dst[..., 2] = src[..., 2]
    return cv2.cvtColor(np.clip(dst, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def adaptive_per_class_exposure(
    rgb: np.ndarray,
    scene: Scene,
    tones: dict,
    analysis: dict,
) -> tuple[np.ndarray, dict]:
    """
    Phase 3 rules (v3.2 -- fully implemented):
    - Walls, ceilings and doors drive room brightness (illumination classes).
    - Materials (floor, cabinets, furniture, rugs...) have no targets and
      inherit the FULL room illumination field (Fix 4). The field itself is
      the natural amount of light they should receive.
    - Lifts are per-component constants from medians (Fix 3), propagated
      edge-aware so light stops at material and paint boundaries.
    - Correction is multiplicative in luminance (Fix 2): "correct the light,
      never the materials."
    """
    y = _luminance(rgb)
    h, w = y.shape
    structure_gain = np.zeros_like(y, dtype=np.float32)
    per_class: dict[str, dict] = {}

    # WALLS: per-component classification from tone_classify. The target map
    # is constant within each component, so median-based gain per component
    # is recovered by grouping on the map values within the wall mask.
    wall_mask = scene.masks["wall"]
    wall_target_map = tones["wall_target_map"]
    wall_sel = wall_mask > 0.5
    wall_median = float(np.median(y[wall_sel])) if np.any(wall_sel) else None
    wall_gains = []
    if np.any(wall_sel):
        for target_value in np.unique(wall_target_map[wall_sel]):
            if target_value <= 0:
                continue
            comp_mask = (np.abs(wall_target_map - target_value) < 1e-4) & wall_sel
            codes = tones["wall_class_map"][comp_mask]
            code = int(np.median(codes)) if codes.size else 2
            default_caps = {1: 1.30, 2: 1.55, 3: 1.90}
            caps = {**default_caps, **{int(k): float(v) for k, v in analysis.get("wall_ratio_caps", {}).items()}}
            base_ratio_cap = caps.get(code, 1.55)
            comp_values = y[comp_mask]
            estimated_median = float(np.median(comp_values)) if comp_values.size else 0.0
            ratio_cap = _adaptive_ratio_cap(
                estimated_median, base_ratio_cap, surface="wall"
            )
            gain, comp_median, gval = _component_log_gain(
                y, comp_mask.astype(np.float32), float(target_value), 0.45,
                max_ratio=ratio_cap,
            )
            structure_gain = np.maximum(structure_gain, gain)
            eff = float(comp_median * np.exp(gval)) if comp_median else float(target_value)
            wall_effective = tones.setdefault("wall_effective_targets", [])
            wall_effective.append({"raw": float(target_value), "effective": eff})
            wall_gains.append({
                "target": float(target_value),
                "effective_target": eff,
                "median": comp_median,
                "log_gain": gval,
            })
    per_class["wall"] = {"median_before": wall_median, "components": wall_gains}

    # CEILING
    ceiling_mask = scene.masks["ceiling"]
    ceiling_target = float(tones["ceiling"]["target"])
    ceiling_values = y[ceiling_mask > 0.5]
    estimated_ceiling_median = (
        float(np.median(ceiling_values)) if ceiling_values.size else 0.0
    )
    ceiling_ratio_cap = _adaptive_ratio_cap(
        estimated_ceiling_median,
        float(analysis.get("ceiling_ratio_cap", 1.90)),
        surface="ceiling",
    )
    ceiling_gain, ceiling_median, ceiling_gval = _component_log_gain(
        y, ceiling_mask, ceiling_target, 0.52,
        max_ratio=ceiling_ratio_cap,
    )
    structure_gain = np.maximum(structure_gain, ceiling_gain)
    tones["ceiling"]["effective_target"] = (
        float(ceiling_median * np.exp(ceiling_gval)) if ceiling_median else ceiling_target
    )
    per_class["ceiling"] = {
        "median_before": ceiling_median if np.any(ceiling_mask > 0.5) else None,
        "target": ceiling_target,
        "effective_target": tones["ceiling"]["effective_target"],
        "log_gain": ceiling_gval,
    }

    # DOOR: small direct lift only.
    door_mask = scene.masks["door"]
    door_target = float(tones["door"]["target"])
    door_gain, door_median, door_gval = _component_log_gain(
        y, door_mask, door_target, 0.10, max_ratio=1.30
    )
    structure_gain = np.maximum(structure_gain, door_gain)
    per_class["door"] = {
        "median_before": door_median if np.any(door_mask > 0.5) else None,
        "target": door_target,
        "log_gain": door_gval,
    }

    # FLOOR: protected -- no direct target (Phase 3 Q2).
    floor_mask = scene.masks["floor"]
    per_class["floor"] = {
        "mode": "protected_no_direct_target",
        "median_before": (
            float(np.median(y[floor_mask > 0.5]))
            if np.any(floor_mask > 0.5)
            else None
        ),
        "direct_lift": 0.0,
    }

    # Edge-aware propagation: light stops at boundaries.
    radius = max(16, round(max(h, w) / 110))
    smoothed_structure, propagation_mode = _edge_aware_smooth(rgb, structure_gain, radius)

    # Broad ambient illumination field for materials. Deliberately DIFFUSE
    # (large Gaussian), not edge-aware: ambient room light crosses material
    # boundaries by nature. Edge-awareness is applied to the STRUCTURE gain
    # above (so lifts stop at paint/material edges); the ambient field is the
    # smooth average of that corrected light, delivered to materials.
    field_sigma = max(24.0, max(h, w) / 12.0)
    local_field = cv2.GaussianBlur(smoothed_structure, (0, 0), field_sigma)
    # Room-level ambient term: image-space distance is the wrong model for
    # ambient light -- a floor three metres from a wall still receives the
    # room's light. The scalar is the mean illumination correction over the
    # structure surfaces; bright rooms (near-zero gains) contribute ~0, so
    # already-good photos stay untouched.
    # v3.3: only illumination-credible surfaces vote on the room's ambient
    # light. A black accent wall wanting a whisper of lift says nothing about
    # how much light the ROOM needs -- excluding it stops dark-gain leakage
    # from over-lighting floors. Ambient itself is capped at 1.5x.
    structure_sel = (scene.masks["structure"] > 0.5) & (y >= 0.18)
    ambient_scalar = (
        float(np.mean(smoothed_structure[structure_sel]))
        if np.count_nonzero(structure_sel) > 400
        else 0.0
    )
    ambient_scalar = min(
        ambient_scalar,
        float(np.log(float(analysis.get("ambient_cap_ratio", 1.5)))),
    )
    illumination_field = np.maximum(local_field, ambient_scalar)
    field_mode = f"gaussian_sigma_{field_sigma:.0f}_plus_ambient_{ambient_scalar:.3f}"

    protected = scene.masks["protected"]
    floor = scene.masks["floor"]
    protected_all = np.clip(np.maximum(protected, floor), 0.0, 1.0)

    # v3.2 Fix 4: materials inherit the FULL illumination field. The field is
    # already the correct ambient quantity; attenuating it (the old 8%) starves
    # 30-50% of the frame of light and is the root of chronic underexposure.
    material_inherit_factor = float(analysis.get("material_inherit_factor", 1.0))
    inherited_material_gain = material_inherit_factor * illumination_field * protected_all

    final_gain = np.maximum(
        smoothed_structure * (1.0 - protected_all),
        inherited_material_gain,
    )

    exclusion = np.maximum.reduce([
        scene.masks["window"],
        scene.masks["curtain"],
        scene.masks["mirror"],
    ])
    final_gain *= 1.0 - np.clip(exclusion, 0.0, 1.0)

    # Guards operate on original luminance: hold true blacks, protect highlights.
    black_anchor = _smoothstep(0.025, 0.13, y)
    highlight_guard = 1.0 - _smoothstep(0.68, 0.92, y)
    final_gain = final_gain * black_anchor * highlight_guard

    out = _apply_log_gain(rgb, final_gain)
    y_first = _luminance(out)

    structure_reference = (
        (scene.masks["wall"] > 0.5)
        | (scene.masks["ceiling"] > 0.5)
    ) & (y > 0.10) & (y < 0.82)
    raw_targets = []
    if wall_gains:
        raw_targets.extend(float(item["target"]) for item in wall_gains)
    if ceiling_target > 0:
        raw_targets.append(ceiling_target)
    convergence_target = (
        float(np.median(raw_targets)) if raw_targets else 0.50
    )
    correction_gain, convergence_log = _convergence_log_gain(
        y, y_first, convergence_target, structure_reference, max_ratio=1.28
    )
    correction_gain *= 1.0 - np.clip(exclusion, 0.0, 1.0)
    total_gain = final_gain + correction_gain

    out = _apply_log_gain(rgb, total_gain)
    out = _damp_chroma_growth(
        rgb, out, total_gain, y,
        material_mask=protected_all,
        neutralize_mask=scene.masks["ceiling"],
    )
    y_after = _luminance(out)

    return out, {
        "route": "SEMANTIC_PHASE3",
        "model": "log_luminance_multiplicative_v3.2",
        "per_class": per_class,
        "propagation_mode": propagation_mode,
        "field_mode": field_mode,
        "convergence": convergence_log,
        "propagation_radius": int(radius),
        "field_sigma": float(field_sigma),
        "material_inherit_factor": material_inherit_factor,
        "mean_material_inherited_gain": (
            float(np.mean(inherited_material_gain[protected_all > 0.5]))
            if np.any(protected_all > 0.5)
            else 0.0
        ),
        "max_floor_lift": (
            float(np.max(y_after[floor > 0.5] - y[floor > 0.5]))
            if np.any(floor > 0.5)
            else 0.0
        ),
        "chroma_boost_max": 1.0,  # retired: multiplicative model needs none
        "mean_image_luminance_before": float(np.mean(y)),
        "mean_image_luminance_after": float(np.mean(y_after)),
    }


def global_safe_exposure(
    rgb: np.ndarray,
    analysis: dict,
) -> tuple[np.ndarray, dict]:
    """
    Geometry-safe fallback exposure with adaptive reach and one convergence
    pass. It remains conservative on true blacks and highlights, but no longer
    starves medium-dark rooms merely because semantic routing was unavailable.
    """
    y = _luminance(rgb)
    local = cv2.GaussianBlur(y, (0, 0), 36.0)

    dark_region = 1.0 - _smoothstep(0.22, 0.62, local)
    shadow_pixels = 1.0 - _smoothstep(0.14, 0.58, y)
    mid_pixels = np.clip(1.0 - np.abs(y - 0.46) / 0.38, 0.0, 1.0)

    # Preserve true blacks while allowing dark wood/floors and dim painted
    # surfaces to receive useful MLS lift. The old 0.16 shoulder starved
    # legitimate dark materials; this tighter anchor still holds pixels near
    # black while releasing usable shadow detail sooner.
    black_anchor = _smoothstep(0.020, 0.120, y)
    highlight_guard = 1.0 - _smoothstep(0.78, 0.96, y)

    shadow_lift = float(analysis["shadow_lift"]) * 0.92
    midtone_lift = float(analysis["midtone_lift"]) * 0.86

    linear_lift = (
        shadow_lift * (0.58 * shadow_pixels + 0.42 * dark_region)
        + midtone_lift * (0.60 * mid_pixels + 0.40 * dark_region)
    ) * black_anchor * highlight_guard

    profile = str(analysis.get("room_profile", "generic"))
    first_pass_cap = {
        "dark_room": 1.85,
        "low_light": 2.30,
        "bathroom": 1.90,
        "kitchen": 1.95,
        "living": 1.95,
        "bedroom": 1.90,
        "generic": 1.95,
    }.get(profile, 1.80)

    log_gain = np.log(
        np.clip(
            (y + linear_lift) / np.maximum(y, 1e-4),
            1.0,
            first_pass_cap,
        )
    )
    first = _apply_log_gain(rgb, log_gain)
    y_first = _luminance(first)

    target = {
        "dark_room": 0.50,
        "low_light": 0.54,
        "bathroom": 0.55,
        "kitchen": 0.56,
        "living": 0.55,
        "bedroom": 0.54,
        "generic": 0.55,
    }.get(profile, 0.50)

    h = y.shape[0]
    if profile == "dark_room":
        # In deliberate dark decor, use the upper half's brighter painted
        # surfaces as the exposure witness so black walls remain black.
        spatial = np.zeros_like(y, dtype=bool)
        spatial[: max(1, int(h * 0.58)), :] = True
        reference = spatial & (y > 0.13) & (y < 0.82)
        convergence_cap = 1.50
    else:
        reference = (y > 0.10) & (y < 0.84)
        convergence_cap = 1.45

    correction_gain, convergence_log = _convergence_log_gain(
        y, y_first, target, reference, max_ratio=convergence_cap
    )
    total_gain = log_gain + correction_gain
    out = _apply_log_gain(rgb, total_gain)
    # GLOBAL_SAFE has no semantic material masks. Treat the whole image as
    # color-protected during exposure so added light cannot make hardwood,
    # cabinets, or painted surfaces run warmer/more saturated. This does not
    # alter the incoming WB; it only prevents exposure from amplifying chroma.
    out = _lock_lab_chroma(rgb, out)

    return out, {
        "route": "GLOBAL_SAFE",
        "model": "adaptive_log_luminance_convergence_v4.1.1",
        "room_profile": profile,
        "shadow_lift": shadow_lift,
        "midtone_lift": midtone_lift,
        "first_pass_cap": first_pass_cap,
        "convergence": convergence_log,
        "chroma_boost_max": 1.0,
    }
