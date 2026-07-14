from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .scene import Scene


def _lum(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _uniformity_score(y: np.ndarray, mask: np.ndarray) -> float:
    pixels = y[mask > 0.5]
    if pixels.size < 100:
        return 0.0
    p10, p90 = np.percentile(pixels, [10, 90])
    return float(p90 - p10)


def _safe_mask(scene: Scene, name: str, shape: tuple[int, int]) -> np.ndarray:
    mask = scene.masks.get(name)
    if mask is None:
        return np.zeros(shape, dtype=np.float32)
    return mask.astype(np.float32)


def _hsv_drift(original: np.ndarray, final: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask > 0.5
    if not np.any(valid):
        return 0.0, 0.0
    src = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
    dst = cv2.cvtColor(final, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_delta = np.abs(dst[..., 0] - src[..., 0])
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
    sat_delta = np.abs(dst[..., 1] - src[..., 1]) / 255.0
    return float(np.percentile(hue_delta[valid], 95)), float(np.percentile(sat_delta[valid], 95))


def _texture_ratio(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if np.count_nonzero(valid) < 256:
        return 1.0
    before_detail = cv2.Laplacian(before.astype(np.float32), cv2.CV_32F, ksize=3)
    after_detail = cv2.Laplacian(after.astype(np.float32), cv2.CV_32F, ksize=3)
    # Normalize detail by local brightness so a clean exposure lift is not
    # misclassified as artificial sharpening.
    before_level = float(np.mean(np.abs(before[valid])))
    after_level = float(np.mean(np.abs(after[valid])))
    if before_level < 0.12:
        # Relative detail energy is numerically unstable near black; chroma,
        # blotch, and discontinuity gates cover those surfaces more reliably.
        return 1.0
    before_energy = float(np.mean(np.abs(before_detail[valid]))) / max(before_level, 1e-4)
    after_energy = float(np.mean(np.abs(after_detail[valid]))) / max(after_level, 1e-4)
    return after_energy / max(before_energy, 1e-4)


def _artifact_metrics(
    before: np.ndarray,
    after: np.ndarray,
    masks: list[np.ndarray],
    floor: np.ndarray,
) -> dict[str, float]:
    """Measure correction-field artifacts rather than photographed edges."""
    correction = after - before
    local = cv2.GaussianBlur(correction, (0, 0), 12.0)
    floor_valid = floor > 0.65
    floor_blotch = (
        float(np.percentile(np.abs(local[floor_valid] - np.median(local[floor_valid])), 95))
        if np.count_nonzero(floor_valid) >= 256 else 0.0
    )

    boundary = np.zeros(before.shape, np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8)
        boundary = np.maximum(boundary, cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel))
    gx = cv2.Sobel(correction, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(correction, cv2.CV_32F, 0, 1, ksize=3)
    correction_gradient = np.sqrt(gx * gx + gy * gy)
    hard_boundary = (
        float(np.percentile(correction_gradient[boundary > 0], 95))
        if np.any(boundary > 0) else 0.0
    )
    discontinuity = float(np.percentile(np.abs(cv2.Laplacian(correction, cv2.CV_32F)), 99))
    return {
        "floor_blotch_p95": floor_blotch,
        "mask_boundary_gradient_p95": hard_boundary,
        "local_contrast_discontinuity_p99": discontinuity,
    }


def _wall_chroma_divergence(lab: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.55
    if np.count_nonzero(valid) < 512:
        return 0.0
    ab = lab[..., 1:3] - 128.0
    weight = valid.astype(np.float32)
    h, w = mask.shape
    scale = min(1.0, 640.0 / max(h, w))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    weight = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA)
    ab = cv2.resize(ab, (sw, sh), interpolation=cv2.INTER_AREA)
    valid = weight > 0.55
    sigma = max(12.0, max(sh, sw) / 10.0)
    den = cv2.GaussianBlur(weight, (0, 0), sigma) + 1e-5
    local_a = cv2.GaussianBlur(ab[..., 0] * weight, (0, 0), sigma) / den
    local_b = cv2.GaussianBlur(ab[..., 1] * weight, (0, 0), sigma) / den
    target = np.median(ab[valid], axis=0)
    divergence = np.sqrt((local_a - target[0]) ** 2 + (local_b - target[1]) ** 2)
    return float(np.percentile(divergence[valid], 90))


def evaluate(
    original: np.ndarray,
    final: np.ndarray,
    scene: Scene,
    tones: dict,
    *,
    settings: dict[str, Any] | None = None,
    wb_log: dict[str, Any] | None = None,
    exposure_log: dict[str, Any] | None = None,
    material_reference: np.ndarray | None = None,
    evaluate_semantic_targets: bool = False,
) -> dict:
    """Evaluate final output and return PASS/REVIEW diagnostics.

    Phase 5 adds explicit clipping, brightness, WB-gain, hue/saturation and
    protected-material gates. Thresholds are configurable under
    ``quality_validation`` and default to conservative production values.
    """
    cfg = (settings or {}).get("quality_validation", {})
    limits = {
        "highlight_clip_percent": float(cfg.get("highlight_clip_percent", 1.8)),
        "shadow_clip_percent": float(cfg.get("shadow_clip_percent", 3.0)),
        "global_brightness_delta": float(cfg.get("global_brightness_delta", 0.20)),
        "wb_gain_deviation": float(cfg.get("wb_gain_deviation", 0.18)),
        "protected_chroma_p95": float(cfg.get("protected_chroma_p95", 9.0)),
        "floor_chroma_p95": float(cfg.get("floor_chroma_p95", 7.0)),
        "material_hue_p95": float(cfg.get("material_hue_p95", 8.0)),
        "material_saturation_p95": float(cfg.get("material_saturation_p95", 0.16)),
        "floor_luminance_shift": float(cfg.get("floor_luminance_shift", 0.18)),
        "wall_uniformity_increase": float(cfg.get("wall_uniformity_increase", 0.08)),
        "residual_cast": float(cfg.get("residual_cast", 5.0)),
        "input_material_chroma_p95": float(cfg.get("input_material_chroma_p95", 14.0)),
        "floor_blotch_p95": float(cfg.get("floor_blotch_p95", 0.045)),
        "mask_boundary_gradient_p95": float(cfg.get("mask_boundary_gradient_p95", 1.0)),
        "local_contrast_discontinuity_p99": float(cfg.get("local_contrast_discontinuity_p99", 0.34)),
        "texture_retention_min": float(cfg.get("texture_retention_min", 0.72)),
        "texture_amplification_max": float(cfg.get("texture_amplification_max", 2.20)),
        "wall_chroma_divergence_p90": float(cfg.get("wall_chroma_divergence_p90", 7.0)),
        "ceiling_warmth_b_median": float(cfg.get("ceiling_warmth_b_median", 8.0)),
    }

    reference = material_reference if material_reference is not None else original
    before = _lum(reference)
    after = _lum(final)
    flags: list[str] = []
    achieved: dict[str, Any] = {}
    h, w = before.shape

    wall_mask_f = _safe_mask(scene, "wall", (h, w))
    ceiling_mask_f = _safe_mask(scene, "ceiling", (h, w))
    floor_mask_f = _safe_mask(scene, "floor", (h, w))
    protected_f = _safe_mask(scene, "protected", (h, w))
    window_f = _safe_mask(scene, "window", (h, w))
    mirror_f = _safe_mask(scene, "mirror", (h, w))

    wall_target = tones.get("wall_target_map")
    wall_mask = wall_mask_f > 0.5
    if evaluate_semantic_targets and wall_target is not None and np.any(wall_mask):
        target_median = float(np.median(wall_target[wall_mask]))
        eff_list = tones.get("wall_effective_targets") or []
        if eff_list:
            target_median = min(target_median, max(e["effective"] for e in eff_list))
        achieved_median = float(np.median(after[wall_mask]))
        achieved["wall"] = {"target": target_median, "after": achieved_median}
        if achieved_median < target_median - 0.06:
            flags.append("WALL_TARGET_MISSED")
        if achieved_median > target_median + 0.10:
            flags.append("WALL_OVERSHOT")

    ceiling_mask = ceiling_mask_f > 0.5
    ceiling_tone = tones.get("ceiling", {})
    if evaluate_semantic_targets and np.any(ceiling_mask) and ceiling_tone:
        target = float(ceiling_tone.get("effective_target", ceiling_tone.get("target", 0.0)))
        median_after = float(np.median(after[ceiling_mask]))
        achieved["ceiling"] = {"target": target, "after": median_after}
        if target > 0 and median_after < target - 0.06:
            flags.append("CEILING_TARGET_MISSED")
        if target > 0 and median_after > target + 0.10:
            flags.append("CEILING_OVERSHOT")

    non_window = np.maximum(window_f, mirror_f) < 0.3
    highlight_clipped = float(np.mean((after > 0.985) & non_window) * 100.0)
    shadow_clipped = float(np.mean((after < 0.012) & non_window) * 100.0)
    brightness_before = float(np.median(before[non_window])) if np.any(non_window) else float(np.median(before))
    brightness_after = float(np.median(after[non_window])) if np.any(non_window) else float(np.median(after))
    brightness_delta = brightness_after - brightness_before

    local = cv2.GaussianBlur(after.astype(np.float32), (0, 0), 28.0)
    glare = float(np.mean((after > 0.90) & (local > 0.72) & non_window) * 100.0)

    src_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    input_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(final, cv2.COLOR_RGB2LAB).astype(np.float32)
    delta = np.sqrt(
        (dst_lab[..., 1] - src_lab[..., 1]) ** 2
        + (dst_lab[..., 2] - src_lab[..., 2]) ** 2
    )
    input_delta = np.sqrt(
        (dst_lab[..., 1] - input_lab[..., 1]) ** 2
        + (dst_lab[..., 2] - input_lab[..., 2]) ** 2
    )

    protected_mask = np.maximum(protected_f, floor_mask_f) > 0.5
    protected_p95 = float(np.percentile(delta[protected_mask], 95)) if np.any(protected_mask) else 0.0
    protected_mean = float(np.mean(delta[protected_mask])) if np.any(protected_mask) else 0.0
    input_protected_p95 = float(np.percentile(input_delta[protected_mask], 95)) if np.any(protected_mask) else 0.0

    floor_mask = floor_mask_f > 0.5
    floor_luminance_shift = (
        float(abs(np.median(after[floor_mask]) - np.median(before[floor_mask])))
        if np.any(floor_mask) else 0.0
    )
    floor_chroma_p95 = float(np.percentile(delta[floor_mask], 95)) if np.any(floor_mask) else 0.0
    material_hue_p95, material_saturation_p95 = _hsv_drift(reference, final, np.maximum(protected_f, floor_mask_f))
    input_material_hue_p95, input_material_saturation_p95 = _hsv_drift(original, final, np.maximum(protected_f, floor_mask_f))

    artifacts = _artifact_metrics(
        before,
        after,
        [wall_mask_f, ceiling_mask_f, floor_mask_f, protected_f],
        floor_mask_f,
    )
    floor_texture_ratio = _texture_ratio(before, after, floor_mask_f)
    protected_texture_ratio = _texture_ratio(before, after, protected_f)

    wall_uniformity_before = _uniformity_score(before, wall_mask_f)
    wall_uniformity_after = _uniformity_score(after, wall_mask_f)
    uniformity_increase = wall_uniformity_after - wall_uniformity_before

    neutral_sel = (np.maximum(wall_mask_f, ceiling_mask_f) > 0.5) & (after > 0.15) & (after < 0.95)
    if np.any(neutral_sel):
        residual_a = float(np.mean(np.abs(dst_lab[..., 1][neutral_sel] - 128.0)))
        residual_b = float(np.mean(np.abs(dst_lab[..., 2][neutral_sel] - 128.0)))
    else:
        residual_a = residual_b = 0.0
    wall_chroma_divergence = _wall_chroma_divergence(dst_lab, wall_mask_f)
    ceiling_valid = (ceiling_mask_f > 0.55) & (after > 0.15) & (after < 0.95)
    ceiling_warmth = (
        float(np.median(dst_lab[..., 2][ceiling_valid] - 128.0))
        if np.count_nonzero(ceiling_valid) >= 256 else 0.0
    )

    gains = [float(x) for x in (wb_log or {}).get("gains", [1.0, 1.0, 1.0])]
    wb_max_deviation = max(abs(g - 1.0) for g in gains) if gains else 0.0

    if highlight_clipped > limits["highlight_clip_percent"]:
        flags.append("EXCESSIVE_HIGHLIGHT_CLIPPING")
    if shadow_clipped > limits["shadow_clip_percent"]:
        flags.append("EXCESSIVE_SHADOW_CLIPPING")
    if abs(brightness_delta) > limits["global_brightness_delta"]:
        flags.append("EXCESSIVE_GLOBAL_BRIGHTNESS_CHANGE")
    if wb_max_deviation > limits["wb_gain_deviation"]:
        flags.append("WHITE_BALANCE_GAIN_LIMIT")
    if glare > 3.0:
        flags.append("WALL_GLARE")
    if protected_p95 > limits["protected_chroma_p95"]:
        flags.append("PROTECTED_MATERIAL_COLOR_SHIFT")
    if input_protected_p95 > limits["input_material_chroma_p95"]:
        flags.append("MATERIAL_COLOR_SHIFT_FROM_INPUT")
    if floor_luminance_shift > limits["floor_luminance_shift"]:
        flags.append("FLOOR_BRIGHTNESS_CHANGED")
    if max(residual_a, residual_b) > limits["residual_cast"]:
        flags.append("RESIDUAL_COLOR_CAST")
    if floor_chroma_p95 > limits["floor_chroma_p95"]:
        flags.append("FLOOR_COLOR_CHANGED")
    if material_hue_p95 > limits["material_hue_p95"]:
        flags.append("MATERIAL_HUE_DRIFT")
    if material_saturation_p95 > limits["material_saturation_p95"]:
        flags.append("MATERIAL_SATURATION_DRIFT")
    if uniformity_increase > limits["wall_uniformity_increase"]:
        flags.append("WALL_UNEVENNESS_INCREASED")
    if artifacts["floor_blotch_p95"] > limits["floor_blotch_p95"]:
        flags.append("FLOOR_BLOTCHING_DETECTED")
    if artifacts["mask_boundary_gradient_p95"] > limits["mask_boundary_gradient_p95"]:
        flags.append("HARD_MASK_BOUNDARY_OR_HALO")
    if artifacts["local_contrast_discontinuity_p99"] > limits["local_contrast_discontinuity_p99"]:
        flags.append("EXCESSIVE_LOCAL_CONTRAST_DISCONTINUITY")
    if min(floor_texture_ratio, protected_texture_ratio) < limits["texture_retention_min"]:
        flags.append("MATERIAL_TEXTURE_LOSS")
    if max(floor_texture_ratio, protected_texture_ratio) > limits["texture_amplification_max"]:
        flags.append("MATERIAL_TEXTURE_OVERENHANCED")
    if wall_chroma_divergence > limits["wall_chroma_divergence_p90"]:
        flags.append("WALL_CHROMA_DIVERGENCE")
    if ceiling_warmth > limits["ceiling_warmth_b_median"]:
        flags.append("EXCESSIVE_CEILING_WARMTH")
    if scene.route == "GLOBAL_SAFE":
        flags.append("GLOBAL_SAFE_ROUTE")

    return {
        "status": "REVIEW" if flags else "PASS",
        "flags": flags,
        "route": scene.route,
        "thresholds": limits,
        "achieved_targets": achieved,
        "highlight_clipped_percent_excluding_windows": highlight_clipped,
        "shadow_clipped_percent_excluding_windows": shadow_clipped,
        "clipped_percent_after_excluding_windows": highlight_clipped,
        "global_median_brightness_before": brightness_before,
        "global_median_brightness_after": brightness_after,
        "global_median_brightness_delta": brightness_delta,
        "white_balance_gains": gains,
        "white_balance_max_gain_deviation": wb_max_deviation,
        "wall_glare_percent_excluding_window_mirror": glare,
        "protected_chroma_mean_delta": protected_mean,
        "protected_chroma_p95_delta": protected_p95,
        "input_protected_chroma_p95_delta": input_protected_p95,
        "floor_luminance_shift": floor_luminance_shift,
        "floor_chroma_p95_delta": floor_chroma_p95,
        "material_hue_p95_delta": material_hue_p95,
        "material_saturation_p95_delta": material_saturation_p95,
        "input_material_hue_p95_delta": input_material_hue_p95,
        "input_material_saturation_p95_delta": input_material_saturation_p95,
        "floor_texture_retention_ratio": floor_texture_ratio,
        "protected_texture_retention_ratio": protected_texture_ratio,
        "wall_chroma_divergence_p90": wall_chroma_divergence,
        "ceiling_warmth_b_median": ceiling_warmth,
        **artifacts,
        "wall_uniformity_before": wall_uniformity_before,
        "wall_uniformity_after": wall_uniformity_after,
        "wall_uniformity_increase": uniformity_increase,
        "residual_cast_a": residual_a,
        "residual_cast_b": residual_b,
        "exposure_metrics": exposure_log or {},
    }
