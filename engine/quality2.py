
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


def _lum(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return (
        0.2126 * f[..., 0]
        + 0.7152 * f[..., 1]
        + 0.0722 * f[..., 2]
    )


def _uniformity_score(y: np.ndarray, mask: np.ndarray) -> float:
    pixels = y[mask > 0.5]
    if pixels.size < 100:
        return 0.0
    p10, p90 = np.percentile(pixels, [10, 90])
    return float(p90 - p10)


def evaluate(
    original: np.ndarray,
    final: np.ndarray,
    scene: Scene,
    tones: dict,
) -> dict:
    before = _lum(original)
    after = _lum(final)
    flags: list[str] = []
    achieved = {}

    wall_target = tones["wall_target_map"]
    wall_mask = scene.masks["wall"] > 0.5

    if np.any(wall_mask):
        target_median = float(np.median(wall_target[wall_mask]))
        eff_list = tones.get("wall_effective_targets") or []
        if eff_list:
            # judge against what the safety caps actually allowed
            target_median = min(target_median, max(e["effective"] for e in eff_list))
        achieved_median = float(np.median(after[wall_mask]))
        achieved["wall"] = {
            "target": target_median,
            "after": achieved_median,
        }

        if achieved_median < target_median - 0.06:
            flags.append("WALL_TARGET_MISSED")
        if achieved_median > target_median + 0.10:
            flags.append("WALL_OVERSHOT")

    ceiling_mask = scene.masks["ceiling"] > 0.5
    if np.any(ceiling_mask):
        target = float(tones["ceiling"].get("effective_target", tones["ceiling"]["target"]))
        median_after = float(np.median(after[ceiling_mask]))
        achieved["ceiling"] = {
            "target": target,
            "after": median_after,
        }

        if median_after < target - 0.06:
            flags.append("CEILING_TARGET_MISSED")
        if median_after > target + 0.10:
            flags.append("CEILING_OVERSHOT")

    exclude = np.maximum(
        scene.masks["window"],
        scene.masks["mirror"],
    )
    non_window = exclude < 0.3

    clipped = float(
        np.mean((after > 0.985) & non_window) * 100.0
    )
    local = cv2.GaussianBlur(
        after.astype(np.float32),
        (0, 0),
        28.0,
    )
    glare = float(
        np.mean(
            (after > 0.90)
            & (local > 0.72)
            & non_window
        )
        * 100.0
    )

    src_lab = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    dst_lab = cv2.cvtColor(
        final,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    delta = np.sqrt(
        (dst_lab[..., 1] - src_lab[..., 1]) ** 2
        + (dst_lab[..., 2] - src_lab[..., 2]) ** 2
    )

    protected_mask = np.maximum(
        scene.masks["protected"],
        scene.masks["floor"],
    ) > 0.5

    protected_p95 = (
        float(np.percentile(delta[protected_mask], 95))
        if np.any(protected_mask)
        else 0.0
    )
    protected_mean = (
        float(np.mean(delta[protected_mask]))
        if np.any(protected_mask)
        else 0.0
    )

    floor_mask = scene.masks["floor"] > 0.5
    floor_luminance_shift = (
        float(
            abs(
                np.median(after[floor_mask])
                - np.median(before[floor_mask])
            )
        )
        if np.any(floor_mask)
        else 0.0
    )
    floor_chroma_p95 = (
        float(np.percentile(delta[floor_mask], 95))
        if np.any(floor_mask)
        else 0.0
    )

    wall_uniformity_before = _uniformity_score(
        before,
        scene.masks["wall"],
    )
    wall_uniformity_after = _uniformity_score(
        after,
        scene.masks["wall"],
    )
    uniformity_increase = (
        wall_uniformity_after - wall_uniformity_before
    )

    # v3.2: residual color cast on illumination surfaces after WB
    neutral_sel = (
        (np.maximum(scene.masks["wall"], scene.masks["ceiling"]) > 0.5)
        & (after > 0.15) & (after < 0.95)
    )
    if np.any(neutral_sel):
        residual_a = float(np.mean(np.abs(dst_lab[..., 1][neutral_sel] - 128.0)))
        residual_b = float(np.mean(np.abs(dst_lab[..., 2][neutral_sel] - 128.0)))
    else:
        residual_a = residual_b = 0.0

    if clipped > 1.8:
        flags.append("EXCESSIVE_CLIPPING")
    if glare > 3.0:
        flags.append("WALL_GLARE")
    if protected_p95 > 9.0:
        flags.append("PROTECTED_MATERIAL_COLOR_SHIFT")
    # v3.2: floors legitimately inherit room light (full field). The gate now
    # catches only ABNORMAL shifts; color fidelity gates remain strict.
    if floor_luminance_shift > 0.18:
        flags.append("FLOOR_BRIGHTNESS_CHANGED")
    if max(residual_a, residual_b) > 5.0:
        flags.append("RESIDUAL_COLOR_CAST")
    if floor_chroma_p95 > 7.0:
        flags.append("FLOOR_COLOR_CHANGED")
    if uniformity_increase > 0.08:
        flags.append("WALL_UNEVENNESS_INCREASED")
    if scene.route == "GLOBAL_SAFE":
        flags.append("GLOBAL_SAFE_ROUTE")

    return {
        "status": "REVIEW" if flags else "PASS",
        "flags": flags,
        "route": scene.route,
        "achieved_targets": achieved,
        "clipped_percent_after_excluding_windows": clipped,
        "wall_glare_percent_excluding_window_mirror": glare,
        "protected_chroma_mean_delta": protected_mean,
        "protected_chroma_p95_delta": protected_p95,
        "floor_luminance_shift": floor_luminance_shift,
        "floor_chroma_p95_delta": floor_chroma_p95,
        "wall_uniformity_before": wall_uniformity_before,
        "wall_uniformity_after": wall_uniformity_after,
        "wall_uniformity_increase": uniformity_increase,
        "residual_cast_a": residual_a,
        "residual_cast_b": residual_b,
    }
