
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene


def _luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _chroma(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)


def _masked_stats(y: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    pixels = y[mask > 0.5]
    if pixels.size == 0:
        return 0.0, 0.0, 0
    return float(np.median(pixels)), float(np.percentile(pixels, 70)), int(pixels.size)


def classify_tones(rgb: np.ndarray, scene: Scene, settings: dict | None = None) -> tuple[dict, dict]:
    cfg = settings or {}
    y = _luminance(rgb)
    chroma = _chroma(rgb)
    h, w = y.shape
    total = h * w

    wall_mask = (scene.masks["wall"] > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
    wall_target_map = np.zeros_like(y, dtype=np.float32)
    wall_class_map = np.zeros_like(wall_mask, dtype=np.uint8)
    wall_components = []

    component_records = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(total * 0.03):
            continue
        mask = labels == label
        median = float(np.median(y[mask]))
        p70 = float(np.percentile(y[mask], 70))
        # p70 helps avoid treating a white wall in shadow as black paint.
        material_score = 0.55 * median + 0.45 * p70
        if material_score >= cfg.get("wall_light_threshold", 0.45):
            tone_class = "LIGHT"
            target = cfg.get("wall_light_target", 0.62)
            code = 3
        elif material_score >= cfg.get("wall_medium_threshold", 0.28):
            tone_class = "MEDIUM"
            target = median + min(max(0.55 - median, 0.0), cfg.get("wall_medium_max_lift", 0.16))
            code = 2
        else:
            tone_class = "DARK"
            target = median + cfg.get("wall_dark_lift", 0.12)
            target = min(target, median + cfg.get("wall_dark_max_lift", 0.14))
            code = 1
        wall_target_map[mask] = target
        wall_class_map[mask] = code
        component_records.append((area, tone_class, target, code))
        wall_components.append({
            "label": int(label),
            "area_percent": float(area / total * 100.0),
            "median": median,
            "p70": p70,
            "material_score": material_score,
            "class": tone_class,
            "target": float(target),
        })

    if component_records:
        majority = max(component_records, key=lambda x: x[0])
        majority_class, majority_target, majority_code = majority[1], majority[2], majority[3]
    else:
        majority_class, majority_target, majority_code = "MEDIUM", 0.55, 2

    small_wall = (wall_mask > 0) & (wall_target_map == 0)
    wall_target_map[small_wall] = majority_target
    wall_class_map[small_wall] = majority_code

    floor_median, floor_p70, floor_count = _masked_stats(y, scene.masks["floor"])
    floor_score = 0.60 * floor_median + 0.40 * floor_p70
    if floor_count == 0:
        floor_class, floor_target = "ABSENT", 0.0
    elif floor_score >= cfg.get("floor_light_threshold", 0.45):
        floor_class = "LIGHT"
        floor_target = min(floor_median + cfg.get("floor_light_lift", 0.04), 0.90)
    elif floor_score >= cfg.get("floor_medium_threshold", 0.30):
        floor_class = "MEDIUM"
        floor_target = min(floor_median + cfg.get("floor_medium_lift", 0.12), cfg.get("floor_medium_target_cap", 0.52))
    else:
        floor_class = "DARK"
        floor_target = floor_median + min(cfg.get("floor_dark_lift", 0.10), cfg.get("floor_dark_max_lift", 0.12))

    ceiling_median, ceiling_p70, ceiling_count = _masked_stats(y, scene.masks["ceiling"])
    ceiling_binary = scene.masks["ceiling"] > 0.5
    ceiling_valid = ceiling_binary & (y > 0.15) & (y < 0.95)
    ceiling_pixels_chroma = chroma[ceiling_valid]
    ceiling_chroma = (
        float(np.median(ceiling_pixels_chroma))
        if ceiling_pixels_chroma.size
        else 999.0
    )

    # Neutrality is a color/spatial-consistency decision, not an exposure
    # decision. A dark neutral ceiling remains a valid neutral reference.
    mid = w // 2
    left_valid = ceiling_valid.copy()
    left_valid[:, mid:] = False
    right_valid = ceiling_valid.copy()
    right_valid[:, :mid] = False
    left_chroma = chroma[left_valid]
    right_chroma = chroma[right_valid]
    if left_chroma.size >= 800 and right_chroma.size >= 800:
        ceiling_chroma_half_difference = float(
            abs(np.median(left_chroma) - np.median(right_chroma))
        )
        ceiling_spatially_consistent = ceiling_chroma_half_difference <= 4.0
    else:
        ceiling_chroma_half_difference = None
        ceiling_spatially_consistent = ceiling_pixels_chroma.size >= 2000

    ceiling_neutral = bool(
        ceiling_pixels_chroma.size >= 2000
        and ceiling_chroma
        < cfg.get("ceiling_neutral_chroma_threshold", 14.0)
        and ceiling_spatially_consistent
    )
    if ceiling_count == 0:
        ceiling_target = 0.0
    elif ceiling_neutral:
        configured = cfg.get("ceiling_target", 0.80)
        headroom_safe = min(configured, max(ceiling_median + 0.08, 0.86 - max(0.0, float(np.percentile(y, 95)) - 0.88)))
        ceiling_target = max(ceiling_median, headroom_safe)
    else:
        ceiling_target = ceiling_median + min(cfg.get("ceiling_nonneutral_lift", 0.12), cfg.get("ceiling_nonneutral_max_lift", 0.15))

    door_median, door_p70, door_count = _masked_stats(y, scene.masks["door"])
    if door_count == 0:
        door_class, door_target = "ABSENT", 0.0
    elif 0.55 * door_median + 0.45 * door_p70 >= 0.45:
        door_class, door_target = "LIGHT", cfg.get("door_light_target", 0.68)
    else:
        door_class, door_target = "DARK", door_median + cfg.get("door_dark_lift", 0.10)

    tones = {
        "wall_target_map": wall_target_map,
        "wall_class_map": wall_class_map,
        "wall_components": wall_components,
        "wall_majority_class": majority_class,
        "floor": {
            "median": floor_median, "p70": floor_p70,
            "class": floor_class, "target": float(floor_target),
        },
        "ceiling": {
            "median": ceiling_median, "p70": ceiling_p70,
            "chroma_median": ceiling_chroma,
            "chroma_half_difference": ceiling_chroma_half_difference,
            "spatially_consistent": ceiling_spatially_consistent,
            "valid_reference_pixels": int(ceiling_pixels_chroma.size),
            "target": float(ceiling_target),
            "ceiling_is_neutral_reference": ceiling_neutral,
        },
        "door": {
            "median": door_median, "p70": door_p70,
            "class": door_class, "target": float(door_target),
        },
    }

    log = {
        "wall_components": wall_components,
        "wall_majority_class": majority_class,
        "floor": tones["floor"],
        "ceiling": tones["ceiling"],
        "door": tones["door"],
    }
    return tones, log
