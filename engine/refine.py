
from __future__ import annotations

import cv2
import numpy as np


FEATHER_SIGMA = {
    "wall": 6.0,
    "ceiling": 6.0,
    "floor": 6.0,
    "window": 3.0,
    "curtain": 3.0,
    "mirror": 3.0,
    "door": 4.0,
    "cabinet": 4.0,
    "sofa": 4.0,
    "table": 4.0,
    "chair": 4.0,
    "bed": 4.0,
    "rug": 4.0,
    "plant": 4.0,
    "painting": 4.0,
    "lamp": 4.0,
}


def _remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    removed = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 1
        else:
            removed += 1
    return cleaned, removed


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> tuple[np.ndarray, int]:
    binary = (mask > 0).astype(np.uint8)
    inverse = 1 - binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    filled = binary.copy()
    holes_filled = 0
    h, w = binary.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches_border and area <= max_hole_area:
            filled[labels == label] = 1
            holes_filled += 1
    return filled, holes_filled


def _edge_refine(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    src = mask.astype(np.float32)
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        refined = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=src,
            radius=16,
            eps=1e-3,
        )
        return refined.astype(np.float32), "guided_filter"
    refined = cv2.bilateralFilter(src, d=9, sigmaColor=0.1, sigmaSpace=16)
    return refined.astype(np.float32), "bilateral_fallback"


def refine_masks(
    rgb_full: np.ndarray,
    label_map_full: np.ndarray,
    class_names,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    class_names may be either a name->id mapping or an iterable of (name,id) pairs.
    The public signature remains exactly three parameters.
    """
    if isinstance(class_names, dict):
        name_to_id = {str(k).lower(): int(v) for k, v in class_names.items()}
    else:
        name_to_id = {str(name).lower(): int(class_id) for name, class_id in class_names}

    total = label_map_full.size
    min_component_area = max(1, int(total * 0.0005))
    max_hole_area = max(1, int(total * 0.0002))

    masks: dict[str, np.ndarray] = {}
    class_logs: dict[str, dict] = {}
    filter_modes: set[str] = set()

    for name, class_id in name_to_id.items():
        raw = (label_map_full == class_id).astype(np.uint8)
        if not np.any(raw):
            continue

        raw_percent = float(np.mean(raw > 0) * 100.0)
        cleaned, components_removed = _remove_small_components(raw, min_component_area)
        filled, holes_filled = _fill_small_holes(cleaned, max_hole_area)
        edge_refined, filter_mode = _edge_refine(rgb_full, filled)
        filter_modes.add(filter_mode)

        sigma = FEATHER_SIGMA.get(name, 4.0)
        feathered = cv2.GaussianBlur(edge_refined, (0, 0), sigma)
        feathered = np.clip(feathered, 0.0, 1.0).astype(np.float32)

        masks[name] = feathered
        class_logs[name] = {
            "raw_percent": raw_percent,
            "refined_percent": float(np.mean(feathered > 0.5) * 100.0),
            "components_removed": int(components_removed),
            "holes_filled": int(holes_filled),
            "feather_sigma": float(sigma),
        }

    return masks, {
        "filter_modes": sorted(filter_modes),
        "classes": class_logs,
        "min_component_area_pixels": int(min_component_area),
        "max_hole_area_pixels": int(max_hole_area),
    }
