
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


REQUESTED_CLASSES = [
    "wall", "ceiling", "floor", "window", "curtain", "door",
    "cabinet", "mirror", "sofa", "table", "chair", "bed",
    "rug", "plant", "painting", "lamp",
]


@dataclass
class Scene:
    masks: dict[str, np.ndarray]
    coverage: dict[str, float]
    route: str
    reasons: list[str]


def _union(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    for mask in masks:
        result = np.maximum(result, mask.astype(np.float32))
    return np.clip(result, 0.0, 1.0)


def build_scene(refined_masks: dict, image_shape) -> Scene:
    h, w = int(image_shape[0]), int(image_shape[1])
    shape = (h, w)
    masks = {
        name: refined_masks.get(name, np.zeros(shape, dtype=np.float32)).astype(np.float32)
        for name in REQUESTED_CLASSES
    }

    structure = _union([masks["wall"], masks["ceiling"], masks["floor"]], shape)
    furnishings = _union([
        masks["sofa"], masks["table"], masks["chair"], masks["bed"],
        masks["cabinet"], masks["rug"],
    ], shape)
    protected = _union([
        furnishings, masks["painting"], masks["plant"], masks["mirror"],
    ], shape)

    masks["structure"] = structure
    masks["furnishings"] = furnishings
    masks["protected"] = protected

    coverage = {
        name: float(np.mean(mask > 0.5) * 100.0)
        for name, mask in masks.items()
    }

    labeled = _union([masks[name] for name in REQUESTED_CLASSES], shape)
    unlabeled_percent = float(np.mean(labeled <= 0.5) * 100.0)
    structure_percent = coverage["structure"]

    reasons: list[str] = []
    if structure_percent < 35.0:
        reasons.append(f"structure coverage {structure_percent:.1f}% < 35%")
    if unlabeled_percent > 45.0:
        reasons.append(f"unlabeled area {unlabeled_percent:.1f}% > 45%")

    # Geometric sanity checks.
    if coverage["ceiling"] > 1.0:
        top_touch = float(np.mean(masks["ceiling"][: max(1, h // 20)] > 0.5))
        if top_touch < 0.10:
            reasons.append("ceiling does not sufficiently touch top border")
    if coverage["floor"] > 1.0:
        bottom_touch = float(np.mean(masks["floor"][-max(1, h // 20):] > 0.5))
        if bottom_touch < 0.10:
            reasons.append("floor does not sufficiently touch bottom border")
        upper_floor = float(np.mean(masks["floor"][: h // 3] > 0.5) * 100.0)
        if upper_floor > 8.0:
            reasons.append(f"floor occupies {upper_floor:.1f}% of upper image")
    if coverage["window"] > 0.2 and coverage["wall"] > 1.0:
        wall_near = np.maximum(
            masks["wall"],
            np.clip(np.asarray(cv2.dilate(
                (masks["wall"] > 0.3).astype(np.uint8),
                np.ones((31, 31), np.uint8),
            ), dtype=np.float32), 0, 1),
        )
        adjacency = float(np.mean(wall_near[masks["window"] > 0.5])) if np.any(masks["window"] > 0.5) else 1.0
        if adjacency < 0.20:
            reasons.append("window masks are not adjacent to walls")

    route = "SEMANTIC" if not reasons else "GLOBAL_SAFE"
    coverage["unlabeled"] = unlabeled_percent
    return Scene(masks=masks, coverage=coverage, route=route, reasons=reasons)
