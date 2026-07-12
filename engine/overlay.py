
from __future__ import annotations

import math

import cv2
import numpy as np

from .segmentation import SegmentationResult


PALETTE = {
    "wall": (255, 90, 40),
    "ceiling": (255, 220, 40),
    "floor": (70, 120, 180),
    "window": (40, 40, 255),
    "curtain": (220, 40, 220),
    "door": (40, 190, 70),
    "cabinet": (20, 150, 255),
    "painting": (180, 120, 60),
    "sofa": (110, 80, 170),
    "table": (120, 150, 180),
    "chair": (100, 180, 180),
    "bed": (180, 120, 180),
    "lamp": (80, 220, 255),
    "mirror": (180, 180, 220),
    "rug": (120, 100, 80),
    "sky": (255, 180, 60),
    "plant": (50, 160, 50),
}

MUTED_OTHER = (120, 120, 120)


def _coverage_for_requested(
    result: SegmentationResult,
    requested: str,
) -> float:
    class_id = result.name_to_id.get(requested)

    if class_id is None:
        return 0.0

    return float(
        np.mean(result.label_map == class_id) * 100.0
    )


def render_overlay(
    rgb: np.ndarray,
    result: SegmentationResult,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    color_map = np.zeros_like(rgb)

    assigned = np.zeros((height, width), dtype=bool)

    for name, color in PALETTE.items():
        class_id = result.name_to_id.get(name)
        if class_id is None:
            continue

        mask = result.label_map == class_id
        color_map[mask] = color
        assigned |= mask

    color_map[~assigned] = MUTED_OTHER

    blended = cv2.addWeighted(
        rgb,
        0.55,
        color_map,
        0.45,
        0,
    )

    legend_items = []
    for name in PALETTE:
        coverage = _coverage_for_requested(
            result,
            name,
        )
        if coverage > 0.1:
            legend_items.append(
                (name, PALETTE[name], coverage)
            )

    if not legend_items:
        return blended

    rows = max(1, math.ceil(len(legend_items) / 4))
    strip_height = rows * 36 + 12
    canvas = np.zeros(
        (height + strip_height, width, 3),
        dtype=np.uint8,
    )
    canvas[:height] = blended
    canvas[height:] = 28

    cell_width = max(1, width // 4)

    for index, (name, color, coverage) in enumerate(
        legend_items
    ):
        row = index // 4
        col = index % 4

        x = col * cell_width + 12
        y = height + row * 36 + 25

        cv2.rectangle(
            canvas,
            (x, y - 15),
            (x + 18, y + 3),
            color,
            -1,
        )

        cv2.putText(
            canvas,
            f"{name} {coverage:.1f}%",
            (x + 26, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    return canvas


def render_class_panel(
    rgb: np.ndarray,
    result: SegmentationResult,
    class_names: list[str],
) -> np.ndarray:
    names = class_names[:6]

    tile_height = max(1, rgb.shape[0] // 2)
    tile_width = max(1, rgb.shape[1] // 3)

    tiles: list[np.ndarray] = []

    for name in names:
        mask = result.masks(name)

        dimmed = np.clip(
            rgb.astype(np.float32) * 0.30,
            0,
            255,
        ).astype(np.uint8)

        highlighted = dimmed.copy()
        highlighted[mask > 0.5] = rgb[mask > 0.5]

        coverage = float(np.mean(mask > 0.5) * 100.0)

        cv2.rectangle(
            highlighted,
            (0, 0),
            (highlighted.shape[1] - 1, 44),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            highlighted,
            f"{name}: {coverage:.1f}%",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        tile = cv2.resize(
            highlighted,
            (tile_width, tile_height),
            interpolation=cv2.INTER_AREA,
        )
        tiles.append(tile)

    while len(tiles) < 6:
        tiles.append(
            np.zeros(
                (tile_height, tile_width, 3),
                dtype=np.uint8,
            )
        )

    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:6])

    return np.vstack([top, bottom])
