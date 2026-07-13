from __future__ import annotations

import numpy as np

from engine.scene import build_scene


def _blank(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.float32)


def test_low_coverage_but_usable_structure_uses_semantic_route():
    h, w = 100, 140
    masks = {
        "wall": _blank(h, w),
        "ceiling": _blank(h, w),
        "floor": _blank(h, w),
    }
    masks["wall"][20:70, 10:60] = 1.0  # under old 35% threshold

    scene = build_scene(masks, (h, w))

    assert scene.route == "SEMANTIC"
    assert any(
        "structure coverage" in reason
        for reason in scene.reasons
    )


def test_no_meaningful_structure_uses_global_safe():
    h, w = 100, 140
    masks = {
        "wall": _blank(h, w),
        "ceiling": _blank(h, w),
        "floor": _blank(h, w),
    }

    scene = build_scene(masks, (h, w))

    assert scene.route == "GLOBAL_SAFE"
    assert any(
        reason.startswith("CRITICAL:")
        for reason in scene.reasons
    )


def test_geometry_warning_does_not_force_global_safe():
    h, w = 100, 140
    masks = {
        "wall": _blank(h, w),
        "ceiling": _blank(h, w),
        "floor": _blank(h, w),
    }
    masks["ceiling"][40:60, 20:120] = 1.0

    scene = build_scene(masks, (h, w))

    assert scene.route == "SEMANTIC"
    assert any(
        "ceiling does not sufficiently touch top border" in reason
        for reason in scene.reasons
    )
