from __future__ import annotations

import numpy as np

from engine.scene import Scene
from engine.tonal import adaptive_per_class_exposure


def _scene(h: int, w: int) -> Scene:
    zeros = np.zeros((h, w), np.float32)
    wall = zeros.copy()
    ceiling = zeros.copy()
    floor = zeros.copy()
    cabinet = zeros.copy()

    wall[80:420, 0:w] = 1.0
    ceiling[:80, :] = 1.0
    floor[420:, :] = 1.0
    cabinet[300:520, 420:600] = 1.0

    furnishings = cabinet.copy()
    protected = cabinet.copy()

    masks = {
        "wall": wall,
        "ceiling": ceiling,
        "floor": floor,
        "door": zeros.copy(),
        "window": zeros.copy(),
        "curtain": zeros.copy(),
        "mirror": zeros.copy(),
        "cabinet": cabinet,
        "sofa": zeros.copy(),
        "table": zeros.copy(),
        "chair": zeros.copy(),
        "bed": zeros.copy(),
        "rug": zeros.copy(),
        "plant": zeros.copy(),
        "painting": zeros.copy(),
        "lamp": zeros.copy(),
        "furnishings": furnishings,
        "protected": protected,
        "structure": np.maximum.reduce([wall, ceiling, floor]),
    }
    coverage = {
        k: float(np.mean(v > 0.5) * 100.0)
        for k, v in masks.items()
    }
    return Scene(masks=masks, coverage=coverage, route="SEMANTIC", reasons=[])


def test_black_wall_gain_is_uniform_and_floor_is_capped():
    h, w = 600, 800
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = [30, 30, 30]
    rgb[:80] = [100, 100, 100]
    rgb[420:] = [65, 38, 25]
    rgb[300:520, 420:600] = [45, 25, 15]

    scene = _scene(h, w)
    tones = {
        "wall_target_map": np.full((h, w), 0.22, np.float32),
        "ceiling": {"target": 0.75},
        "door": {"target": 0.65},
    }
    analysis = {"material_inherit_factor": 0.08}

    out, log = adaptive_per_class_exposure(rgb, scene, tones, analysis)

    assert out.shape == rgb.shape
    assert log["wall_gain_std"] < 0.03
    assert log["floor_gain_max"] <= 1.081
    assert log["protected_gain_max"] <= 1.081
    assert log["chroma_compensation"] == "retired"
