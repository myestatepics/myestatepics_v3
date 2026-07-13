from __future__ import annotations

import numpy as np

from engine.scene import Scene
from engine.tone_classify import classify_tones
from engine.wb import rough_global_white_balance, semantic_white_balance


def _synthetic_scene(h: int, w: int) -> Scene:
    zeros = np.zeros((h, w), np.float32)
    ceiling = zeros.copy()
    wall = zeros.copy()
    floor = zeros.copy()
    ceiling[: h // 3] = 1.0
    wall[h // 3 : 2 * h // 3] = 1.0
    floor[2 * h // 3 :] = 1.0
    structure = np.maximum.reduce([ceiling, wall, floor])
    masks = {
        "ceiling": ceiling,
        "wall": wall,
        "floor": floor,
        "door": zeros.copy(),
        "window": zeros.copy(),
        "curtain": zeros.copy(),
        "mirror": zeros.copy(),
        "cabinet": zeros.copy(),
        "sofa": zeros.copy(),
        "table": zeros.copy(),
        "chair": zeros.copy(),
        "bed": zeros.copy(),
        "rug": zeros.copy(),
        "plant": zeros.copy(),
        "painting": zeros.copy(),
        "lamp": zeros.copy(),
        "furnishings": zeros.copy(),
        "protected": zeros.copy(),
        "structure": structure,
    }
    coverage = {k: float(np.mean(v > 0.5) * 100.0) for k, v in masks.items()}
    return Scene(masks=masks, coverage=coverage, route="SEMANTIC", reasons=[])


def test_two_pass_wb_dark_tungsten_ceiling_stays_neutral_reference():
    h, w = 600, 900
    neutral = np.full((h, w, 3), 0.42, np.float32)
    cast = np.array([1.18, 1.00, 0.72], np.float32)
    rgb = np.clip(neutral * cast * 255.0, 0, 255).astype(np.uint8)
    scene = _synthetic_scene(h, w)

    pass1, log1 = rough_global_white_balance(rgb, scene)
    tones, _ = classify_tones(pass1, scene, {})
    final, log2 = semantic_white_balance(pass1, scene, tones)

    ceiling = scene.masks["ceiling"] > 0.5
    means = final[ceiling].reshape(-1, 3).mean(axis=0) / 255.0
    spread = float(means.max() - means.min())

    assert log1["applied"] is True
    assert tones["ceiling"]["ceiling_is_neutral_reference"] is True
    assert tones["ceiling"]["target"] >= 0.79
    assert log2["reference_used"] == "ceiling"
    assert spread < 0.006, means
