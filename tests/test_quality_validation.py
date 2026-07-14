from types import SimpleNamespace

import cv2
import numpy as np

from engine.quality2 import evaluate


def _scene(h=40, w=40):
    z = np.zeros((h, w), np.float32)
    o = np.ones((h, w), np.float32)
    return SimpleNamespace(
        masks={
            "wall": o,
            "ceiling": z,
            "floor": z,
            "protected": z,
            "window": z,
            "mirror": z,
        },
        route="SEMANTIC",
    )


def _tones(h=40, w=40):
    return {
        "wall_target_map": np.full((h, w), 0.5, np.float32),
        "wall_effective_targets": [],
        "ceiling": {},
    }


def test_quality_passes_for_small_safe_change():
    src = np.full((40, 40, 3), 110, np.uint8)
    dst = np.full((40, 40, 3), 120, np.uint8)
    result = evaluate(src, dst, _scene(), _tones(), wb_log={"gains": [1.02, 0.99, 0.99]})
    assert result["status"] == "PASS"
    assert result["flags"] == []


def test_quality_flags_highlight_clipping():
    src = np.full((40, 40, 3), 120, np.uint8)
    dst = src.copy()
    dst[:20] = 255
    result = evaluate(src, dst, _scene(), _tones())
    assert "EXCESSIVE_HIGHLIGHT_CLIPPING" in result["flags"]
    assert result["status"] == "REVIEW"


def test_quality_flags_shadow_clipping():
    src = np.full((40, 40, 3), 90, np.uint8)
    dst = src.copy()
    dst[:10] = 0
    result = evaluate(src, dst, _scene(), _tones())
    assert "EXCESSIVE_SHADOW_CLIPPING" in result["flags"]


def test_quality_flags_wb_gain_limit():
    src = np.full((40, 40, 3), 110, np.uint8)
    result = evaluate(src, src.copy(), _scene(), _tones(), wb_log={"gains": [1.25, 0.90, 0.90]})
    assert "WHITE_BALANCE_GAIN_LIMIT" in result["flags"]


def test_quality_reports_brightness_and_drift_metrics():
    src = np.full((40, 40, 3), [100, 90, 80], np.uint8)
    dst = np.full((40, 40, 3), [112, 101, 90], np.uint8)
    result = evaluate(src, dst, _scene(), _tones(), exposure_log={"method": "mertens"})
    assert "global_median_brightness_delta" in result
    assert "material_hue_p95_delta" in result
    assert result["exposure_metrics"]["method"] == "mertens"


def test_quality_flags_floor_blotches_and_hard_boundaries():
    h, w = 120, 160
    z = np.zeros((h, w), np.float32)
    floor = z.copy()
    floor[h // 2 :] = 1.0
    scene = SimpleNamespace(
        masks={
            "wall": 1.0 - floor,
            "ceiling": z,
            "floor": floor,
            "protected": z,
            "window": z,
            "mirror": z,
        },
        route="SEMANTIC",
    )
    src = np.full((h, w, 3), 100, np.uint8)
    dst = src.copy()
    dst[h // 2 :, : w // 2] = 220
    result = evaluate(src, dst, scene, {"ceiling": {}})
    assert "FLOOR_BLOTCHING_DETECTED" in result["flags"]
    assert "HARD_MASK_BOUNDARY_OR_HALO" in result["flags"]


def test_quality_flags_material_texture_loss():
    h, w = 120, 160
    checker = ((np.indices((h, w)).sum(axis=0) % 2) * 50 + 80).astype(np.uint8)
    src = np.repeat(checker[..., None], 3, axis=2)
    dst = cv2.GaussianBlur(src, (9, 9), 0)
    scene = _scene(h, w)
    scene.masks["floor"] = np.ones((h, w), np.float32)
    result = evaluate(src, dst, scene, _tones(h, w))
    assert "MATERIAL_TEXTURE_LOSS" in result["flags"]


def test_quality_flags_excessive_ceiling_warmth():
    h, w = 80, 100
    scene = _scene(h, w)
    scene.masks["wall"] = np.zeros((h, w), np.float32)
    scene.masks["ceiling"] = np.ones((h, w), np.float32)
    src = np.full((h, w, 3), 140, np.uint8)
    dst = np.full((h, w, 3), [165, 140, 105], np.uint8)
    result = evaluate(src, dst, scene, {"ceiling": {}})
    assert "EXCESSIVE_CEILING_WARMTH" in result["flags"]
