import cv2
import numpy as np

from engine.exposure_fusion import ExposureFusionConfig, apply_semantic_exposure
from engine.scene import build_scene


def _scene(h=180, w=240):
    masks = {name: np.zeros((h, w), np.float32) for name in ["ceiling", "wall", "floor", "window"]}
    masks["ceiling"][: h // 4] = 1.0
    masks["wall"][h // 4 : 2 * h // 3] = 1.0
    masks["floor"][2 * h // 3 :] = 1.0
    masks["window"][h // 4 : h // 2, 3 * w // 4 :] = 1.0
    return build_scene(masks, (h, w))


def _luma(rgb):
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def test_semantic_exposure_lifts_floor_more_than_wall():
    h, w = 180, 240
    ref = np.zeros((h, w, 3), np.uint8)
    ref[: h // 4] = [130, 130, 130]
    ref[h // 4 : 2 * h // 3] = [95, 92, 88]
    ref[2 * h // 3 :] = [48, 34, 24]
    corrected = ref.copy()
    scene = _scene(h, w)
    cfg = ExposureFusionConfig()
    out, log = apply_semantic_exposure(ref, corrected, scene, cfg)
    wall = scene.masks["wall"] > 0.5
    floor = scene.masks["floor"] > 0.5
    wall_lift = float(np.median(_luma(out)[wall] - _luma(ref)[wall]))
    floor_lift = float(np.median(_luma(out)[floor] - _luma(ref)[floor]))
    assert log["applied"] is True
    assert floor_lift > wall_lift
    assert floor_lift > 0.06


def test_semantic_exposure_preserves_hue_in_floor():
    h, w = 180, 240
    ref = np.full((h, w, 3), [95, 90, 84], np.uint8)
    ref[2 * h // 3 :] = [54, 34, 20]
    scene = _scene(h, w)
    out, _ = apply_semantic_exposure(ref, ref.copy(), scene, ExposureFusionConfig())
    floor = scene.masks["floor"] > 0.5
    before = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(float)
    after = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(float)
    shift = np.median(np.linalg.norm(after[floor, 1:3] - before[floor, 1:3], axis=1))
    assert shift < 4.0


def test_semantic_exposure_does_not_brighten_window_region():
    h, w = 180, 240
    ref = np.full((h, w, 3), 80, np.uint8)
    scene = _scene(h, w)
    window = scene.masks["window"] > 0.5
    ref[window] = 245
    out, _ = apply_semantic_exposure(ref, ref.copy(), scene, ExposureFusionConfig())
    assert float(np.mean(_luma(out)[window] - _luma(ref)[window])) < 0.01
