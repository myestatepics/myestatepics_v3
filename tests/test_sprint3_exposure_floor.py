import cv2
import numpy as np

from engine.exposure_fusion import adaptive_exposure_config, exposure_fusion
from engine.materials2 import apply_material_guardrails
from engine.scene import build_scene


def _scene(h=180, w=240):
    names = ["wall", "ceiling", "floor", "window", "cabinet"]
    masks = {name: np.zeros((h, w), np.float32) for name in names}
    masks["ceiling"][: h // 4] = 1.0
    masks["wall"][h // 4 : 2 * h // 3] = 1.0
    masks["floor"][2 * h // 3 :] = 1.0
    masks["window"][h // 4 : h // 2, 3 * w // 4 :] = 1.0
    return build_scene(masks, (h, w))


def _luma(rgb):
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def test_scene_target_ignores_bright_window_and_raises_dark_room_target():
    h, w = 180, 240
    rgb = np.full((h, w, 3), 68, np.uint8)
    rgb[h // 4 : h // 2, 3 * w // 4 :] = 250
    cfg, log = adaptive_exposure_config(rgb, scene=_scene(h, w))
    assert log["adaptive_profile"] == "dark_interior"
    assert log["room_median"] < log["input_mean"]
    assert cfg.target_median >= 0.58


def test_floor_recovery_opens_dark_wood_without_large_hue_shift():
    h, w = 180, 240
    rgb = np.full((h, w, 3), [92, 88, 82], np.uint8)
    rgb[2 * h // 3 :] = [55, 34, 20]
    scene = _scene(h, w)
    out, log = exposure_fusion(rgb, scene=scene)
    floor = scene.masks["floor"] > 0.5
    assert np.median(_luma(out)[floor]) > np.median(_luma(rgb)[floor]) + 0.08
    assert log["floor_recovery"]["applied"] is True
    before = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    after = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(float)
    chroma_shift = np.median(np.linalg.norm(after[floor, 1:3] - before[floor, 1:3], axis=1))
    assert chroma_shift < 8.0


def test_material_guardrail_allows_floor_recovery_but_holds_dark_wall():
    h, w = 180, 240
    ref = np.full((h, w, 3), 55, np.uint8)
    ref[2 * h // 3 :] = [50, 31, 18]
    corrected = np.full((h, w, 3), 110, np.uint8)
    corrected[2 * h // 3 :] = [105, 72, 48]
    scene = _scene(h, w)
    out, _ = apply_material_guardrails(ref, corrected, scene, {"floor_max_lift": 0.20})
    wall = scene.masks["wall"] > 0.5
    floor = scene.masks["floor"] > 0.5
    wall_lift = np.mean(_luma(out)[wall] - _luma(ref)[wall])
    floor_lift = np.mean(_luma(out)[floor] - _luma(ref)[floor])
    assert wall_lift < 0.075
    assert floor_lift > wall_lift + 0.04
    assert floor_lift <= 0.22
