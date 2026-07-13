
import numpy as np
import cv2

from engine.scene import build_scene
from engine.tone_classify import classify_tones
from engine.tonal import adaptive_per_class_exposure
from engine.wb import global_pass1_wb, semantic_white_balance


def make_cast_room(cast=(1.18, 1.00, 0.72), ceil=0.42, wall=0.34):
    H, W = 450, 600
    img = np.zeros((H, W, 3), np.float32)
    c = np.array(cast)
    img[: int(H * 0.30)] = ceil * c
    img[int(H * 0.30):int(H * 0.70)] = wall * c
    rng = np.random.default_rng(3)
    grain = rng.normal(0, 0.02, (H - int(H * 0.70), W, 1))
    img[int(H * 0.70):] = np.clip(np.array([0.30, 0.20, 0.12]) + grain, 0, 1)
    grad = np.linspace(-0.05, 0.05, W)[None, :, None]
    img[int(H * 0.30):int(H * 0.70)] += grad
    rgb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    masks = {k: np.zeros((H, W), np.float32) for k in ["ceiling", "wall", "floor"]}
    masks["ceiling"][: int(H * 0.30)] = 1.0
    masks["wall"][int(H * 0.30):int(H * 0.70)] = 1.0
    masks["floor"][int(H * 0.70):] = 1.0
    return rgb, build_scene(masks, (H, W))


def _lum(rgb):
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def test_two_pass_wb_removes_tungsten_cast():
    rgb, scene = make_cast_room()
    pass1, p1log = global_pass1_wb(rgb, scene)
    assert p1log["pass1_applied"] is True
    tones, _ = classify_tones(pass1, scene, {})
    assert tones["ceiling"]["ceiling_is_neutral_reference"] is True
    wb, _ = semantic_white_balance(pass1, scene, tones)
    f = wb.astype(np.float32) / 255.0
    ceil_rgb = f[:135].reshape(-1, 3).mean(axis=0)
    assert float(ceil_rgb.max() - ceil_rgb.min()) < 0.006


def test_dark_neutral_ceiling_keeps_full_target():
    rgb, scene = make_cast_room()
    pass1, _ = global_pass1_wb(rgb, scene)
    tones, _ = classify_tones(pass1, scene, {})
    assert tones["ceiling"]["target"] >= 0.72


def test_multiplicative_preserves_material_ratios():
    rgb, scene = make_cast_room(cast=(1.0, 1.0, 1.0))
    tones, _ = classify_tones(rgb, scene, {})
    out, log = adaptive_per_class_exposure(rgb, scene, tones, {"material_inherit_factor": 1.0})
    floor = scene.masks["floor"] > 0.5
    f0 = rgb.astype(np.float32)[floor] + 1e-3
    f1 = out.astype(np.float32)[floor] + 1e-3
    r0 = (f0[:, 0] / f0[:, 1]).mean()
    r1 = (f1[:, 0] / f1[:, 1]).mean()
    assert abs(r1 - r0) / r0 < 0.02  # R:G ratio within 2 percent


def test_wall_texture_preserved():
    rgb, scene = make_cast_room(cast=(1.0, 1.0, 1.0))
    tones, _ = classify_tones(rgb, scene, {})
    out, _ = adaptive_per_class_exposure(rgb, scene, tones, {"material_inherit_factor": 1.0})
    m = scene.masks["wall"] > 0.5
    y0, y1 = _lum(rgb), _lum(out)
    assert y1[m].std() >= 0.80 * y0[m].std()


def test_floor_receives_room_light():
    rgb, scene = make_cast_room(cast=(1.0, 1.0, 1.0))
    tones, _ = classify_tones(rgb, scene, {})
    out, _ = adaptive_per_class_exposure(rgb, scene, tones, {"material_inherit_factor": 1.0})
    y0, y1 = _lum(rgb), _lum(out)
    fm = scene.masks["floor"] > 0.5
    wm = scene.masks["wall"] > 0.5
    floor_lift = float(np.median(y1[fm]) - np.median(y0[fm]))
    wall_lift = float(np.median(y1[wm]) - np.median(y0[wm]))
    assert floor_lift > 0.0
    assert floor_lift >= 0.3 * wall_lift
