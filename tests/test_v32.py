
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


def test_single_pass_wb_reduces_tungsten_cast():
    rgb, scene = make_cast_room()
    pass1, p1log = global_pass1_wb(rgb, scene)
    assert p1log["pass1_applied"] is True
    tones, _ = classify_tones(pass1, scene, {})
    wb, second_log = semantic_white_balance(pass1, scene, tones)
    assert second_log["reason"] == "single_pass_already_completed"
    assert np.array_equal(wb, pass1)
    before = rgb[:135].reshape(-1, 3).mean(axis=0)
    after = wb[:135].reshape(-1, 3).mean(axis=0)
    assert float(after.max() - after.min()) < float(before.max() - before.min())


def test_dark_neutral_ceiling_keeps_full_target():
    rgb, scene = make_cast_room()
    pass1, _ = global_pass1_wb(rgb, scene)
    tones, _ = classify_tones(pass1, scene, {})
    assert tones["ceiling"]["target"] >= 0.72


def test_material_hue_preserved_saturation_managed():
    # v3.3: hue is the invariant; saturation growth under lift is damped
    # (colorist behavior), so raw channel ratios may compress slightly.
    rgb, scene = make_cast_room(cast=(1.0, 1.0, 1.0))
    tones, _ = classify_tones(rgb, scene, {})
    out, log = adaptive_per_class_exposure(rgb, scene, tones, {"material_inherit_factor": 1.0})
    floor = scene.masks["floor"] > 0.5
    h0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0][floor]
    h1 = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[..., 0][floor]
    assert abs(float(np.median(h1)) - float(np.median(h0))) * 2 < 3.0  # hue < 3 deg
    s0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1][floor]
    s1 = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[..., 1][floor]
    assert float(np.median(s1)) <= float(np.median(s0)) * 1.25  # growth bounded


def test_black_room_no_color_explosion():
    # The z-66 regression test: black walls, dim ceiling, dark red floor.
    H, W = 450, 600
    img = np.zeros((H, W, 3), np.float32)
    rng = np.random.default_rng(7)
    img[:135] = np.array([0.34, 0.32, 0.31]) + rng.normal(0, 0.012, (135, W, 3))
    img[135:315] = np.array([0.045, 0.045, 0.050]) + rng.normal(0, 0.010, (180, W, 3))
    img[315:] = np.array([0.16, 0.09, 0.06]) + rng.normal(0, 0.012, (135, W, 3))
    rgb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    masks = {k: np.zeros((H, W), np.float32) for k in ["ceiling", "wall", "floor"]}
    masks["ceiling"][:135] = 1; masks["wall"][135:315] = 1; masks["floor"][315:] = 1
    scene = build_scene(masks, (H, W))
    tones, _ = classify_tones(rgb, scene, {})
    out, log = adaptive_per_class_exposure(rgb, scene, tones, {"material_inherit_factor": 1.0})
    def chroma_mag(a, m):
        # perceptual color magnitude: LAB chroma, which weights by intensity
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
        c = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
        return float(np.mean(c[m]))
    def lum(a):
        f = a.astype(np.float32) / 255
        return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    y0, y1 = lum(rgb), lum(out)
    wm, fm = masks["wall"] > 0.5, masks["floor"] > 0.5
    assert chroma_mag(out, wm) <= chroma_mag(rgb, wm) + 1.0   # black walls: no color bloom
    assert np.median(y1[wm]) <= np.median(y0[wm]) + 0.05      # black stays black
    assert np.median(y1[fm]) <= np.median(y0[fm]) * 1.55      # floor lift bounded
    assert chroma_mag(out, fm) <= chroma_mag(rgb, fm) * 1.15  # floor: no fire-red


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
