
import numpy as np

from engine.scene import Scene, build_scene
from engine.tone_classify import classify_tones
from engine.tonal import adaptive_per_class_exposure
from engine.quality2 import evaluate
from engine.windows2 import treat_window_zones


def scene_with(shape, **named_masks):
    masks = {k: np.zeros(shape, np.float32) for k in [
        "wall","ceiling","floor","window","curtain","door","cabinet","mirror",
        "sofa","table","chair","bed","rug","plant","painting","lamp"
    ]}
    masks.update(named_masks)
    return build_scene(masks, shape)


def test_dark_wall_not_grayed():
    shape = (300, 400)
    wall = np.zeros(shape, np.float32); wall[:, :250] = 1
    ceiling = np.zeros(shape, np.float32); ceiling[:60] = 1
    floor = np.zeros(shape, np.float32); floor[220:] = 1
    scene = scene_with(shape, wall=wall, ceiling=ceiling, floor=floor)
    img = np.full((300, 400, 3), 31, np.uint8)
    tones, _ = classify_tones(img, scene, {})
    assert tones["wall_components"][0]["class"] == "DARK"
    assert tones["wall_components"][0]["target"] <= 0.27
    out, _ = adaptive_per_class_exposure(img, scene, tones, {"furnishing_protection":0.35})
    assert np.median(out[:, :250]) / 255.0 <= 0.30
    qc = evaluate(img, out, scene, tones)
    assert "WALL_OVERSHOT" not in qc["flags"]


def test_accent_wall_split():
    shape = (300, 500)
    wall = np.zeros(shape, np.float32)
    wall[:, :180] = 1
    wall[:, 320:] = 1
    ceiling = np.zeros(shape, np.float32); ceiling[:40] = 1
    floor = np.zeros(shape, np.float32); floor[250:] = 1
    scene = scene_with(shape, wall=wall, ceiling=ceiling, floor=floor)
    img = np.full((300, 500, 3), 140, np.uint8)
    img[:, :180] = 38
    tones, _ = classify_tones(img, scene, {})
    classes = {x["class"] for x in tones["wall_components"]}
    assert "DARK" in classes and "LIGHT" in classes


def test_dark_floor_keeps_depth():
    shape = (300, 400)
    floor = np.zeros(shape, np.float32); floor[160:] = 1
    wall = np.zeros(shape, np.float32); wall[40:160] = 1
    ceiling = np.zeros(shape, np.float32); ceiling[:40] = 1
    scene = scene_with(shape, wall=wall, ceiling=ceiling, floor=floor)
    img = np.full((300, 400, 3), 46, np.uint8)
    tones, _ = classify_tones(img, scene, {})
    assert tones["floor"]["class"] == "DARK"
    assert tones["floor"]["target"] <= 0.31


def test_mirror_never_window():
    shape = (200, 300)
    window = np.zeros(shape, np.float32); window[30:170, 60:240] = 1
    mirror = window.copy()
    wall = np.ones(shape, np.float32)
    scene = scene_with(shape, wall=wall, window=window, mirror=mirror)
    img = np.full((200, 300, 3), 245, np.uint8)
    out, log = treat_window_zones(img, scene)
    assert abs(float(out[100,150].mean()) - float(img[100,150].mean())) < 3
    assert log["mirror_excluded_percent"] > 1
