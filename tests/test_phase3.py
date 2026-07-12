
import numpy as np

from engine.scene import build_scene
from engine.tone_classify import classify_tones
from engine.tonal import adaptive_per_class_exposure
from engine.quality2 import evaluate


def make_scene(shape):
    wall = np.zeros(shape, np.float32)
    ceiling = np.zeros(shape, np.float32)
    floor = np.zeros(shape, np.float32)
    cabinet = np.zeros(shape, np.float32)

    wall[40:180, :] = 1
    ceiling[:40, :] = 1
    floor[180:, :] = 1
    cabinet[100:180, 120:220] = 1

    masks = {
        "wall": wall,
        "ceiling": ceiling,
        "floor": floor,
        "cabinet": cabinet,
    }
    return build_scene(masks, shape)


def test_floor_has_no_direct_target():
    shape = (240, 320)
    scene = make_scene(shape)
    image = np.full((240, 320, 3), 100, np.uint8)
    tones, _ = classify_tones(image, scene, {})
    out, log = adaptive_per_class_exposure(
        image,
        scene,
        tones,
        {"material_inherit_factor": 0.08},
    )
    assert log["per_class"]["floor"]["mode"] == "protected_no_direct_target"
    assert log["max_floor_lift"] < 0.06


def test_material_inherit_factor_is_small():
    shape = (240, 320)
    scene = make_scene(shape)
    image = np.full((240, 320, 3), 90, np.uint8)
    tones, _ = classify_tones(image, scene, {})
    _, log = adaptive_per_class_exposure(
        image,
        scene,
        tones,
        {"material_inherit_factor": 0.08},
    )
    assert log["material_inherit_factor"] == 0.08
    assert log["mean_material_inherited_lift"] < 0.05


def test_floor_change_routes_review():
    shape = (240, 320)
    scene = make_scene(shape)
    original = np.full((240, 320, 3), 100, np.uint8)
    final = original.copy()
    final[180:] = 140
    tones, _ = classify_tones(original, scene, {})
    qc = evaluate(original, final, scene, tones)
    assert "FLOOR_BRIGHTNESS_CHANGED" in qc["flags"]
    assert qc["status"] == "REVIEW"
