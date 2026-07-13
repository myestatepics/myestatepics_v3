import numpy as np

from engine.scene import build_scene
from engine.profiles import detect_room_profile


def _scene_with(cov_masks: dict, H=300, W=400):
    masks = {k: np.zeros((H, W), np.float32) for k in
             ["ceiling", "wall", "floor", "cabinet", "mirror", "sofa", "bed"]}
    masks["ceiling"][:90] = 1.0
    masks["wall"][90:210] = 1.0
    masks["floor"][210:] = 1.0
    for k, frac in cov_masks.items():
        masks[k][:, : int(W * frac)] = 1.0
    return build_scene(masks, (H, W))


def _img(level, H=300, W=400):
    return np.full((H, W, 3), int(level * 255), np.uint8)


def test_kitchen_detected():
    scene = _scene_with({"cabinet": 0.30})
    name, _, _ = detect_room_profile(scene, _img(0.45))
    assert name == "kitchen"


def test_dark_room_vs_low_light():
    # black walls + light ceiling = dark decor
    H, W = 300, 400
    img = np.full((H, W, 3), 12, np.uint8)
    img[:90] = 90  # ceiling much lighter than walls
    scene = _scene_with({})
    name, _, _ = detect_room_profile(scene, img)
    assert name == "dark_room"
    # uniformly dim room = low light, not dark decor
    name2, targets2, dyn2 = detect_room_profile(scene, _img(0.10))
    assert name2 == "low_light"
    assert dyn2["ambient_cap_ratio"] > 1.5


def test_generic_bright_room():
    scene = _scene_with({})
    name, _, _ = detect_room_profile(scene, _img(0.55))
    assert name == "generic"
