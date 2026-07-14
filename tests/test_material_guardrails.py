import cv2
import numpy as np

from engine.materials2 import apply_material_guardrails
from engine.scene import build_scene


def _scene(h=120, w=160):
    masks = {name: np.zeros((h, w), np.float32) for name in [
        "wall", "ceiling", "floor", "cabinet", "sofa"
    ]}
    masks["ceiling"][: h // 3] = 1.0
    masks["wall"][h // 3 : 2 * h // 3] = 1.0
    masks["floor"][2 * h // 3 :] = 1.0
    masks["cabinet"][h // 2 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    return build_scene(masks, (h, w))


def _luma(rgb):
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def test_dark_wall_lift_is_bounded():
    h, w = 120, 160
    ref = np.full((h, w, 3), [35, 34, 33], np.uint8)
    corrected = np.full((h, w, 3), [100, 99, 98], np.uint8)
    out, log = apply_material_guardrails(ref, corrected, _scene(h, w))
    wall = _scene(h, w).masks["wall"] > 0.5
    lift = np.mean(_luma(out)[wall] - _luma(ref)[wall])
    assert lift < 0.075
    assert log["engine"] == "material_dark_surface_guardrails_v2"


def test_floor_hue_moves_toward_reference():
    h, w = 120, 160
    ref = np.full((h, w, 3), [90, 55, 30], np.uint8)
    corrected = np.full((h, w, 3), [105, 80, 70], np.uint8)
    scene = _scene(h, w)
    out, _ = apply_material_guardrails(ref, corrected, scene)
    floor = scene.masks["floor"] > 0.5
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(float)
    corr_lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(float)
    out_lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(float)
    before = np.mean(np.linalg.norm(corr_lab[floor, 1:3] - ref_lab[floor, 1:3], axis=1))
    after = np.mean(np.linalg.norm(out_lab[floor, 1:3] - ref_lab[floor, 1:3], axis=1))
    assert after < before


def test_neutral_ceiling_does_not_gain_excess_chroma():
    h, w = 120, 160
    ref = np.full((h, w, 3), [170, 170, 170], np.uint8)
    corrected = np.full((h, w, 3), [190, 160, 160], np.uint8)
    scene = _scene(h, w)
    out, _ = apply_material_guardrails(ref, corrected, scene)
    ceiling = scene.masks["ceiling"] > 0.5
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(float)
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    assert float(np.mean(chroma[ceiling])) <= 4.5
