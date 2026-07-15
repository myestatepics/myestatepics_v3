import numpy as np

from engine.photographic_finish import apply_lightroom_look, _srgb_to_linear
from engine.scene import build_scene


def _scene(h: int, w: int):
    masks = {name: np.zeros((h, w), np.float32) for name in (
        "wall", "ceiling", "floor", "window", "curtain", "door", "cabinet",
        "mirror", "sofa", "table", "chair", "bed", "rug", "plant",
        "painting", "lamp",
    )}
    masks["floor"][h // 2 :, :] = 1.0
    masks["wall"][: h // 2, : w // 2] = 1.0
    masks["window"][: h // 2, w // 2 :] = 1.0
    masks["curtain"][: h // 2, w // 2 : 3 * w // 4] = 1.0
    masks["lamp"][: h // 5, : w // 5] = 1.0
    masks["ceiling"][: h // 5, w // 3 : 2 * w // 3] = 1.0
    return build_scene(masks, (h, w))


def _linear_luma(rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


def test_finish_preserves_geometry_and_linear_rgb_ratios():
    h, w = 120, 160
    x = np.linspace(0.12, 0.72, w, dtype=np.float32)
    rgb = np.stack(np.broadcast_arrays(x[None, :], x[None, :] * 0.72, x[None, :] * 0.48), axis=2)
    rgb = np.repeat(rgb, h, axis=0)
    output, log = apply_lightroom_look(rgb, _scene(h, w))
    before, after = _srgb_to_linear(rgb), _srgb_to_linear(output)
    valid = np.min(before, axis=2) > 1e-4

    assert output.shape == rgb.shape
    assert np.max(np.abs(after[..., 0][valid] / after[..., 1][valid] - before[..., 0][valid] / before[..., 1][valid])) < 2e-5
    assert log["luminance_only"] is True


def test_finish_increases_floor_detail_but_leaves_wall_unchanged():
    h, w = 160, 200
    yy, xx = np.indices((h, w))
    texture = 0.42 + 0.025 * np.sin(xx * 0.9) * np.sin(yy * 0.7)
    rgb = np.repeat(texture[..., None], 3, axis=2).astype(np.float32)
    scene = _scene(h, w)
    output, _ = apply_lightroom_look(rgb, scene)
    before_y, after_y = _linear_luma(rgb), _linear_luma(output)
    floor = scene.masks["floor"] > 0.9
    wall = scene.masks["wall"] > 0.9

    assert np.std(cv2_laplacian(after_y)[floor]) > np.std(cv2_laplacian(before_y)[floor])
    assert np.max(np.abs(output[wall] - rgb[wall])) < 2e-5


def cv2_laplacian(image: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.Laplacian(image.astype(np.float32), cv2.CV_32F, ksize=3)


def test_black_anchor_and_highlight_rolloffs_are_bounded():
    h, w = 100, 140
    rgb = np.full((h, w, 3), 0.42, np.float32)
    rgb[: h // 2, w // 2 :] = 0.96
    rgb[: h // 5, : w // 5] = 0.98
    rgb[: h // 5, w // 3 : 2 * w // 3] = 0.98
    scene = _scene(h, w)
    output, _ = apply_lightroom_look(rgb, scene)
    before_y, after_y = _linear_luma(rgb), _linear_luma(output)
    floor = scene.masks["floor"] > 0.9
    curtain = scene.masks["curtain"] > 0.9
    bare_window = (scene.masks["window"] > 0.9) & ~curtain
    lamp = scene.masks["lamp"] > 0.9
    ceiling_light = scene.masks["ceiling"] > 0.9

    assert 0.96 < float(np.median(after_y[floor] / before_y[floor])) < 1.0
    assert float(np.median(after_y[bare_window] / before_y[bare_window])) < 0.93
    assert np.max(np.abs(output[curtain] - rgb[curtain])) < 2e-5
    assert 0.94 < float(np.median(after_y[lamp] / before_y[lamp])) < 1.0
    assert 0.94 < float(np.median(after_y[ceiling_light] / before_y[ceiling_light])) < 1.0
