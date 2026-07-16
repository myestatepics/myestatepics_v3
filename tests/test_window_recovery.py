import numpy as np

from engine.scene import build_scene
from engine.window_recovery import generate_mls_sky, recover_windows


NAMES = ("wall", "ceiling", "floor", "window", "curtain", "door", "cabinet",
         "mirror", "sofa", "table", "chair", "bed", "rug", "plant", "painting", "lamp")


def _scene(h=160, w=240, *, curtain=False, mirror=False, windows=1):
    masks = {name: np.zeros((h, w), np.float32) for name in NAMES}
    masks["wall"][:] = 1.0
    if windows == 1:
        masks["window"][20:140, 50:190] = 1.0
    elif windows == 2:
        masks["window"][20:140, 20:100] = 1.0
        masks["window"][20:140, 140:220] = 1.0
    masks["wall"] *= 1.0 - masks["window"]
    if curtain:
        masks["curtain"][20:140, 50:85] = 1.0
    if mirror:
        masks["mirror"] = masks["window"].copy()
    return build_scene(masks, (h, w))


def test_no_window_is_exact_noop():
    scene = _scene(windows=0)
    rendered = np.full((160, 240, 3), 0.6, np.float32)
    output, log = recover_windows(rendered, rendered, scene, seed="none")
    assert np.array_equal(output, rendered)
    assert log["status"] == "no_windows"


def test_recovery_preserves_geometry_frames_and_outside_pixels():
    h, w = 160, 240
    scene = _scene(h, w)
    rendered = np.full((h, w, 3), 0.45, np.float32)
    rendered[20:140, 50:190] = 0.97
    rendered[20:140, 116:122] = 0.12  # mullion
    linear = np.full_like(rendered, 0.12)
    linear[20:140, 50:190, 1] = np.linspace(0.05, 1.2, 140)[None, :]
    output, log = recover_windows(linear, rendered, scene, seed="frames")
    outside = scene.masks["window"] < 0.01
    assert output.shape == rendered.shape
    assert np.array_equal(output[outside], rendered[outside])
    assert np.mean(np.abs(output[20:140, 116:122] - rendered[20:140, 116:122])) < 0.01
    assert log["blue_spill_percent"] == 0.0


def test_curtain_and_mirror_are_rejected():
    rendered = np.full((160, 240, 3), 0.92, np.float32)
    linear = np.full_like(rendered, 0.2)
    curtain_scene = _scene(curtain=True)
    curtain = curtain_scene.masks["curtain"] > 0.9
    output, _ = recover_windows(linear, rendered, curtain_scene, seed="curtain")
    assert np.max(np.abs(output[curtain] - rendered[curtain])) < 1e-6
    mirror_scene = _scene(mirror=True)
    mirror_output, log = recover_windows(linear, rendered, mirror_scene, seed="mirror")
    assert np.array_equal(mirror_output, rendered)
    assert log["status"] == "no_windows"


def test_unrecoverable_upper_window_gets_light_sky_only():
    scene = _scene()
    rendered = np.full((160, 240, 3), 0.45, np.float32)
    rendered[20:140, 50:190] = 0.99
    linear = np.full_like(rendered, 0.08)
    linear[20:140, 50:190] = 4.0
    output, log = recover_windows(linear, rendered, scene, seed="sky")
    upper = output[30:65, 70:170]
    lower = output[105:130, 70:170]
    assert log["unrecoverable_sky_percent"] > 0.0
    assert float(np.mean(upper[..., 2] - upper[..., 0])) > 0.08
    assert float(np.mean(lower[..., 2] - lower[..., 0])) < float(np.mean(upper[..., 2] - upper[..., 0]))


def test_multi_window_sky_is_deterministic_and_consistent():
    scene = _scene(windows=2)
    rendered = np.full((160, 240, 3), 0.98, np.float32)
    linear = np.full_like(rendered, 3.0)
    first, _ = recover_windows(linear, rendered, scene, seed="room")
    second, _ = recover_windows(linear, rendered, scene, seed="room")
    assert np.array_equal(first, second)
    sky_a = generate_mls_sky((160, 240), "room")
    sky_b = generate_mls_sky((160, 240), "other")
    assert not np.array_equal(sky_a, sky_b)
