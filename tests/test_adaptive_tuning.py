import cv2
import numpy as np

from engine.exposure_fusion import adaptive_exposure_config, exposure_fusion


def test_dark_room_selects_stronger_profile():
    rgb = np.full((120, 160, 3), 45, dtype=np.uint8)
    cfg, log = adaptive_exposure_config(rgb)
    assert log["adaptive_profile"] == "dark_interior"
    assert cfg.target_median >= 0.53
    assert cfg.final_strength >= 0.90


def test_balanced_room_stays_conservative():
    rgb = np.full((120, 160, 3), 135, dtype=np.uint8)
    cfg, log = adaptive_exposure_config(rgb)
    assert log["adaptive_profile"] == "balanced_interior"
    assert cfg.final_strength <= 0.84
    assert cfg.max_bright_ev <= 1.15


def test_local_contrast_does_not_shift_uniform_warm_hue():
    rgb = np.full((120, 160, 3), [100, 65, 38], dtype=np.uint8)
    out, log = exposure_fusion(rgb)
    before = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    after = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    assert np.max(np.abs(before[..., 0] - after[..., 0])) <= 1.0
    assert np.max(np.abs(before[..., 1] - after[..., 1])) <= 2.0
    assert log["adaptive"]["adaptive_profile"] in {"dim_interior", "balanced_interior"}


def test_bright_window_remains_guarded():
    rgb = np.full((160, 220, 3), 70, dtype=np.uint8)
    rgb[20:120, 150:215] = 250
    out, _ = exposure_fusion(rgb)
    before_clip = np.mean(rgb >= 251)
    after_clip = np.mean(out >= 251)
    assert after_clip <= before_clip + 0.005
