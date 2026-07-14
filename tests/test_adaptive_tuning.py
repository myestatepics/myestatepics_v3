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
    before = rgb.astype(np.float32).mean(axis=(0, 1))
    after = out.astype(np.float32).mean(axis=(0, 1))
    before_ratio = before / before.sum()
    after_ratio = after / after.sum()
    assert np.max(np.abs(before_ratio - after_ratio)) < 0.035
    assert log["adaptive"]["adaptive_profile"] in {"dim_interior", "balanced_interior"}


def test_bright_window_remains_guarded():
    rgb = np.full((160, 220, 3), 70, dtype=np.uint8)
    rgb[20:120, 150:215] = 250
    out, _ = exposure_fusion(rgb)
    before_clip = np.mean(rgb >= 251)
    after_clip = np.mean(out >= 251)
    assert after_clip <= before_clip + 0.005
