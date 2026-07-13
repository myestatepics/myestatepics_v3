import cv2
import numpy as np

from engine.tonal import global_safe_exposure


def _analysis(profile: str = "generic") -> dict:
    return {
        "shadow_lift": 0.30,
        "midtone_lift": 0.24,
        "room_profile": profile,
    }


def _luma(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _lab_chroma(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)


def test_global_safe_reaches_mls_brightness_without_darkening():
    rgb = np.full((180, 240, 3), [88, 78, 70], dtype=np.uint8)
    out, log = global_safe_exposure(rgb, _analysis("generic"))
    assert _luma(out).mean() > _luma(rgb).mean() + 0.12
    assert log["convergence"]["target"] == 0.55


def test_global_safe_holds_warm_material_chroma():
    rgb = np.full((180, 240, 3), [92, 58, 34], dtype=np.uint8)
    out, _ = global_safe_exposure(rgb, _analysis("generic"))
    before = float(np.mean(_lab_chroma(rgb)))
    after = float(np.mean(_lab_chroma(out)))
    assert after <= before * 1.03


def test_dark_room_keeps_true_black_anchored_but_lifts_floor_tones():
    rgb = np.zeros((180, 240, 3), dtype=np.uint8)
    rgb[:90] = [8, 8, 9]
    rgb[90:] = [42, 24, 18]
    out, _ = global_safe_exposure(rgb, _analysis("dark_room"))
    y0, y1 = _luma(rgb), _luma(out)
    assert float(np.mean(y1[:90])) <= float(np.mean(y0[:90])) + 0.015
    assert float(np.mean(y1[90:])) > float(np.mean(y0[90:])) + 0.035
