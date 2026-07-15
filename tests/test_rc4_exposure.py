import cv2
import numpy as np

from engine.scene import build_scene
from engine.tonal import _luminance, source_referenced_exposure


def _scene(h: int = 240, w: int = 320):
    masks = {name: np.zeros((h, w), np.float32) for name in (
        "wall", "ceiling", "floor", "window", "lamp", "cabinet", "sofa",
        "table", "chair", "bed", "rug",
    )}
    masks["ceiling"][: h // 4] = 1.0
    masks["wall"][h // 4 : 2 * h // 3] = 1.0
    masks["floor"][2 * h // 3 :] = 1.0
    masks["cabinet"][h // 3 : 3 * h // 4, : w // 4] = 1.0
    masks["window"][h // 4 : h // 2, 3 * w // 4 :] = 1.0
    masks["lamp"][: h // 8, w // 2 - 10 : w // 2 + 10] = 1.0
    return build_scene(masks, (h, w))


def _lab_ab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1:3].astype(np.int16)


def test_already_good_image_is_near_noop():
    rgb = np.full((240, 320, 3), [154, 151, 147], np.uint8)
    out, log = source_referenced_exposure(rgb, _scene())
    assert float(np.mean(np.abs(out.astype(np.int16) - rgb.astype(np.int16)))) < 1.0
    assert log["median_room_luminance_after"] >= log["median_room_luminance_before"]


def test_dark_room_lifts_readable_surfaces_but_holds_black_wall():
    rgb = np.full((240, 320, 3), [66, 59, 52], np.uint8)
    rgb[60:160, :80] = [8, 8, 9]
    out, _ = source_referenced_exposure(rgb, _scene())
    before, after = _luminance(rgb), _luminance(out)
    assert float(np.median(after[170:])) > float(np.median(before[170:])) + 0.035
    assert float(np.mean(after[70:150, :70] - before[70:150, :70])) < 0.012


def test_windows_and_highlights_are_preserved():
    rgb = np.full((240, 320, 3), [70, 65, 58], np.uint8)
    rgb[60:120, 240:] = [248, 250, 252]
    out, log = source_referenced_exposure(rgb, _scene())
    assert np.array_equal(out[65:115, 245:], rgb[65:115, 245:])
    assert log["window_core_mean_abs_change"] == 0.0


def test_exposure_preserves_hue_chroma_and_geometry():
    rgb = np.full((240, 320, 3), [82, 49, 28], np.uint8)
    out, _ = source_referenced_exposure(rgb, _scene())
    assert out.shape == rgb.shape
    delta_ab = np.abs(_lab_ab(out).astype(np.int16) - _lab_ab(rgb).astype(np.int16))
    assert float(np.percentile(delta_ab, 99)) <= 2.0


def test_convergence_is_bounded_to_one_corrective_pass():
    rgb = np.full((240, 320, 3), [54, 50, 46], np.uint8)
    out, log = source_referenced_exposure(rgb, _scene())
    assert log["maximum_passes"] == 2
    assert log["passes_executed"] in {1, 2}
    assert log["median_room_luminance_after"] >= log["median_room_luminance_before"]
    assert float(np.median(_luminance(out))) >= float(np.median(_luminance(rgb)))
