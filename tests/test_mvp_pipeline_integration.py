import numpy as np

from engine.scene import build_scene
from engine.mvp_pipeline import process_mvp_core


def _synthetic_scene(h: int, w: int):
    masks = {
        "ceiling": np.zeros((h, w), np.float32),
        "wall": np.zeros((h, w), np.float32),
        "floor": np.zeros((h, w), np.float32),
        "cabinet": np.zeros((h, w), np.float32),
    }
    masks["ceiling"][: h // 3] = 1.0
    masks["wall"][h // 3 : 2 * h // 3] = 1.0
    masks["floor"][2 * h // 3 :] = 1.0
    masks["cabinet"][h // 2 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    return build_scene(masks, (h, w))


def test_mvp_core_runs_single_wb_then_exposure_fusion():
    h, w = 240, 320
    image = np.empty((h, w, 3), np.uint8)
    image[: h // 3] = [135, 120, 100]
    image[h // 3 : 2 * h // 3] = [92, 82, 70]
    image[2 * h // 3 :] = [72, 48, 31]

    result = process_mvp_core(image, _synthetic_scene(h, w), {"targets": {}, "phase3": {}})

    assert result["protected"].shape == image.shape
    assert result["protected"].dtype == np.uint8
    assert result["exposure_log"]["engine"] == "scene_aware_mertens_exposure_fusion_v3"
    assert result["window_log"]["status"] == "skipped_mvp_phase3"
    assert "gains" in result["wb_log"]


def test_mvp_material_guardrail_uses_wb_corrected_reference():
    h, w = 180, 240
    image = np.full((h, w, 3), [100, 85, 70], np.uint8)
    scene = _synthetic_scene(h, w)

    result = process_mvp_core(image, scene, {"targets": {}, "phase3": {}})

    # The guardrail is active, but cannot undo the white-balance pass because
    # its source is the WB-corrected image rather than the raw input.
    assert result["materials_log"]["floor_included"] is True
    assert result["wb"].shape == result["protected"].shape
