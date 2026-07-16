import numpy as np

from engine.photographic_finish import apply_lightroom_look
from engine.window_recovery import recover_windows
from validation.synthetic_dataset import CASES, make_fixture


def test_internal_dataset_contains_all_fifty_required_cases():
    assert len(CASES) == 50
    assert len({case.name for case in CASES}) == 50


def test_all_synthetic_cases_preserve_geometry_and_bounds():
    for case in CASES:
        rendered, linear, scene = make_fixture(case, (96, 144))
        finished, _ = apply_lightroom_look(rendered, scene)
        output, log = recover_windows(linear, finished, scene, seed=case.name)
        assert output.shape == rendered.shape, case.name
        assert np.isfinite(output).all(), case.name
        assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0, case.name
        assert log["blue_spill_percent"] == 0.0, case.name


def test_protected_neutrals_and_no_window_cases_are_stable():
    for case in CASES:
        rendered, linear, scene = make_fixture(case, (96, 144))
        finished, _ = apply_lightroom_look(rendered, scene)
        output, _ = recover_windows(linear, finished, scene, seed=case.name)
        if "no_window" in case.tags or "batch" in case.tags:
            assert np.array_equal(output, finished), case.name
        neutral = (scene.masks["wall"] > 0.9) | (scene.masks["ceiling"] > 0.9)
        if np.any(neutral):
            before = rendered[neutral]
            after = output[neutral]
            before_chroma = before / np.maximum(before.sum(axis=1, keepdims=True), 1e-6)
            after_chroma = after / np.maximum(after.sum(axis=1, keepdims=True), 1e-6)
            assert float(np.percentile(np.abs(after_chroma - before_chroma), 99)) < 2e-4, case.name
