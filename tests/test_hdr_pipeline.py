from pathlib import Path
from types import MappingProxyType

import numpy as np

from engine.hdr_pipeline import (
    HDRMaster,
    _compress_extreme_highlight_color,
    _hdr_b2_tone_luminance,
    _hdr_c_perceptual_color_tile,
    _hdr_c_source_locked_color_tile,
    _hdr_c_source_referenced_color_tile,
    _linear_srgb_to_oklab,
    _hdr_c_tone_luminance,
    _hdr_b1_tone_luminance,
    analyze_hdr_master,
    measure_hdr_b1_scene,
    master_fingerprint,
    render_hdr_a_foundation,
    render_hdr_b1_global,
    render_hdr_b2_global,
    render_hdr_c_global,
    render_neutral_display,
)


def _master() -> HDRMaster:
    y = np.linspace(-0.002, 4.0, 80 * 120, dtype=np.float32).reshape(80, 120)
    rgb = np.stack([y * 0.9, y, y * 0.8], axis=2)
    xyz = np.stack([y * 0.95, y, y * 1.08], axis=2)
    camera = rgb.copy()
    for array in (camera, xyz, rgb):
        array.setflags(write=False)
    return HDRMaster(
        source=Path("synthetic.dng"),
        camera_linear=camera,
        xyz_d50=xyz,
        linear_srgb=rgb,
        metadata=MappingProxyType({"dng_type": "linear HDR DNG"}),
    )


def test_hdr_stages_preserve_immutable_master():
    master = _master()
    before = master_fingerprint(master)
    analysis = analyze_hdr_master(master)
    neutral = render_neutral_display(master)
    output = render_hdr_a_foundation(master, analysis)

    assert master_fingerprint(master) == before
    assert before["writeable"] == [False, False, False]
    assert neutral.rgb.dtype == np.float32
    assert output.rgb.dtype == np.float32
    assert 0.0 <= float(neutral.rgb.min()) <= float(neutral.rgb.max()) <= 1.0
    assert 0.0 <= float(output.rgb.min()) <= float(output.rgb.max()) <= 1.0


def test_hdr_analysis_uses_scene_linear_headroom():
    analysis = analyze_hdr_master(_master())

    assert analysis.dynamic_range_stops > 8.0
    assert analysis.pixels_above_display_white_percent > 50.0
    assert analysis.pixels_below_zero_percent > 0.0


def test_hdr_a_render_is_global_only():
    master = _master()
    output = render_hdr_a_foundation(master, analyze_hdr_master(master))

    assert output.log["local_corrections"] is False
    assert output.log["semantic_exposure"] is False
    assert output.log["sharpening"] is False
    assert output.log["denoising"] is False


def test_hdr_b1_tone_curve_is_monotonic_and_bounded():
    y = np.linspace(0.0, 12.0, 100_000, dtype=np.float32)
    mapped = _hdr_b1_tone_luminance(
        y,
        exposure_ev=1.2,
        toe=0.02,
        midtone_gamma=0.84,
        shoulder_power=1.6,
    )

    assert mapped[0] == 0.0
    assert np.all(np.diff(mapped) >= 0.0)
    assert float(mapped[-1]) < 1.0


def test_hdr_b1_is_deterministic_global_only_and_preserves_master():
    master = _master()
    before = master_fingerprint(master)
    measurements = measure_hdr_b1_scene(master)
    first = render_hdr_b1_global(master, measurements)
    second = render_hdr_b1_global(master, measurements)

    assert np.array_equal(first.rgb, second.rgb)
    assert first.log["candidate_scores"] == second.log["candidate_scores"]
    assert first.log["local_corrections"] is False
    assert first.log["semantic_exposure"] is False
    assert first.log["sharpening"] is False
    assert first.log["denoising"] is False
    assert first.log["gamut_compression"] is False
    assert master_fingerprint(master) == before


def test_hdr_b1_measurements_report_requested_scene_traits():
    measurements = measure_hdr_b1_scene(_master())

    assert measurements.interior_diffuse_linear > 0.0
    assert 0.0 <= measurements.shadow_occupancy_percent <= 100.0
    assert 0.0 <= measurements.natural_black_occupancy_percent <= 100.0
    assert measurements.window_area_percent == 0.0
    assert measurements.window_to_room_ratio_stops is None
    assert measurements.effective_dynamic_range_stops > 8.0


def test_hdr_b2_curve_is_smooth_monotonic_and_has_black_anchor():
    y = np.linspace(0.0, 16.0, 200_000, dtype=np.float32)
    mapped = _hdr_b2_tone_luminance(
        y,
        exposure_ev=2.2,
        black_anchor_linear=0.001,
        toe_strength=0.9,
        midtone_slope=1.12,
        shoulder_onset=0.7,
    )

    assert mapped[0] == 0.0
    assert np.all(np.diff(mapped) >= 0.0)
    assert float(mapped[-1]) < 1.0
    assert np.all(np.isfinite(np.diff(mapped)))


def test_hdr_b2_highlight_color_compression_targets_only_extremes():
    rgb = np.array([[[0.4, 0.3, 0.2], [1.2, 0.3, 0.8]]], dtype=np.float32)
    y = np.array([[0.3, 0.7]], dtype=np.float32)
    result = _compress_extreme_highlight_color(rgb, y)

    assert np.array_equal(result[0, 0], rgb[0, 0])
    assert float(np.ptp(result[0, 1])) < float(np.ptp(rgb[0, 1]))


def test_hdr_b2_is_deterministic_and_global_only():
    master = _master()
    measurements = measure_hdr_b1_scene(master)
    before = master_fingerprint(master)
    first = render_hdr_b2_global(master, measurements)
    second = render_hdr_b2_global(master, measurements)

    assert np.array_equal(first.rgb, second.rgb)
    assert first.log["candidate_scores"] == second.log["candidate_scores"]
    assert first.log["local_corrections"] is False
    assert first.log["semantic_exposure"] is False
    assert first.log["global_vibrance"] == 0.0
    assert master_fingerprint(master) == before


def test_hdr_c_curve_is_monotonic_with_true_black():
    y = np.linspace(0.0, 16.0, 200_000, dtype=np.float32)
    mapped = _hdr_c_tone_luminance(
        y,
        exposure_ev=3.0,
        black_point_linear=0.002,
        toe_strength=1.2,
        midtone_slope=1.22,
        shoulder=0.8,
    )

    assert mapped[0] == 0.0
    assert float(np.min(np.diff(mapped))) >= -1e-7
    assert float(mapped[-1]) < 1.0


def test_hdr_c_perceptual_color_protects_neutrals_and_maps_gamut():
    rgb = np.array([[[0.4, 0.4, 0.4], [1.3, 0.1, 0.05], [0.3, 0.18, 0.08]]], dtype=np.float32)
    rendered, compressed = _hdr_c_perceptual_color_tile(rgb, richness=1.07)

    assert np.max(np.abs(rendered[0, 0] - 0.4)) < 1e-5
    assert 0.0 <= float(rendered.min()) <= float(rendered.max()) <= 1.0
    assert compressed > 0.0


def test_hdr_c_revised_color_locks_source_hue_and_absolute_chroma():
    source = np.array([[[0.08, 0.035, 0.018], [0.4, 0.4, 0.4]]], dtype=np.float32)
    mapped = source * 3.0
    rendered, _ = _hdr_c_source_locked_color_tile(mapped, source)
    source_lab = _linear_srgb_to_oklab(source)
    rendered_lab = _linear_srgb_to_oklab(rendered)
    source_hue = np.arctan2(source_lab[..., 2], source_lab[..., 1])
    rendered_hue = np.arctan2(rendered_lab[..., 2], rendered_lab[..., 1])
    source_chroma = np.hypot(source_lab[..., 1], source_lab[..., 2])
    rendered_chroma = np.hypot(rendered_lab[..., 1], rendered_lab[..., 2])

    assert abs(float(source_hue[0, 0] - rendered_hue[0, 0])) < 1e-5
    assert abs(float(source_chroma[0, 0] - rendered_chroma[0, 0])) < 1e-5
    assert np.max(np.abs(rendered[0, 1] - rendered[0, 1, 0])) < 1e-5


def test_hdr_c_bounded_color_locks_hue_and_protects_neutrals_and_dark_wood():
    source = np.array([[[0.050, 0.040, 0.030], [0.4, 0.4, 0.4], [0.30, 0.13, 0.05]]], dtype=np.float32)
    mapped = source * 3.0
    rendered, stats = _hdr_c_source_referenced_color_tile(mapped, source)
    source_lab = _linear_srgb_to_oklab(source)
    rendered_lab = _linear_srgb_to_oklab(rendered)
    source_hue = np.arctan2(source_lab[..., 2], source_lab[..., 1])
    rendered_hue = np.arctan2(rendered_lab[..., 2], rendered_lab[..., 1])
    source_chroma = np.hypot(source_lab[..., 1], source_lab[..., 2])
    rendered_chroma = np.hypot(rendered_lab[..., 1], rendered_lab[..., 2])

    assert abs(float(source_hue[0, 0] - rendered_hue[0, 0])) < 1e-4
    assert float(rendered_chroma[0, 0] / source_chroma[0, 0]) < 1.03
    assert np.max(np.abs(rendered[0, 1] - rendered[0, 1, 0])) < 1e-5
    assert 1.0 < float(rendered_chroma[0, 2] / source_chroma[0, 2]) <= 1.11
    assert 0.0 <= stats["maximum_chroma_enhancement_percent"] <= 11.01


def test_hdr_c_is_deterministic_and_global_only():
    master = _master()
    measurements = measure_hdr_b1_scene(master)
    before = master_fingerprint(master)
    first = render_hdr_c_global(master, measurements)
    second = render_hdr_c_global(master, measurements)

    assert np.array_equal(first.rgb, second.rgb)
    assert first.log["candidate_scores"] == second.log["candidate_scores"]
    assert first.log["local_corrections"] is False
    assert first.log["semantic_exposure"] is False
    assert master_fingerprint(master) == before
