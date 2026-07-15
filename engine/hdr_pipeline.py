from __future__ import annotations

"""HDR-A scene-linear foundation.

This module intentionally contains no local corrections. It establishes four
independent contracts: immutable scene-linear master, analysis, rendering, and
QA. Window pull, sky work, semantic exposure, sharpening, and denoising do not
belong here.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .dng import read_lightroom_hdr_dng


@dataclass(frozen=True)
class HDRMaster:
    source: Path
    camera_linear: np.ndarray
    xyz_d50: np.ndarray
    linear_srgb: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class HDRAnalysis:
    luminance_percentiles: Mapping[str, float]
    log2_luminance_percentiles: Mapping[str, float]
    room_median_linear: float
    room_median_stops: float
    noise_floor_linear: float
    highlight_p99_9_linear: float
    dynamic_range_stops: float
    pixels_above_display_white_percent: float
    pixels_below_zero_percent: float
    window_headroom_stops_p99: float | None


@dataclass(frozen=True)
class HDRRender:
    rgb: np.ndarray
    log: Mapping[str, Any]


@dataclass(frozen=True)
class HDRB1Measurements:
    """Robust scene measurements used by the global HDR-B1 planner."""

    interior_diffuse_linear: float
    photographic_black_linear: float
    shadow_occupancy_percent: float
    natural_black_occupancy_percent: float
    window_area_percent: float
    window_luminance_linear: float | None
    window_to_room_ratio_stops: float | None
    highlight_headroom_stops: float
    effective_dynamic_range_stops: float
    usable_interior_percent: float
    measurement_confidence: float


def _immutable(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=np.float32)
    result.setflags(write=False)
    return result


def load_hdr_master(path: str | Path) -> HDRMaster:
    decoded = read_lightroom_hdr_dng(path)
    return HDRMaster(
        source=Path(path).resolve(),
        camera_linear=_immutable(decoded.camera_linear),
        xyz_d50=_immutable(decoded.xyz_d50),
        linear_srgb=_immutable(decoded.linear_srgb),
        metadata=MappingProxyType(dict(decoded.metadata)),
    )


def _percentile_map(values: np.ndarray) -> dict[str, float]:
    levels = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
    measured = np.percentile(values, levels)
    return {f"p{str(level).replace('.', '_')}": float(value) for level, value in zip(levels, measured)}


def analyze_hdr_master(master: HDRMaster, scene=None) -> HDRAnalysis:
    """Measure the immutable master in scene-linear/log-stop domains."""
    y = master.xyz_d50[..., 1].astype(np.float32)
    positive = y[y > 1e-8]
    if positive.size < 1024:
        raise ValueError("HDR master contains insufficient positive luminance samples")
    noise_floor = float(np.percentile(positive, 0.1))
    highlight = float(np.percentile(positive, 99.9))
    dynamic_range = float(np.log2(highlight / max(noise_floor, 1e-12)))

    room = np.ones(y.shape, dtype=bool)
    window_headroom = None
    if scene is not None:
        structure = scene.masks.get("structure")
        window = scene.masks.get("window")
        if structure is not None:
            room = structure > 0.45
        if window is not None:
            room &= window < 0.25
            window_values = y[window > 0.5]
            window_values = window_values[window_values > 1e-8]
            if window_values.size >= 256:
                window_p99 = float(np.percentile(window_values, 99))
                window_headroom = float(np.log2(max(window_p99, 1e-8)))
    room &= y > max(noise_floor, 1e-8)
    if np.count_nonzero(room) < 1024:
        room = y > max(noise_floor, 1e-8)
    room_median = float(np.median(y[room]))

    return HDRAnalysis(
        luminance_percentiles=MappingProxyType(_percentile_map(positive)),
        log2_luminance_percentiles=MappingProxyType(_percentile_map(np.log2(positive))),
        room_median_linear=room_median,
        room_median_stops=float(np.log2(max(room_median, 1e-12))),
        noise_floor_linear=noise_floor,
        highlight_p99_9_linear=highlight,
        dynamic_range_stops=dynamic_range,
        pixels_above_display_white_percent=float(np.mean(y > 1.0) * 100.0),
        pixels_below_zero_percent=float(np.mean(y < 0.0) * 100.0),
        window_headroom_stops_p99=window_headroom,
    )


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    positive = np.maximum(linear, 0.0)
    return np.where(
        positive <= 0.0031308,
        12.92 * positive,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _ratio_preserving_display_shoulder(
    linear_rgb: np.ndarray,
    *,
    start: float = 0.60,
) -> np.ndarray:
    """Map extended highlights to display white with one global monotonic curve."""
    positive = np.maximum(linear_rgb.astype(np.float32), 0.0)
    peak = np.max(positive, axis=2)
    compressed_peak = np.where(
        peak <= start,
        peak,
        start + (1.0 - start) * (1.0 - np.exp(-(peak - start) / (1.0 - start))),
    )
    scale = compressed_peak / np.maximum(peak, 1e-8)
    return np.clip(positive * scale[..., None], 0.0, 1.0)


def render_neutral_display(master: HDRMaster) -> HDRRender:
    """Faithful viewing render; not an MLS edit."""
    display_linear = _ratio_preserving_display_shoulder(master.linear_srgb)
    rgb = _linear_to_srgb(display_linear)
    return HDRRender(
        rgb=rgb,
        log=MappingProxyType(
            {
                "stage": "neutral_display",
                "creative_edit": False,
                "global_exposure_ev": 0.0,
                "display_shoulder_start_linear": 0.60,
                "local_corrections": False,
                "sharpening": False,
                "denoising": False,
                "output_space": "sRGB",
                "working_precision": "float32",
            }
        ),
    )


def render_hdr_a_foundation(
    master: HDRMaster,
    analysis: HDRAnalysis,
    *,
    diffuse_room_target_linear: float = 0.10,
    maximum_lift_ev: float = 2.0,
) -> HDRRender:
    """Provisional HDR-A global render used only to validate the foundation.

    This is not the HDR-B MLS exposure model. It places the measured room
    median using one true scene-linear scalar and the same global display
    shoulder used by the neutral render.
    """
    requested_ev = float(
        np.log2(
            max(diffuse_room_target_linear, 1e-8)
            / max(analysis.room_median_linear, 1e-8)
        )
    )
    applied_ev = float(np.clip(requested_ev, -1.0, maximum_lift_ev))
    placed = master.linear_srgb * float(2.0**applied_ev)
    display_linear = _ratio_preserving_display_shoulder(placed)
    rgb = _linear_to_srgb(display_linear)
    return HDRRender(
        rgb=rgb,
        log=MappingProxyType(
            {
                "stage": "hdr_a_foundation_render",
                "purpose": "validate scene-linear boundary; not final HDR-B MLS model",
                "diffuse_room_target_linear": diffuse_room_target_linear,
                "requested_global_exposure_ev": requested_ev,
                "applied_global_exposure_ev": applied_ev,
                "maximum_lift_ev": maximum_lift_ev,
                "display_shoulder_start_linear": 0.60,
                "local_corrections": False,
                "semantic_exposure": False,
                "sharpening": False,
                "denoising": False,
                "output_space": "sRGB",
                "working_precision": "float32",
            }
        ),
    )


def measure_hdr_b1_scene(master: HDRMaster, scene=None) -> HDRB1Measurements:
    """Measure photographic scene traits without creating rendering masks.

    Semantic masks only select robust analysis populations. They never alter
    the image or produce spatially varying rendering parameters.
    """
    y = master.xyz_d50[..., 1].astype(np.float32)
    finite = np.isfinite(y) & (y > 1e-8)
    positive = y[finite]
    if positive.size < 1024:
        raise ValueError("HDR master contains insufficient usable luminance")

    noise_floor = float(np.percentile(positive, 0.1))
    usable = finite & (y > noise_floor)
    structure = None if scene is None else scene.masks.get("structure")
    window = None if scene is None else scene.masks.get("window")
    interior = usable.copy()
    confidence = 0.45
    if structure is not None and np.count_nonzero(structure > 0.35) >= 1024:
        interior &= structure > 0.35
        confidence += 0.30
    if window is not None:
        interior &= window < 0.25
        confidence += 0.15
    if np.count_nonzero(interior) < 1024:
        interior = usable & ((window < 0.25) if window is not None else True)
        confidence = min(confidence, 0.45)

    interior_values = y[interior]
    lo, hi = np.percentile(interior_values, (10, 85))
    diffuse_values = interior_values[(interior_values >= lo) & (interior_values <= hi)]
    if diffuse_values.size < 1024:
        diffuse_values = interior_values
    # A trimmed geometric center is stable across deep shadows and bright trim.
    diffuse = float(2.0 ** np.mean(np.log2(np.maximum(diffuse_values, 1e-8))))
    dark_usable = interior_values[interior_values > noise_floor * 1.5]
    photographic_black = float(
        np.percentile(dark_usable, 1.0)
        if dark_usable.size >= 1024
        else np.percentile(interior_values, 1.0)
    )

    shadow_occupancy = float(np.mean(interior_values < diffuse / 4.0) * 100.0)
    # This is deliberately conservative: it estimates the population that must
    # remain visually black, not individual black materials.
    natural_black = float(
        np.mean(
            (interior_values < diffuse / 5.0)
            & (interior_values > noise_floor * 2.0)
        )
        * 100.0
    )

    window_area = 0.0
    window_luminance = None
    ratio = None
    if window is not None:
        window_valid = usable & (window > 0.5)
        window_area = float(np.mean(window > 0.5) * 100.0)
        if np.count_nonzero(window_valid) >= 256:
            values = y[window_valid]
            # Central-to-upper window luminance represents visible exterior,
            # while rejecting a few extreme sun/specular samples.
            window_luminance = float(np.percentile(values, 70))
            ratio = float(np.log2(max(window_luminance, 1e-8) / max(diffuse, 1e-8)))
        else:
            confidence -= 0.10

    highlight = float(np.percentile(positive, 99.7))
    effective_range = float(np.log2(highlight / max(noise_floor, 1e-12)))
    return HDRB1Measurements(
        interior_diffuse_linear=diffuse,
        photographic_black_linear=photographic_black,
        shadow_occupancy_percent=shadow_occupancy,
        natural_black_occupancy_percent=natural_black,
        window_area_percent=window_area,
        window_luminance_linear=window_luminance,
        window_to_room_ratio_stops=ratio,
        highlight_headroom_stops=float(np.log2(max(highlight, 1e-8))),
        effective_dynamic_range_stops=effective_range,
        usable_interior_percent=float(np.mean(interior) * 100.0),
        measurement_confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _hdr_b1_tone_luminance(
    luminance: np.ndarray,
    *,
    exposure_ev: float,
    toe: float,
    midtone_gamma: float,
    shoulder_power: float,
) -> np.ndarray:
    """One smooth, monotonic photographic curve in scene-linear luminance."""
    x = np.maximum(luminance.astype(np.float32) * float(2.0**exposure_ev), 0.0)
    pivot = 0.18
    gamma_lift = pivot * np.power(np.maximum(x / pivot, 0.0), midtone_gamma)
    # Blend back to a linear origin to keep a gentle finite toe and true black.
    blend = x / (x + max(toe, 1e-6))
    mid = x + blend * (gamma_lift - x)
    # Smooth global shoulder with asymptote at display-linear white.
    return mid / np.power(1.0 + np.power(mid, shoulder_power), 1.0 / shoulder_power)


def _render_hdr_b1_candidate(
    master: HDRMaster,
    exposure_ev: float,
    toe: float,
    midtone_gamma: float,
    shoulder_power: float,
    linear_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    linear = master.linear_srgb.astype(np.float32) if linear_rgb is None else linear_rgb.astype(np.float32)
    scene_y = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )
    mapped_y = _hdr_b1_tone_luminance(
        scene_y,
        exposure_ev=exposure_ev,
        toe=toe,
        midtone_gamma=midtone_gamma,
        shoulder_power=shoulder_power,
    )
    exposed_y = np.maximum(scene_y * float(2.0**exposure_ev), 0.0)
    scale = mapped_y / np.maximum(exposed_y, 1e-8)
    mapped = np.maximum(linear * float(2.0**exposure_ev), 0.0) * scale[..., None]
    # HDR-B1 intentionally defers perceptual gamut compression.
    return _linear_to_srgb(np.clip(mapped, 0.0, 1.0)), mapped_y


def render_hdr_b1_global(
    master: HDRMaster,
    measurements: HDRB1Measurements,
) -> HDRRender:
    """Select one deterministic global MLS exposure and photographic curve."""
    room = max(measurements.interior_diffuse_linear, 1e-8)
    # Candidate exposure is centered on a bright photographic placement. The
    # room/window ratio mostly controls shoulder candidates, not room exposure.
    base_ev = float(np.clip(np.log2(0.16 / room), 0.35, 2.5))
    exposure_values = [base_ev + offset for offset in (-0.15, 0.10, 0.35)]
    toe_values = (0.012, 0.022)
    gamma_values = (0.82, 0.88)
    ratio = measurements.window_to_room_ratio_stops or 0.0
    if ratio > 5.0 or measurements.window_area_percent > 8.0:
        shoulder_values = (1.35, 1.60)
    else:
        shoulder_values = (1.60, 1.90)

    # Candidate scoring uses a deterministic regular sample. The selected
    # transform is then rendered once at full resolution.
    step = max(1, int(np.ceil(max(master.linear_srgb.shape[:2]) / 900.0)))
    linear = master.linear_srgb[::step, ::step].astype(np.float32)
    source_y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    positive = source_y > 1e-8
    interior_low = room / 4.0
    interior_high = room * 3.0
    diffuse_mask = positive & (source_y >= room * 0.55) & (source_y <= interior_high)
    black_mask = positive & (source_y < interior_low)
    highlight_threshold = float(np.percentile(source_y[positive], 99.0))
    highlight_mask = positive & (source_y >= highlight_threshold)

    candidates: list[dict[str, float]] = []
    best = None
    for exposure_ev in exposure_values:
        for toe in toe_values:
            for gamma in gamma_values:
                for shoulder in shoulder_values:
                    rgb, mapped_y = _render_hdr_b1_candidate(
                        master, exposure_ev, toe, gamma, shoulder, linear_rgb=linear
                    )
                    display_y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                    room_display = float(np.median(display_y[diffuse_mask])) if np.any(diffuse_mask) else float(np.median(display_y))
                    black_display = float(np.median(display_y[black_mask])) if np.any(black_mask) else 0.0
                    clipped = float(np.mean(display_y > 0.995) * 100.0)
                    highlight_sep = float(np.std(display_y[highlight_mask])) if np.any(highlight_mask) else 0.0

                    room_cost = abs(room_display - 0.57) * 8.0
                    black_limit = 0.115 + min(measurements.shadow_occupancy_percent, 20.0) * 0.0015
                    black_cost = max(0.0, black_display - black_limit) * 9.0
                    clipping_cost = max(0.0, clipped - 1.25) * 0.18
                    window_weight = min(1.0, measurements.window_area_percent / 8.0)
                    window_cost = max(0.0, 0.025 - highlight_sep) * 4.0 * window_weight
                    # Prefer clean but not flattened midtones and avoid extreme
                    # parameter combinations when photographic scores tie.
                    contrast_cost = abs(gamma - 0.86) * 0.35 + abs(shoulder - 1.65) * 0.05
                    score = room_cost + black_cost + clipping_cost + window_cost + contrast_cost
                    item = {
                        "score": float(score),
                        "exposure_ev": float(exposure_ev),
                        "toe": float(toe),
                        "midtone_gamma": float(gamma),
                        "shoulder_power": float(shoulder),
                        "room_display_median": room_display,
                        "black_display_median": black_display,
                        "clipped_percent": clipped,
                        "highlight_separation": highlight_sep,
                        "room_cost": float(room_cost),
                        "black_cost": float(black_cost),
                        "clipping_cost": float(clipping_cost),
                        "window_cost": float(window_cost),
                        "contrast_cost": float(contrast_cost),
                    }
                    candidates.append(item)
                    if best is None or (item["score"], item["exposure_ev"], item["toe"], item["midtone_gamma"], item["shoulder_power"]) < (
                        best["score"], best["exposure_ev"], best["toe"], best["midtone_gamma"], best["shoulder_power"]
                    ):
                        best = item

    assert best is not None
    rgb, _ = _render_hdr_b1_candidate(
        master,
        best["exposure_ev"],
        best["toe"],
        best["midtone_gamma"],
        best["shoulder_power"],
    )
    candidates.sort(key=lambda item: (item["score"], item["exposure_ev"], item["toe"], item["midtone_gamma"], item["shoulder_power"]))
    return HDRRender(
        rgb=rgb,
        log=MappingProxyType(
            {
                "stage": "hdr_b1_global_mls_render",
                "global_exposure_ev": best["exposure_ev"],
                "toe": best["toe"],
                "midtone_gamma": best["midtone_gamma"],
                "midtone_target_display": 0.57,
                "shoulder_power": best["shoulder_power"],
                "window_to_room_ratio_stops": measurements.window_to_room_ratio_stops,
                "window_area_percent": measurements.window_area_percent,
                "chosen_score": best["score"],
                "chosen_score_components": {key: best[key] for key in ("room_cost", "black_cost", "clipping_cost", "window_cost", "contrast_cost")},
                "candidate_scores": candidates,
                "local_corrections": False,
                "semantic_exposure": False,
                "sharpening": False,
                "denoising": False,
                "gamut_compression": False,
                "output_space": "sRGB",
                "working_precision": "float32",
            }
        ),
    )


def _hdr_b2_tone_luminance(
    luminance: np.ndarray,
    *,
    exposure_ev: float,
    black_anchor_linear: float,
    toe_strength: float,
    midtone_slope: float,
    shoulder_onset: float,
) -> np.ndarray:
    """Smooth global photographic S-curve with a true black origin."""
    gain = float(2.0**exposure_ev)
    x = np.maximum(luminance.astype(np.float32) * gain, 0.0)
    black = max(black_anchor_linear * gain, 1e-8)
    # Smooth rational black anchor. It approaches a linear response above the
    # toe while preserving separation and a true zero in the deepest values.
    anchored = x * x / np.maximum(x + black * toe_strength, 1e-8)
    power = max(midtone_slope, 1.001)
    raised = np.power(anchored, power)
    shoulder = max(shoulder_onset, 1e-4) ** power
    # A Hill curve is monotonic, continuously differentiable, and combines a
    # clean midtone slope with a broad asymptotic highlight shoulder.
    return raised / np.maximum(raised + shoulder, 1e-8)


def _compress_extreme_highlight_color(
    mapped_rgb: np.ndarray,
    mapped_y: np.ndarray,
    *,
    onset: float = 0.82,
    strength: float = 0.78,
) -> np.ndarray:
    """Globally compress only extreme display-linear chroma toward white."""
    peak = np.max(mapped_rgb, axis=2)
    t = np.clip((peak - onset) / max(1.0 - onset, 1e-6), 0.0, 1.0)
    weight = (t * t * (3.0 - 2.0 * t) * strength).astype(np.float32)
    neutral = np.clip(mapped_y, 0.0, 1.0)[..., None]
    return mapped_rgb * (1.0 - weight[..., None]) + neutral * weight[..., None]


def _render_hdr_b2_candidate(
    master: HDRMaster,
    *,
    exposure_ev: float,
    black_anchor_linear: float,
    toe_strength: float,
    midtone_slope: float,
    shoulder_onset: float,
    linear_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    linear = master.linear_srgb.astype(np.float32) if linear_rgb is None else linear_rgb.astype(np.float32)
    scene_y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    mapped_y = _hdr_b2_tone_luminance(
        scene_y,
        exposure_ev=exposure_ev,
        black_anchor_linear=black_anchor_linear,
        toe_strength=toe_strength,
        midtone_slope=midtone_slope,
        shoulder_onset=shoulder_onset,
    )
    exposed_y = np.maximum(scene_y * float(2.0**exposure_ev), 0.0)
    scale = mapped_y / np.maximum(exposed_y, 1e-8)
    mapped = np.maximum(linear * float(2.0**exposure_ev), 0.0) * scale[..., None]
    mapped = _compress_extreme_highlight_color(mapped, mapped_y)
    return _linear_to_srgb(np.clip(mapped, 0.0, 1.0)), mapped_y


def render_hdr_b2_global(master: HDRMaster, measurements: HDRB1Measurements) -> HDRRender:
    """Focused HDR-B2 global MLS render; no spatially varying operations."""
    room = max(measurements.interior_diffuse_linear, 1e-8)
    base_ev = float(np.clip(np.log2(0.23 / room), 0.6, 3.0))
    exposure_values = tuple(base_ev + offset for offset in (0.05, 0.30, 0.55))
    black_values = tuple(measurements.photographic_black_linear * scale for scale in (0.65, 0.95))
    toe_values = (0.75, 1.05)
    slope_values = (1.08, 1.16)
    ratio = measurements.window_to_room_ratio_stops or 0.0
    shoulder_values = (0.64, 0.72) if ratio > 4.5 else (0.70, 0.80)

    step = max(1, int(np.ceil(max(master.linear_srgb.shape[:2]) / 900.0)))
    linear = master.linear_srgb[::step, ::step].astype(np.float32)
    source_y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    positive = source_y > 1e-8
    diffuse_mask = positive & (source_y >= room * 0.55) & (source_y <= room * 3.0)
    black_mask = positive & (source_y < room / 4.0)
    upper_diffuse_mask = positive & (source_y >= room) & (source_y <= room * 3.0)
    highlight_threshold = float(np.percentile(source_y[positive], 99.0))
    highlight_mask = positive & (source_y >= highlight_threshold)

    candidates: list[dict[str, float]] = []
    best = None
    for exposure_ev in exposure_values:
        for black_anchor in black_values:
            for toe in toe_values:
                for slope in slope_values:
                    for shoulder in shoulder_values:
                        rgb, _ = _render_hdr_b2_candidate(
                            master,
                            exposure_ev=exposure_ev,
                            black_anchor_linear=black_anchor,
                            toe_strength=toe,
                            midtone_slope=slope,
                            shoulder_onset=shoulder,
                            linear_rgb=linear,
                        )
                        display_y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                        room_median = float(np.median(display_y[diffuse_mask])) if np.any(diffuse_mask) else float(np.median(display_y))
                        room_upper = float(np.percentile(display_y[upper_diffuse_mask], 75)) if np.any(upper_diffuse_mask) else float(np.percentile(display_y, 75))
                        black_median = float(np.median(display_y[black_mask])) if np.any(black_mask) else 0.0
                        black_p10 = float(np.percentile(display_y[black_mask], 10)) if np.any(black_mask) else 0.0
                        clipped = float(np.mean(display_y > 0.995) * 100.0)
                        highlight_sep = float(np.std(display_y[highlight_mask])) if np.any(highlight_mask) else 0.0

                        room_cost = abs(room_median - 0.60) * 9.0 + abs(room_upper - 0.72) * 2.5
                        black_cost = abs(black_median - 0.095) * 3.5 + max(0.0, 0.018 - black_p10) * 3.0
                        clipping_cost = max(0.0, clipped - 2.5) * 0.12
                        window_weight = min(1.0, measurements.window_area_percent / 8.0)
                        window_cost = max(0.0, 0.018 - highlight_sep) * 2.0 * window_weight
                        contrast_cost = abs(slope - 1.13) * 0.22 + abs(toe - 0.90) * 0.04
                        score = room_cost + black_cost + clipping_cost + window_cost + contrast_cost
                        item = {
                            "score": float(score),
                            "exposure_ev": float(exposure_ev),
                            "black_anchor_linear": float(black_anchor),
                            "toe_strength": float(toe),
                            "midtone_slope": float(slope),
                            "shoulder_onset": float(shoulder),
                            "shoulder_compression": float(slope),
                            "diffuse_room_median": room_median,
                            "upper_diffuse_p75": room_upper,
                            "black_display_median": black_median,
                            "black_display_p10": black_p10,
                            "clipped_percent": clipped,
                            "highlight_separation": highlight_sep,
                            "room_cost": float(room_cost),
                            "black_cost": float(black_cost),
                            "clipping_cost": float(clipping_cost),
                            "window_cost": float(window_cost),
                            "contrast_cost": float(contrast_cost),
                        }
                        candidates.append(item)
                        key = (item["score"], item["exposure_ev"], item["black_anchor_linear"], item["toe_strength"], item["midtone_slope"], item["shoulder_onset"])
                        if best is None or key < (
                            best["score"], best["exposure_ev"], best["black_anchor_linear"], best["toe_strength"], best["midtone_slope"], best["shoulder_onset"]
                        ):
                            best = item

    assert best is not None
    rgb, _ = _render_hdr_b2_candidate(
        master,
        exposure_ev=best["exposure_ev"],
        black_anchor_linear=best["black_anchor_linear"],
        toe_strength=best["toe_strength"],
        midtone_slope=best["midtone_slope"],
        shoulder_onset=best["shoulder_onset"],
    )
    candidates.sort(key=lambda item: (item["score"], item["exposure_ev"], item["black_anchor_linear"], item["toe_strength"], item["midtone_slope"], item["shoulder_onset"]))
    return HDRRender(
        rgb=rgb,
        log=MappingProxyType(
            {
                "stage": "hdr_b2_global_mls_render",
                "global_exposure_ev": best["exposure_ev"],
                "black_anchor_linear": best["black_anchor_linear"],
                "toe_strength": best["toe_strength"],
                "midtone_slope": best["midtone_slope"],
                "shoulder_onset": best["shoulder_onset"],
                "shoulder_compression": best["shoulder_compression"],
                "final_diffuse_room_median": best["diffuse_room_median"],
                "final_upper_diffuse_p75": best["upper_diffuse_p75"],
                "display_clipping_percent": best["clipped_percent"],
                "window_to_room_ratio_stops": measurements.window_to_room_ratio_stops,
                "natural_black_occupancy_percent": measurements.natural_black_occupancy_percent,
                "highlight_chroma_compression_onset": 0.82,
                "highlight_chroma_compression_strength": 0.78,
                "global_vibrance": 0.0,
                "chosen_score": best["score"],
                "chosen_score_components": {key: best[key] for key in ("room_cost", "black_cost", "clipping_cost", "window_cost", "contrast_cost")},
                "candidate_scores": candidates,
                "local_corrections": False,
                "semantic_exposure": False,
                "sharpening": False,
                "denoising": False,
                "output_space": "sRGB",
                "working_precision": "float32",
            }
        ),
    )


def _linear_srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = np.maximum(rgb @ np.array(
        [[0.4122214708, 0.5363325363, 0.0514459929],
         [0.2119034982, 0.6806995451, 0.1073969566],
         [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32).T,
        0.0,
    )
    lms = np.cbrt(lms)
    return lms @ np.array(
        [[0.2104542553, 0.7936177850, -0.0040720468],
         [1.9779984951, -2.4285922050, 0.4505937099],
         [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32).T


def _oklab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    lms = lab @ np.array(
        [[1.0, 0.3963377774, 0.2158037573],
         [1.0, -0.1055613458, -0.0638541728],
         [1.0, -0.0894841775, -1.2914855480]], dtype=np.float32).T
    lms = lms * lms * lms
    return lms @ np.array(
        [[4.0767416621, -3.3077115913, 0.2309699292],
         [-1.2684380046, 2.6097574011, -0.3413193965],
         [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=np.float32).T


def _hdr_c_perceptual_color_tile(
    rgb: np.ndarray,
    *,
    richness: float,
) -> tuple[np.ndarray, float]:
    """Global Oklab color rendering with neutral protection and gamut mapping."""
    lab = _linear_srgb_to_oklab(np.maximum(rgb.astype(np.float32), 0.0))
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    neutral_weight = np.clip((chroma - 0.018) / 0.065, 0.0, 1.0)
    neutral_weight = neutral_weight * neutral_weight * (3.0 - 2.0 * neutral_weight)
    high_chroma_guard = 1.0 / (1.0 + 0.35 * np.power(chroma / 0.24, 4.0))
    requested_scale = 1.0 + (richness - 1.0) * neutral_weight * high_chroma_guard

    base_ab = lab[..., 1:3].copy()
    trial = lab.copy()
    trial[..., 1:3] = base_ab * requested_scale[..., None]
    rendered = _oklab_to_linear_srgb(trial)
    out = (np.min(rendered, axis=2) < 0.0) | (np.max(rendered, axis=2) > 1.0)
    compressed_percent = float(np.mean(out) * 100.0)
    if np.any(out):
        low = np.zeros(chroma.shape, np.float32)
        high = requested_scale.astype(np.float32)
        for _ in range(6):
            mid = (low + high) * 0.5
            probe = lab.copy()
            probe[..., 1:3] = base_ab * mid[..., None]
            candidate = _oklab_to_linear_srgb(probe)
            inside = (np.min(candidate, axis=2) >= 0.0) & (np.max(candidate, axis=2) <= 1.0)
            low = np.where(inside, mid, low)
            high = np.where(inside, high, mid)
        trial[..., 1:3] = base_ab * np.where(out, low, requested_scale)[..., None]
        rendered = _oklab_to_linear_srgb(trial)
    return np.clip(rendered, 0.0, 1.0).astype(np.float32), compressed_percent


def _hdr_c_source_locked_color_tile(
    mapped_rgb: np.ndarray,
    source_rgb: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Preserve source OKLCH hue and absolute chroma after tone mapping."""
    mapped_lab = _linear_srgb_to_oklab(np.maximum(mapped_rgb.astype(np.float32), 0.0))
    source_lab = _linear_srgb_to_oklab(np.maximum(source_rgb.astype(np.float32), 0.0))
    source_c = np.sqrt(source_lab[..., 1] ** 2 + source_lab[..., 2] ** 2)
    trial = mapped_lab.copy()
    trial[..., 1:3] = source_lab[..., 1:3]
    rendered = _oklab_to_linear_srgb(trial)
    out = (np.min(rendered, axis=2) < 0.0) | (np.max(rendered, axis=2) > 1.0)
    compressed_percent = float(np.mean(out) * 100.0)
    if np.any(out):
        low = np.zeros(source_c.shape, np.float32)
        high = np.ones(source_c.shape, np.float32)
        for _ in range(6):
            mid = (low + high) * 0.5
            probe = mapped_lab.copy()
            probe[..., 1:3] = trial[..., 1:3] * mid[..., None]
            candidate = _oklab_to_linear_srgb(probe)
            inside = (np.min(candidate, axis=2) >= 0.0) & (np.max(candidate, axis=2) <= 1.0)
            low = np.where(inside, mid, low)
            high = np.where(inside, high, mid)
        trial[..., 1:3] *= np.where(out, low, 1.0)[..., None]
        rendered = _oklab_to_linear_srgb(trial)
    return np.clip(rendered, 0.0, 1.0).astype(np.float32), compressed_percent


def _smooth01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _hdr_c_source_referenced_color_tile(
    mapped_rgb: np.ndarray,
    source_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bounded global chroma rendering with immutable scene-linear hue.

    The gain is a continuous function of source chroma, source/rendered
    lightness, luminance lift, and display-gamut proximity.  It contains no
    spatial or semantic decisions.
    """
    mapped_lab = _linear_srgb_to_oklab(np.maximum(mapped_rgb.astype(np.float32), 0.0))
    source_lab = _linear_srgb_to_oklab(np.maximum(source_rgb.astype(np.float32), 0.0))
    source_c = np.hypot(source_lab[..., 1], source_lab[..., 2])

    # Near-neutrals approach zero enhancement continuously; already vivid
    # colors likewise receive progressively less additional chroma.
    neutral_gate = _smooth01((source_c - 0.004) / 0.040)
    vivid_guard = 1.0 / (1.0 + np.power(source_c / 0.105, 4.0))

    source_l = np.maximum(source_lab[..., 0], 0.0)
    rendered_l = np.clip(mapped_lab[..., 0], 0.0, 1.0)
    source_midtone = _smooth01(source_l / 0.42)
    rendered_midtone = _smooth01(rendered_l / 0.48) * _smooth01((1.0 - rendered_l) / 0.32)
    lift_stops = np.maximum(np.log2((rendered_l + 0.015) / (source_l + 0.015)), 0.0)
    lift_guard = 1.0 / (1.0 + 0.55 * lift_stops * lift_stops)

    # Measure gamut proximity at the rendered lightness while retaining the
    # source hue/chroma. Colors close to a face of the display cube receive
    # little enhancement before the final bounded gamut fit.
    source_locked = mapped_lab.copy()
    source_locked[..., 1:3] = source_lab[..., 1:3]
    source_locked_rgb = _oklab_to_linear_srgb(source_locked)
    gamut_margin = np.minimum(np.min(source_locked_rgb, axis=2), 1.0 - np.max(source_locked_rgb, axis=2))
    gamut_guard = _smooth01(np.maximum(gamut_margin, 0.0) / 0.075)

    opportunity = (
        neutral_gate
        * vivid_guard
        * (0.22 + 0.78 * source_midtone)
        * (0.30 + 0.70 * rendered_midtone)
        * lift_guard
        * gamut_guard
    )
    chroma_gain = 1.0 + 0.11 * opportunity
    desired_ab = source_lab[..., 1:3] * chroma_gain[..., None]

    # Hue-preserving gamut fit. Binary search yields the largest continuous
    # chroma fraction that fits at the already-fixed rendered lightness.
    desired_probe = mapped_lab.copy()
    desired_probe[..., 1:3] = desired_ab
    desired_rgb = _oklab_to_linear_srgb(desired_probe)
    desired_inside = (np.min(desired_rgb, axis=2) >= 0.0) & (np.max(desired_rgb, axis=2) <= 1.0)
    low = np.zeros(source_c.shape, np.float32)
    high = np.ones(source_c.shape, np.float32)
    for _ in range(8):
        mid = (low + high) * 0.5
        probe = mapped_lab.copy()
        probe[..., 1:3] = desired_ab * mid[..., None]
        candidate = _oklab_to_linear_srgb(probe)
        inside = (np.min(candidate, axis=2) >= 0.0) & (np.max(candidate, axis=2) <= 1.0)
        low = np.where(inside, mid, low)
        high = np.where(inside, high, mid)
    final_fraction = np.where(desired_inside, 1.0, low)
    mapped_lab[..., 1:3] = desired_ab * final_fraction[..., None]
    rendered = _oklab_to_linear_srgb(mapped_lab)
    compressed = final_fraction < (1.0 - 1.0 / 256.0)
    enhanced = chroma_gain - 1.0
    stats = {
        "mean_chroma_enhancement_percent": float(np.mean(enhanced) * 100.0),
        "p95_chroma_enhancement_percent": float(np.percentile(enhanced, 95) * 100.0),
        "maximum_chroma_enhancement_percent": float(np.max(enhanced) * 100.0),
        "gamut_compressed_percent": float(np.mean(compressed) * 100.0),
        "mean_gamut_chroma_reduction_percent": float(np.mean(1.0 - final_fraction) * 100.0),
    }
    return np.clip(rendered, 0.0, 1.0).astype(np.float32), stats


def _hdr_c_tone_luminance(
    luminance: np.ndarray,
    *,
    exposure_ev: float,
    black_point_linear: float,
    toe_strength: float,
    midtone_slope: float,
    shoulder: float,
) -> np.ndarray:
    """HDR-C's global photographic curve, independent from HDR-B2."""
    gain = float(2.0**exposure_ev)
    x = np.maximum(luminance.astype(np.float32) * gain, 0.0)
    black = max(black_point_linear * gain, 1e-8)
    anchored = x * x / np.maximum(x + black * toe_strength, 1e-8)
    power = max(midtone_slope, 1.001)
    raised = np.power(anchored, power)
    return raised / np.maximum(raised + max(shoulder, 1e-4) ** power, 1e-8)


def _render_hdr_c_candidate(
    master: HDRMaster,
    *,
    exposure_ev: float,
    black_point_linear: float,
    toe_strength: float,
    midtone_slope: float,
    shoulder: float,
    color_richness: float,
    linear_rgb: np.ndarray | None = None,
    color_mode: str = "revised",
) -> tuple[np.ndarray, np.ndarray, float]:
    linear = master.linear_srgb.astype(np.float32) if linear_rgb is None else linear_rgb.astype(np.float32)
    scene_y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    mapped_y = _hdr_c_tone_luminance(
        scene_y,
        exposure_ev=exposure_ev,
        black_point_linear=black_point_linear,
        toe_strength=toe_strength,
        midtone_slope=midtone_slope,
        shoulder=shoulder,
    )
    exposed_y = np.maximum(scene_y * float(2.0**exposure_ev), 0.0)
    scale = mapped_y / np.maximum(exposed_y, 1e-8)
    mapped = np.maximum(linear * float(2.0**exposure_ev), 0.0) * scale[..., None]

    result = np.empty_like(mapped, dtype=np.float32)
    weighted_compression = 0.0
    weighted_color_stats: dict[str, float] = {}
    for y0 in range(0, mapped.shape[0], 256):
        y1 = min(mapped.shape[0], y0 + 256)
        if color_mode == "current":
            tile, compressed = _hdr_c_perceptual_color_tile(mapped[y0:y1], richness=color_richness)
        elif color_mode == "disabled":
            tile, compressed = np.clip(mapped[y0:y1], 0.0, 1.0), 0.0
        elif color_mode == "revised":
            tile, compressed = _hdr_c_source_locked_color_tile(mapped[y0:y1], linear[y0:y1])
        elif color_mode == "enhanced":
            tile, color_stats = _hdr_c_source_referenced_color_tile(mapped[y0:y1], linear[y0:y1])
            compressed = color_stats["gamut_compressed_percent"]
            for name, value in color_stats.items():
                weighted_color_stats[name] = weighted_color_stats.get(name, 0.0) + value * (y1 - y0)
        else:
            raise ValueError(f"unknown HDR-C color mode: {color_mode}")
        result[y0:y1] = tile
        weighted_compression += compressed * (y1 - y0)
    compressed_percent = weighted_compression / max(mapped.shape[0], 1)
    _render_hdr_c_candidate.last_color_stats = {
        name: value / max(mapped.shape[0], 1) for name, value in weighted_color_stats.items()
    }
    return _linear_to_srgb(result), mapped_y, float(compressed_percent)


def render_hdr_c_global(master: HDRMaster, measurements: HDRB1Measurements) -> HDRRender:
    """Joint global photographic and perceptual-color renderer for HDR-C."""
    room = max(measurements.interior_diffuse_linear, 1e-8)
    base_ev = float(np.clip(np.log2(0.285 / room), 0.8, 3.35))
    exposure_values = (base_ev + 0.65, base_ev + 0.95, base_ev + 1.25)
    black_values = tuple(measurements.photographic_black_linear * scale for scale in (2.25, 3.00))
    toe_values = (1.35, 1.60)
    slope_values = (1.18, 1.26)
    ratio = measurements.window_to_room_ratio_stops or 0.0
    shoulder_values = (0.74, 0.82) if ratio > 4.5 else (0.80, 0.90)
    richness_values = (1.0,)

    step = max(1, int(np.ceil(max(master.linear_srgb.shape[:2]) / 850.0)))
    linear = master.linear_srgb[::step, ::step].astype(np.float32)
    analysis_y = master.xyz_d50[::step, ::step, 1].astype(np.float32)
    positive = analysis_y > 1e-8
    diffuse = positive & (analysis_y >= room * 0.55) & (analysis_y <= room * 3.0)
    upper = positive & (analysis_y >= room) & (analysis_y <= room * 3.0)
    blacks = positive & (analysis_y < room / 4.0)
    highlight_threshold = float(np.percentile(analysis_y[positive], 99.0))
    highlights = positive & (analysis_y >= highlight_threshold)

    candidates: list[dict[str, float]] = []
    best = None
    for exposure in exposure_values:
        for black in black_values:
            for toe in toe_values:
                for slope in slope_values:
                    for shoulder in shoulder_values:
                        for richness in richness_values:
                            rgb, _, gamut_percent = _render_hdr_c_candidate(
                                master,
                                exposure_ev=exposure,
                                black_point_linear=black,
                                toe_strength=toe,
                                midtone_slope=slope,
                                shoulder=shoulder,
                                color_richness=richness,
                                linear_rgb=linear, color_mode="enhanced",
                            )
                            display_y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                            room_median = float(np.median(display_y[diffuse]))
                            upper_p75 = float(np.percentile(display_y[upper], 75))
                            black_median = float(np.median(display_y[blacks])) if np.any(blacks) else 0.0
                            black_p10 = float(np.percentile(display_y[blacks], 10)) if np.any(blacks) else 0.0
                            clipping = float(np.mean(display_y > 0.995) * 100.0)
                            highlight_sep = float(np.std(display_y[highlights]))

                            room_cost = abs(room_median - 0.635) * 10.0 + abs(upper_p75 - 0.755) * 3.0
                            black_cost = abs(black_median - 0.090) * 4.0 + max(0.0, 0.012 - black_p10) * 2.0
                            clipping_cost = max(0.0, clipping - 1.5) * 0.10
                            window_cost = max(0.0, 0.012 - highlight_sep) * min(1.0, measurements.window_area_percent / 8.0)
                            color_cost = max(0.0, gamut_percent - 12.0) * 0.002
                            contrast_cost = abs(slope - 1.23) * 0.15
                            score = room_cost + black_cost + clipping_cost + window_cost + color_cost + contrast_cost
                            item = {
                                "score": float(score), "exposure_ev": float(exposure),
                                "black_point_linear": float(black), "toe_strength": float(toe),
                                "midtone_slope": float(slope), "shoulder": float(shoulder),
                                "color_richness": float(richness), "gamut_mapped_percent": float(gamut_percent),
                                "room_median": room_median, "upper_diffuse_p75": upper_p75,
                                "black_median": black_median, "black_p10": black_p10,
                                "clipping_percent": clipping, "highlight_separation": highlight_sep,
                                "room_cost": float(room_cost), "black_cost": float(black_cost),
                                "clipping_cost": float(clipping_cost), "window_cost": float(window_cost),
                                "color_cost": float(color_cost), "contrast_cost": float(contrast_cost),
                            }
                            candidates.append(item)
                            key = tuple(item[name] for name in ("score", "exposure_ev", "black_point_linear", "toe_strength", "midtone_slope", "shoulder", "color_richness"))
                            if best is None or key < tuple(best[name] for name in ("score", "exposure_ev", "black_point_linear", "toe_strength", "midtone_slope", "shoulder", "color_richness")):
                                best = item

    assert best is not None
    rgb, _, gamut_percent = _render_hdr_c_candidate(
        master,
        exposure_ev=best["exposure_ev"], black_point_linear=best["black_point_linear"],
        toe_strength=best["toe_strength"], midtone_slope=best["midtone_slope"],
        shoulder=best["shoulder"], color_richness=best["color_richness"], color_mode="enhanced",
    )
    candidates.sort(key=lambda item: tuple(item[name] for name in ("score", "exposure_ev", "black_point_linear", "toe_strength", "midtone_slope", "shoulder", "color_richness")))
    return HDRRender(
        rgb=rgb,
        log=MappingProxyType({
            "stage": "hdr_c_global_photographic_render",
            "global_exposure_ev": best["exposure_ev"], "black_point_linear": best["black_point_linear"],
            "toe_strength": best["toe_strength"], "midtone_slope": best["midtone_slope"],
            "shoulder": best["shoulder"], "color_richness": best["color_richness"],
            "color_rendering": "source_referenced_bounded_oklch", "gamut_mapped_percent": gamut_percent,
            "color_rendering_stats": dict(getattr(_render_hdr_c_candidate, "last_color_stats", {})),
            "display_clipping_percent": best["clipping_percent"],
            "final_diffuse_room_median": best["room_median"], "final_upper_diffuse_p75": best["upper_diffuse_p75"],
            "chosen_score": best["score"],
            "chosen_score_components": {key: best[key] for key in ("room_cost", "black_cost", "clipping_cost", "window_cost", "color_cost", "contrast_cost")},
            "candidate_scores": candidates,
            "local_corrections": False, "semantic_exposure": False, "sharpening": False,
            "denoising": False, "output_space": "sRGB", "working_precision": "float32",
        }),
    )


def hdr_b1_measurements_to_dict(measurements: HDRB1Measurements) -> dict[str, float | None]:
    return {
        "interior_diffuse_linear": measurements.interior_diffuse_linear,
        "photographic_black_linear": measurements.photographic_black_linear,
        "shadow_occupancy_percent": measurements.shadow_occupancy_percent,
        "natural_black_occupancy_percent": measurements.natural_black_occupancy_percent,
        "window_area_percent": measurements.window_area_percent,
        "window_luminance_linear": measurements.window_luminance_linear,
        "window_to_room_ratio_stops": measurements.window_to_room_ratio_stops,
        "highlight_headroom_stops": measurements.highlight_headroom_stops,
        "effective_dynamic_range_stops": measurements.effective_dynamic_range_stops,
        "usable_interior_percent": measurements.usable_interior_percent,
        "measurement_confidence": measurements.measurement_confidence,
    }


def master_fingerprint(master: HDRMaster) -> dict[str, Any]:
    """Cheap mutation guard for the arrays that must remain immutable."""
    return {
        "camera": (master.camera_linear.shape, str(master.camera_linear.dtype), float(np.sum(master.camera_linear, dtype=np.float64))),
        "xyz": (master.xyz_d50.shape, str(master.xyz_d50.dtype), float(np.sum(master.xyz_d50, dtype=np.float64))),
        "srgb": (master.linear_srgb.shape, str(master.linear_srgb.dtype), float(np.sum(master.linear_srgb, dtype=np.float64))),
        "writeable": [master.camera_linear.flags.writeable, master.xyz_d50.flags.writeable, master.linear_srgb.flags.writeable],
    }


def analysis_to_dict(analysis: HDRAnalysis) -> dict[str, Any]:
    return {
        "luminance_percentiles": dict(analysis.luminance_percentiles),
        "log2_luminance_percentiles": dict(analysis.log2_luminance_percentiles),
        "room_median_linear": analysis.room_median_linear,
        "room_median_stops": analysis.room_median_stops,
        "noise_floor_linear": analysis.noise_floor_linear,
        "highlight_p99_9_linear": analysis.highlight_p99_9_linear,
        "dynamic_range_stops": analysis.dynamic_range_stops,
        "pixels_above_display_white_percent": analysis.pixels_above_display_white_percent,
        "pixels_below_zero_percent": analysis.pixels_below_zero_percent,
        "window_headroom_stops_p99": analysis.window_headroom_stops_p99,
    }
