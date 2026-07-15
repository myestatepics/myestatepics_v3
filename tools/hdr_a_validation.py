#!/usr/bin/env python3
from __future__ import annotations

"""Run HDR-A validation with Original DNG neutral render as the only baseline."""

import argparse
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.hdr_pipeline import (
    analysis_to_dict,
    analyze_hdr_master,
    hdr_b1_measurements_to_dict,
    load_hdr_master,
    master_fingerprint,
    measure_hdr_b1_scene,
    render_hdr_a_foundation,
    render_hdr_b1_global,
    render_hdr_b2_global,
    render_hdr_c_global,
    render_neutral_display,
)
from engine.refine import refine_masks
from engine.scene import build_scene
from engine.segmentation import SceneSegmenter
from engine.utils import resize_for_analysis


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _save_render(rgb: np.ndarray, jpeg: Path, png16: Path) -> None:
    jpeg.parent.mkdir(parents=True, exist_ok=True)
    png16.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(jpeg), cv2.cvtColor(_u8(rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    u16 = np.clip(rgb * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(png16), cv2.cvtColor(u16, cv2.COLOR_RGB2BGR))


def _segment(proxy: np.ndarray, segmenter: SceneSegmenter):
    proxy_u8 = _u8(proxy)
    working, _ = resize_for_analysis(proxy_u8, 2048)
    result = segmenter.segment(working)
    labels = result.label_map
    if labels.shape != proxy.shape[:2]:
        labels = cv2.resize(labels, (proxy.shape[1], proxy.shape[0]), interpolation=cv2.INTER_NEAREST)
    selected = {
        name: class_id
        for name, class_id in result.name_to_id.items()
        if name in {
            "wall", "ceiling", "floor", "window", "curtain", "door",
            "cabinet", "mirror", "sofa", "table", "chair", "bed",
            "rug", "plant", "painting", "lamp",
        }
    }
    masks, refine_log = refine_masks(proxy_u8, labels, selected)
    return build_scene(masks, proxy.shape[:2]), {
        "working_resolution": [int(working.shape[1]), int(working.shape[0])],
        "inference_seconds": result.inference_seconds,
        "refinement": refine_log,
    }


def _display_metrics(rgb: np.ndarray) -> dict:
    y = _luma(rgb)
    return {
        "mean": float(np.mean(y)),
        **{f"p{str(p).replace('.', '_')}": float(np.percentile(y, p)) for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)},
        "shadow_below_0_02_percent": float(np.mean(y < 0.02) * 100.0),
        "highlight_above_0_98_percent": float(np.mean(y > 0.98) * 100.0),
    }


def _histogram(rgb: np.ndarray) -> list[int]:
    return np.histogram(_luma(rgb), bins=256, range=(0, 1))[0].astype(int).tolist()


def _material_drift(neutral: np.ndarray, output: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask > 0.4
    if np.count_nonzero(valid) < 256:
        return {"hue_p95_degrees": 0.0, "saturation_p95": 0.0, "lab_chroma_p95": 0.0}
    src_hsv = cv2.cvtColor(neutral.astype(np.float32), cv2.COLOR_RGB2HSV)
    dst_hsv = cv2.cvtColor(output.astype(np.float32), cv2.COLOR_RGB2HSV)
    hue = np.abs(src_hsv[..., 0] - dst_hsv[..., 0])
    hue = np.minimum(hue, 360.0 - hue)
    saturation = np.abs(src_hsv[..., 1] - dst_hsv[..., 1])
    src_lab = cv2.cvtColor(neutral.astype(np.float32), cv2.COLOR_RGB2LAB)
    dst_lab = cv2.cvtColor(output.astype(np.float32), cv2.COLOR_RGB2LAB)
    chroma = np.sqrt(np.sum((dst_lab[..., 1:3] - src_lab[..., 1:3]) ** 2, axis=2))
    return {
        "hue_p95_degrees": float(np.percentile(hue[valid], 95)),
        "saturation_p95": float(np.percentile(saturation[valid], 95)),
        "lab_chroma_p95": float(np.percentile(chroma[valid], 95)),
    }


def _patch_texture(y: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if np.count_nonzero(valid) < 256:
        return 0.0
    detail = cv2.Laplacian(y.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.std(detail[valid]))


def _recovery_metrics(master, neutral: np.ndarray, output: np.ndarray, scene) -> dict:
    neutral_y, output_y = _luma(neutral), _luma(output)
    window = scene.masks.get("window", np.zeros(neutral_y.shape, np.float32))
    structure = scene.masks.get("structure", np.ones(neutral_y.shape, np.float32))
    master_y = master.xyz_d50[..., 1]
    window_valid = window > 0.5
    shadow_threshold = float(np.percentile(master_y[structure > 0.45], 10)) if np.any(structure > 0.45) else float(np.percentile(master_y, 10))
    shadow_valid = (structure > 0.45) & (master_y <= shadow_threshold)
    result = {
        "shadows": {
            "master_linear_threshold_p10": shadow_threshold,
            "neutral_display_median": float(np.median(neutral_y[shadow_valid])) if np.any(shadow_valid) else None,
            "pipeline_display_median": float(np.median(output_y[shadow_valid])) if np.any(shadow_valid) else None,
            "neutral_texture": _patch_texture(neutral_y, shadow_valid.astype(np.float32)),
            "pipeline_texture": _patch_texture(output_y, shadow_valid.astype(np.float32)),
        },
        "highlights": {
            "master_pixels_above_linear_white_percent": float(np.mean(master_y > 1.0) * 100.0),
            "master_window_p99_linear": None,
            "neutral_window_p99_display": None,
            "pipeline_window_p99_display": None,
            "neutral_window_texture": 0.0,
            "pipeline_window_texture": 0.0,
        },
    }
    if np.count_nonzero(window_valid) >= 256:
        result["highlights"].update(
            {
                "master_window_p99_linear": float(np.percentile(master_y[window_valid], 99)),
                "neutral_window_p99_display": float(np.percentile(neutral_y[window_valid], 99)),
                "pipeline_window_p99_display": float(np.percentile(output_y[window_valid], 99)),
                "neutral_window_texture": _patch_texture(neutral_y, window),
                "pipeline_window_texture": _patch_texture(output_y, window),
            }
        )
    return result


def _final_photographic_stats(master, output: np.ndarray, scene, measurements) -> dict:
    display_y = _luma(output)
    source_y = master.xyz_d50[..., 1]
    room = measurements.interior_diffuse_linear
    window = scene.masks.get("window", np.zeros(display_y.shape, np.float32))
    diffuse = (window < 0.25) & (source_y >= room * 0.55) & (source_y <= room * 3.0)
    upper = (window < 0.25) & (source_y >= room) & (source_y <= room * 3.0)
    window_valid = window > 0.5
    room_median = float(np.median(display_y[diffuse])) if np.any(diffuse) else float(np.median(display_y))
    upper_p75 = float(np.percentile(display_y[upper], 75)) if np.any(upper) else float(np.percentile(display_y, 75))
    window_median = float(np.median(display_y[window_valid])) if np.any(window_valid) else None
    return {
        "final_diffuse_room_median": room_median,
        "final_upper_diffuse_p75": upper_p75,
        "display_clipping_above_0_995_percent": float(np.mean(display_y > 0.995) * 100.0),
        "window_display_median": window_median,
        "window_to_room_display_ratio": (window_median / max(room_median, 1e-8)) if window_median is not None else None,
    }


def _crop_box(cx: int, cy: int, width: int, height: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0 = int(np.clip(cx - width // 2, 0, max(0, w - width)))
    y0 = int(np.clip(cy - height // 2, 0, max(0, h - height)))
    return x0, y0, min(w, x0 + width), min(h, y0 + height)


def _point_from_mask(mask: np.ndarray, score: np.ndarray | None = None, maximum: bool = False) -> tuple[int, int]:
    valid = mask > 0.5
    if not np.any(valid):
        return mask.shape[1] // 2, mask.shape[0] // 2
    if score is None:
        ys, xs = np.nonzero(valid)
        return int(np.median(xs)), int(np.median(ys))
    values = np.where(valid, score, -np.inf if maximum else np.inf)
    index = int(np.argmax(values) if maximum else np.argmin(values))
    y, x = np.unravel_index(index, mask.shape)
    return int(x), int(y)


def _crop_regions(neutral: np.ndarray, scene) -> dict[str, tuple[int, int, int, int]]:
    h, w = neutral.shape[:2]
    y = cv2.GaussianBlur(_luma(neutral), (0, 0), max(6.0, min(h, w) / 250.0))
    floor = scene.masks.get("floor", np.zeros((h, w), np.float32))
    ceiling = scene.masks.get("ceiling", np.zeros((h, w), np.float32))
    window = scene.masks.get("window", np.zeros((h, w), np.float32))
    wall = scene.masks.get("wall", np.zeros((h, w), np.float32))
    structure = scene.masks.get("structure", np.ones((h, w), np.float32))
    material = scene.masks.get("protected", structure)
    cabinet = scene.masks.get("cabinet", np.zeros((h, w), np.float32))
    upper_nonwindow = np.zeros((h, w), np.float32)
    upper_nonwindow[: int(h * 0.75)] = 1.0
    upper_nonwindow *= (window < 0.35).astype(np.float32)
    hsv = cv2.cvtColor(neutral.astype(np.float32), cv2.COLOR_RGB2HSV)
    fixture_score = y + 0.25 * hsv[..., 1]
    crop_w, crop_h = min(1400, w), min(950, h)

    points = {
        "darkest_floor": _point_from_mask(np.maximum(floor, cabinet), y),
        "deepest_shadows": _point_from_mask(np.maximum(material, structure * (1 - window)), y),
        "representative_midtone_material": _point_from_mask(material, np.abs(y - np.median(y[material > 0.5])) if np.any(material > 0.5) else y),
        "ceiling": _point_from_mask(ceiling),
        "artificial_light_fixture": _point_from_mask(upper_nonwindow, fixture_score, maximum=True),
        "window": _point_from_mask(window),
    }
    if np.any(window > 0.5) and np.any(wall > 0.5):
        distance = cv2.distanceTransform((window <= 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        points["wall_near_window"] = _point_from_mask(wall, distance)
        points["opposite_wall"] = _point_from_mask(wall, distance, maximum=True)
    else:
        points["wall_near_window"] = _point_from_mask(wall)
        points["opposite_wall"] = _point_from_mask(wall)
    return {name: _crop_box(x, y0, crop_w, crop_h, (h, w)) for name, (x, y0) in points.items()}


def _side_by_side(left: np.ndarray, right: np.ndarray, title: str, path: Path, max_tile=(1000, 667), output_label="MYESTATEPICS HDR-A — global scene-linear foundation", left_label="ORIGINAL HDR DNG — neutral display only") -> None:
    font = ImageFont.load_default(size=26)
    tiles = []
    for array in (left, right):
        image = Image.fromarray(_u8(array))
        image.thumbnail(max_tile, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", max_tile, "black")
        tile.paste(image, ((max_tile[0] - image.width) // 2, (max_tile[1] - image.height) // 2))
        tiles.append(tile)
    canvas = Image.new("RGB", (max_tile[0] * 2, max_tile[1] + 92), (15, 15, 15))
    canvas.paste(tiles[0], (0, 92)); canvas.paste(tiles[1], (max_tile[0], 92))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), title, font=font, fill="white")
    draw.text((20, 50), left_label, font=font, fill=(150, 210, 255))
    draw.text((max_tile[0] + 20, 50), output_label, font=font, fill=(255, 190, 110))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def _three_column(neutral: np.ndarray, b2: np.ndarray, hdr_c: np.ndarray, title: str, path: Path, max_tile=(800, 533)) -> None:
    font = ImageFont.load_default(size=24)
    arrays = (neutral, b2, hdr_c)
    labels = (
        "ORIGINAL HDR DNG — neutral display",
        "MYESTATEPICS HDR-B2",
        "MYESTATEPICS HDR-C",
    )
    colors = ((150, 210, 255), (190, 190, 190), (255, 190, 110))
    canvas = Image.new("RGB", (max_tile[0] * 3, max_tile[1] + 92), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), title, font=font, fill="white")
    for index, (array, label, color) in enumerate(zip(arrays, labels, colors)):
        image = Image.fromarray(_u8(array)); image.thumbnail(max_tile, Image.Resampling.LANCZOS)
        x = index * max_tile[0]
        canvas.paste(image, (x + (max_tile[0] - image.width) // 2, 92 + (max_tile[1] - image.height) // 2))
        draw.text((x + 15, 52), label, font=font, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def _histogram_chart(neutral: list[int], output: list[int], path: Path, output_name="HDR-A output") -> None:
    canvas = Image.new("RGB", (1400, 700), (20, 20, 24)); draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=22)
    for y in (100, 250, 400, 550): draw.line((70, y, 1340, y), fill=(55, 55, 60))
    for name, values, color in (("neutral DNG", neutral, (80, 170, 255)), (output_name, output, (255, 170, 70))):
        a = np.log1p(np.asarray(values, np.float64)); a /= max(float(a.max()), 1)
        points = [(70 + i * 1270 / 255, 620 - float(v) * 520) for i, v in enumerate(a)]
        draw.line(points, fill=color, width=3)
    draw.text((80, 25), "neutral DNG", font=font, fill=(80, 170, 255)); draw.text((350, 25), output_name, font=font, fill=(255, 170, 70))
    draw.text((70, 655), "black", font=font, fill="white"); draw.text((1280, 655), "white", font=font, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="output/hdr_a")
    parser.add_argument("--renderer", choices=("hdr-a", "hdr-b1", "hdr-b2", "hdr-c"), default="hdr-a")
    args = parser.parse_args()
    root, out = Path(args.input).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    segmenter = SceneSegmenter()
    phase = {"hdr-a": "HDR-A", "hdr-b1": "HDR-B1", "hdr-b2": "HDR-B2", "hdr-c": "HDR-C"}[args.renderer]
    suffix = args.renderer.replace("-", "_")
    output_label = {
        "hdr-a": "MYESTATEPICS HDR-A — global scene-linear foundation",
        "hdr-b1": "MYESTATEPICS HDR-B1 — global MLS renderer",
        "hdr-b2": "MYESTATEPICS HDR-B2 — global photographic renderer",
        "hdr-c": "MYESTATEPICS HDR-C — Lightroom-target global renderer",
    }[args.renderer]
    report = {"phase": phase, "source_of_truth": "original HDR DNG", "files": {}}
    comparisons = []

    for path in sorted(root.glob("*.dng")):
        print(f"{phase}: {path.name}", flush=True)
        master = load_hdr_master(path)
        fingerprint = master_fingerprint(master)
        neutral = render_neutral_display(master)
        scene, segmentation_log = _segment(neutral.rgb, segmenter)
        analysis = analyze_hdr_master(master, scene)
        b1_measurements = measure_hdr_b1_scene(master, scene) if args.renderer in {"hdr-b1", "hdr-b2", "hdr-c"} else None
        b1_output = render_hdr_b1_global(master, b1_measurements) if args.renderer == "hdr-b2" else None
        b2_output = render_hdr_b2_global(master, b1_measurements) if args.renderer == "hdr-c" else None
        if args.renderer == "hdr-c":
            output = render_hdr_c_global(master, b1_measurements)
        elif args.renderer == "hdr-b2":
            output = render_hdr_b2_global(master, b1_measurements)
        elif args.renderer == "hdr-b1":
            output = render_hdr_b1_global(master, b1_measurements)
        else:
            output = render_hdr_a_foundation(master, analysis)
        if master_fingerprint(master) != fingerprint:
            raise RuntimeError(f"immutable HDR master changed while processing {path.name}")

        stem = path.stem
        _save_render(neutral.rgb, out / "neutral" / f"{stem}.jpg", out / "neutral_16bit" / f"{stem}.png")
        _save_render(output.rgb, out / "pipeline" / f"{stem}.jpg", out / "pipeline_16bit" / f"{stem}.png")
        comparison_path = out / "comparisons" / f"{stem}_neutral_vs_{suffix}.jpg"
        if b2_output is not None:
            comparison_path = out / "comparisons_three_column" / f"{stem}_neutral_b2_hdr_c.jpg"
            _three_column(neutral.rgb, b2_output.rgb, output.rgb, path.name, comparison_path)
        else:
            _side_by_side(neutral.rgb, output.rgb, path.name, comparison_path, output_label=output_label)
        comparisons.append(comparison_path)
        if b1_output is not None:
            _side_by_side(
                b1_output.rgb,
                output.rgb,
                path.name,
                out / "comparisons_b1_vs_b2" / f"{stem}_hdr_b1_vs_hdr_b2.jpg",
                output_label=output_label,
                left_label="MYESTATEPICS HDR-B1 — previous global render",
            )

        regions = _crop_regions(neutral.rgb, scene)
        if args.renderer == "hdr-b1":
            regions = {name: box for name, box in regions.items() if name in {"darkest_floor", "deepest_shadows", "ceiling", "window"}}
        elif args.renderer == "hdr-b2":
            regions = {name: box for name, box in regions.items() if name in {"darkest_floor", "deepest_shadows", "representative_midtone_material", "ceiling", "artificial_light_fixture", "window"}}
        crop_rows = []
        for name, (x0, y0, x1, y1) in regions.items():
            left, right = neutral.rgb[y0:y1, x0:x1], output.rgb[y0:y1, x0:x1]
            _save_render(left, out / "crops_100pct" / f"{stem}_{name}_neutral.jpg", out / "crops_100pct_16bit" / f"{stem}_{name}_neutral.png")
            _save_render(right, out / "crops_100pct" / f"{stem}_{name}_{suffix}.jpg", out / "crops_100pct_16bit" / f"{stem}_{name}_{suffix}.png")
            crop_pair = out / "crop_pairs" / f"{stem}_{name}.jpg"
            _side_by_side(left, right, name, crop_pair, max_tile=(900, 610), output_label=output_label); crop_rows.append(crop_pair)
        crop_images = [Image.open(item).convert("RGB") for item in crop_rows]
        crop_sheet = Image.new("RGB", (max(i.width for i in crop_images), sum(i.height for i in crop_images)), (10, 10, 10)); offset = 0
        for image in crop_images: crop_sheet.paste(image, (0, offset)); offset += image.height
        (out / "crop_sheets").mkdir(parents=True, exist_ok=True)
        crop_sheet.save(out / "crop_sheets" / f"{stem}_100pct_crops.jpg", quality=95)

        hist_neutral, hist_output = _histogram(neutral.rgb), _histogram(output.rgb)
        _histogram_chart(hist_neutral, hist_output, out / "histograms" / f"{stem}.png", f"{phase} output")
        material = np.maximum(
            scene.masks.get("protected", np.zeros(neutral.rgb.shape[:2], np.float32)),
            scene.masks.get("floor", np.zeros(neutral.rgb.shape[:2], np.float32)),
        )
        report["files"][path.name] = {
            "decoder": dict(master.metadata),
            "analysis": analysis_to_dict(analysis),
            "hdr_b1_measurements": hdr_b1_measurements_to_dict(b1_measurements) if b1_measurements is not None else None,
            "neutral_render": dict(neutral.log),
            "pipeline_render": dict(output.log),
            "final_photographic_stats": _final_photographic_stats(master, output.rgb, scene, b1_measurements) if b1_measurements is not None else None,
            "segmentation": segmentation_log,
            "display_metrics": {"neutral": _display_metrics(neutral.rgb), "pipeline": _display_metrics(output.rgb)},
            "histograms": {"neutral": hist_neutral, "pipeline": hist_output},
            "dynamic_range": {
                "master_scene_linear_stops": analysis.dynamic_range_stops,
                "neutral_display_p99_9_minus_p0_1": _display_metrics(neutral.rgb)["p99_9"] - _display_metrics(neutral.rgb)["p0_1"],
                "pipeline_display_p99_9_minus_p0_1": _display_metrics(output.rgb)["p99_9"] - _display_metrics(output.rgb)["p0_1"],
            },
            "recovery": _recovery_metrics(master, neutral.rgb, output.rgb, scene),
            "material_color": _material_drift(neutral.rgb, output.rgb, material),
            "master_immutable": master_fingerprint(master)["writeable"] == [False, False, False],
            "crop_boxes": {name: list(box) for name, box in regions.items()},
        }
        del master, neutral, output, scene, b1_output, b2_output
        gc.collect()

    (out / f"{suffix}_metrics.json").write_text(json.dumps(report, indent=2))
    if args.renderer == "hdr-b2":
        columns = [
            "file", "diffuse median", "upper diffuse p75", "exposure EV", "black anchor",
            "toe", "midtone slope", "shoulder onset", "shoulder compression", "clip %",
            "window/room display", "natural black %", "vibrance",
        ]
        rows = []
        for name, values in report["files"].items():
            log = values["pipeline_render"]
            stats = values["final_photographic_stats"]
            rows.append([
                name,
                f"{stats['final_diffuse_room_median']:.4f}",
                f"{stats['final_upper_diffuse_p75']:.4f}",
                f"{log['global_exposure_ev']:.3f}",
                f"{log['black_anchor_linear']:.6f}",
                f"{log['toe_strength']:.3f}",
                f"{log['midtone_slope']:.3f}",
                f"{log['shoulder_onset']:.3f}",
                f"{log['shoulder_compression']:.3f}",
                f"{stats['display_clipping_above_0_995_percent']:.3f}",
                f"{stats['window_to_room_display_ratio']:.3f}" if stats["window_to_room_display_ratio"] is not None else "n/a",
                f"{log['natural_black_occupancy_percent']:.3f}",
                f"{log['global_vibrance']:.3f}",
            ])
        lines = ["# HDR-B2 Rendering Parameters", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        (out / "HDR_B2_RENDERING_PARAMETERS.md").write_text("\n".join(lines) + "\n")
    elif args.renderer == "hdr-c":
        columns = ["file", "exposure EV", "black point", "toe", "midtone slope", "shoulder", "color richness", "gamut mapped %", "clip %", "room median", "upper diffuse p75"]
        rows = []
        for name, values in report["files"].items():
            log = values["pipeline_render"]; stats = values["final_photographic_stats"]
            rows.append([
                name, f"{log['global_exposure_ev']:.3f}", f"{log['black_point_linear']:.6f}",
                f"{log['toe_strength']:.3f}", f"{log['midtone_slope']:.3f}", f"{log['shoulder']:.3f}",
                f"{log['color_richness']:.3f}", f"{log['gamut_mapped_percent']:.3f}",
                f"{stats['display_clipping_above_0_995_percent']:.3f}",
                f"{stats['final_diffuse_room_median']:.4f}", f"{stats['final_upper_diffuse_p75']:.4f}",
            ])
        lines = ["# HDR-C Technical Report", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        (out / "HDR_C_TECHNICAL_REPORT.md").write_text("\n".join(lines) + "\n")
    images = [Image.open(item).convert("RGB") for item in comparisons]
    sheet = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), (10, 10, 10)); offset = 0
    for image in images: sheet.paste(image, (0, offset)); offset += image.height
    sheet.save(out / f"{suffix}_five_image_contact_sheet.jpg", quality=95)
    print(json.dumps({"processed": len(report["files"]), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
