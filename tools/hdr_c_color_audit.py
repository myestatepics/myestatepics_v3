#!/usr/bin/env python3
from __future__ import annotations

"""Audit HDR-C color rendering with frozen photographic parameters."""

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
    _linear_srgb_to_oklab,
    _render_hdr_c_candidate,
    load_hdr_master,
    render_neutral_display,
)
from engine.segmentation import SceneSegmenter
from tools.hdr_a_validation import _crop_box, _point_from_mask, _save_render, _segment, _u8


def _three_way(a: np.ndarray, b: np.ndarray, c: np.ndarray, title: str, path: Path, tile=(800, 533)) -> None:
    labels = ("A — CURRENT HDR-C", "B — STRICT SOURCE-CHROMA LOCK", "C — BOUNDED SOURCE-REFERENCED COLOR")
    colors = ((255, 180, 100), (170, 210, 255), (145, 240, 170))
    font = ImageFont.load_default(size=23)
    canvas = Image.new("RGB", (tile[0] * 3, tile[1] + 90), (14, 14, 14)); draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), title, font=font, fill="white")
    for i, (array, label, color) in enumerate(zip((a, b, c), labels, colors)):
        image = Image.fromarray(_u8(array)); image.thumbnail(tile, Image.Resampling.LANCZOS)
        x = i * tile[0]
        canvas.paste(image, (x + (tile[0] - image.width) // 2, 90 + (tile[1] - image.height) // 2))
        draw.text((x + 15, 50), label, font=font, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path, quality=95)


def _union(scene, names: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, np.float32)
    for name in names:
        result = np.maximum(result, scene.masks.get(name, np.zeros(shape, np.float32)))
    return result


def _regions(scene, neutral: np.ndarray) -> dict[str, tuple[np.ndarray, tuple[int, int, int, int]]]:
    h, w = neutral.shape[:2]; crop_w, crop_h = min(1400, w), min(950, h)
    y = cv2.GaussianBlur(0.2126 * neutral[..., 0] + 0.7152 * neutral[..., 1] + 0.0722 * neutral[..., 2], (0, 0), 12)
    masks = {
        "wood_floor": scene.masks.get("floor", np.zeros((h, w), np.float32)),
        "granite": scene.masks.get("table", np.zeros((h, w), np.float32)),
        "cabinets": scene.masks.get("cabinet", np.zeros((h, w), np.float32)),
        "white_bedding": scene.masks.get("bed", np.zeros((h, w), np.float32)),
        "gray_walls": scene.masks.get("wall", np.zeros((h, w), np.float32)),
        "ceiling": scene.masks.get("ceiling", np.zeros((h, w), np.float32)),
        "fabric": _union(scene, ("sofa", "chair", "rug", "curtain", "bed"), (h, w)),
    }
    protected = scene.masks.get("protected", np.ones((h, w), np.float32))
    result = {}
    for name, mask in masks.items():
        if np.count_nonzero(mask > 0.5) < 256:
            mask = protected
        if name in {"wood_floor", "cabinets"}:
            x, cy = _point_from_mask(mask, y)
        else:
            x, cy = _point_from_mask(mask)
        result[name] = (mask, _crop_box(x, cy, crop_w, crop_h, (h, w)))
    return result


def _color_metrics(reference: np.ndarray, variant: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    valid = mask > 0.55
    if np.count_nonzero(valid) < 256:
        return {"pixels": int(np.count_nonzero(valid)), "hue_delta_degrees": None, "chroma_reference": None, "chroma_variant": None, "delta_e76_median": None}
    ref_lab = cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2LAB)
    var_lab = cv2.cvtColor(variant.astype(np.float32), cv2.COLOR_RGB2LAB)
    ref_c = np.sqrt(ref_lab[..., 1] ** 2 + ref_lab[..., 2] ** 2)
    var_c = np.sqrt(var_lab[..., 1] ** 2 + var_lab[..., 2] ** 2)
    chromatic = valid & (ref_c > 1.5)
    if np.count_nonzero(chromatic) >= 256:
        ref_h = np.degrees(np.arctan2(ref_lab[..., 2], ref_lab[..., 1]))
        var_h = np.degrees(np.arctan2(var_lab[..., 2], var_lab[..., 1]))
        delta_h = (var_h - ref_h + 180.0) % 360.0 - 180.0
        hue_delta = float(np.median(delta_h[chromatic]))
    else:
        hue_delta = None
    de = np.sqrt(np.sum((var_lab - ref_lab) ** 2, axis=2))
    return {
        "pixels": int(np.count_nonzero(valid)),
        "hue_delta_degrees": hue_delta,
        "chroma_reference": float(np.median(ref_c[valid])),
        "chroma_variant": float(np.median(var_c[valid])),
        "delta_e76_median": float(np.median(de[valid])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--parameters", default="output/hdr_c/hdr_c_metrics.json")
    parser.add_argument("--out", default="output/hdr_c_color_refinement")
    args = parser.parse_args()
    root, out = Path(args.input).resolve(), Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path(args.parameters).read_text())["files"]
    segmenter = SceneSegmenter(); report = {"phase": "HDR-C bounded source-referenced color refinement", "exposure_and_tone_frozen": True, "files": {}}
    comparisons = []

    for path in sorted(root.glob("*.dng")):
        print(f"HDR-C color audit: {path.name}", flush=True)
        master = load_hdr_master(path); neutral = render_neutral_display(master); scene, _ = _segment(neutral.rgb, segmenter)
        params = frozen[path.name]["pipeline_render"]
        common = dict(
            exposure_ev=params["global_exposure_ev"], black_point_linear=params["black_point_linear"],
            toe_strength=params["toe_strength"], midtone_slope=params["midtone_slope"],
            shoulder=params["shoulder"], color_richness=1.07,
        )
        current, _, current_gamut = _render_hdr_c_candidate(master, **common, color_mode="current")
        strict, _, strict_gamut = _render_hdr_c_candidate(master, **common, color_mode="revised")
        enhanced, _, enhanced_gamut = _render_hdr_c_candidate(master, **common, color_mode="enhanced")
        enhancement_stats = dict(getattr(_render_hdr_c_candidate, "last_color_stats", {}))

        stem = path.stem
        for name, rgb in (("a_current", current), ("b_strict", strict), ("c_enhanced", enhanced)):
            _save_render(rgb, out / "full_resolution" / f"{stem}_{name}.jpg", out / "full_resolution_16bit" / f"{stem}_{name}.png")
        comparison = out / "comparisons" / f"{stem}_abc.jpg"; _three_way(current, strict, enhanced, path.name, comparison); comparisons.append(comparison)

        region_report = {}
        crop_rows = []
        for region_name, (mask, box) in _regions(scene, neutral.rgb).items():
            x0, y0, x1, y1 = box
            crop = out / "crop_comparisons" / f"{stem}_{region_name}.jpg"
            _three_way(current[y0:y1, x0:x1], strict[y0:y1, x0:x1], enhanced[y0:y1, x0:x1], region_name, crop, tile=(700, 475)); crop_rows.append(crop)
            region_report[region_name] = {
                "current": _color_metrics(neutral.rgb, current, mask),
                "strict": _color_metrics(neutral.rgb, strict, mask),
                "enhanced": _color_metrics(neutral.rgb, enhanced, mask),
                "crop_box": list(box),
            }
        crop_images = [Image.open(item).convert("RGB") for item in crop_rows]
        sheet = Image.new("RGB", (max(i.width for i in crop_images), sum(i.height for i in crop_images)), (10, 10, 10)); offset = 0
        for image in crop_images: sheet.paste(image, (0, offset)); offset += image.height
        (out / "crop_sheets").mkdir(parents=True, exist_ok=True); sheet.save(out / "crop_sheets" / f"{stem}.jpg", quality=95)

        # Stage-specific floor audit in scene-linear Oklab.
        floor = scene.masks.get("floor", np.zeros(master.linear_srgb.shape[:2], np.float32)) > 0.55
        def ok_stats(rgb):
            lab = _linear_srgb_to_oklab(np.maximum(rgb, 0.0)); c = np.hypot(lab[..., 1], lab[..., 2]); valid = floor & (c > 0.01)
            h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])); return {"hue_median_degrees": float(np.median(h[valid])), "chroma_median": float(np.median(c[valid]))}
        report["files"][path.name] = {
            "frozen_parameters": common,
            "current_gamut_mapped_percent": current_gamut,
            "strict_gamut_mapped_percent": strict_gamut,
            "enhanced_gamut_mapped_percent": enhanced_gamut,
            "enhancement_stats": enhancement_stats,
            "regions": region_report,
            "floor_stage_audit": {
                "scene_linear": ok_stats(np.maximum(master.linear_srgb, 0.0)),
                "current_display_linear": ok_stats(np.where(current <= 0.04045, current / 12.92, ((current + 0.055) / 1.055) ** 2.4)),
                "strict_display_linear": ok_stats(np.where(strict <= 0.04045, strict / 12.92, ((strict + 0.055) / 1.055) ** 2.4)),
                "enhanced_display_linear": ok_stats(np.where(enhanced <= 0.04045, enhanced / 12.92, ((enhanced + 0.055) / 1.055) ** 2.4)),
            },
        }
        del master, neutral, scene, current, strict, enhanced; gc.collect()

    (out / "hdr_c_color_audit.json").write_text(json.dumps(report, indent=2))
    images = [Image.open(item).convert("RGB") for item in comparisons]
    sheet = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), (10, 10, 10)); offset = 0
    for image in images: sheet.paste(image, (0, offset)); offset += image.height
    sheet.save(out / "hdr_c_color_abc_contact_sheet.jpg", quality=95)
    print(json.dumps({"processed": len(report["files"]), "output": str(out)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
