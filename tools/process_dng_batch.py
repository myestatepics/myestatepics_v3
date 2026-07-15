#!/usr/bin/env python3
from __future__ import annotations

"""Production batch entry point for Lightroom HDR DNG files."""

import argparse
import gc
import json
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.dng import read_lightroom_hdr_dng
from engine.exporter import save_jpeg
from engine.hdr_pipeline import (
    HDRMaster,
    hdr_b1_measurements_to_dict,
    measure_hdr_b1_scene,
    render_hdr_c_global,
    render_neutral_display,
)
from engine.photographic_finish import apply_lightroom_look
from engine.segmentation import SceneSegmenter
from tools.hdr_a_validation import _segment


Processor = Callable[[Path, Path, bool, SceneSegmenter | None], dict]


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def discover_dng_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    return sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".dng"),
        key=_natural_key,
    )


def _immutable(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=np.float32)
    result.setflags(write=False)
    return result


def _master_from_dng(path: Path) -> HDRMaster:
    decoded = read_lightroom_hdr_dng(path)
    return HDRMaster(
        source=path.resolve(),
        camera_linear=_immutable(decoded.camera_linear),
        xyz_d50=_immutable(decoded.xyz_d50),
        linear_srgb=_immutable(decoded.linear_srgb),
        metadata=MappingProxyType(dict(decoded.metadata)),
    )


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _save_debug_stage(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(rgb * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), cv2.cvtColor(data, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write debug stage: {path}")


def _render_one(
    source: Path,
    output_dir: Path,
    debug_stages: bool,
    segmenter: SceneSegmenter | None,
) -> dict:
    master = _master_from_dng(source)
    neutral = render_neutral_display(master)
    if segmenter is None:
        segmenter = SceneSegmenter()
    scene, segmentation_log = _segment(neutral.rgb, segmenter)
    measurements = measure_hdr_b1_scene(master, scene)
    rendered = render_hdr_c_global(master, measurements)
    final_rgb, finish_log = apply_lightroom_look(rendered.rgb, scene)

    output_path = output_dir / f"{source.stem}.jpg"
    export = save_jpeg(
        _u8(final_rgb),
        output_path,
        max_mb=2.5,
        start_quality=92,
        min_quality=82,
        expected_shape=master.linear_srgb.shape[:2],
    )
    if debug_stages:
        debug_dir = output_dir / "debug" / source.stem
        _save_debug_stage(neutral.rgb, debug_dir / "01_neutral_display_16bit.png")
        _save_debug_stage(rendered.rgb, debug_dir / "02_hdr_c_final_16bit.png")
        _save_debug_stage(final_rgb, debug_dir / "03_lightroom_look_16bit.png")
        (debug_dir / "render_log.json").write_text(
            json.dumps(
                {
                    "source": source.name,
                    "decoder": dict(master.metadata),
                    "measurements": hdr_b1_measurements_to_dict(measurements),
                    "segmentation": segmentation_log,
                    "renderer": dict(rendered.log),
                    "photographic_finish": finish_log,
                    "export": export,
                },
                indent=2,
            )
        )

    record = {
        "source_filename": source.name,
        "output_filename": output_path.name,
        "output_path": str(output_path),
        "dimensions": [int(final_rgb.shape[1]), int(final_rgb.shape[0])],
        "decoder": dict(master.metadata),
        "measurements": hdr_b1_measurements_to_dict(measurements),
        "renderer": dict(rendered.log),
        "photographic_finish": finish_log,
        "export": export,
    }
    del master, neutral, scene, measurements, rendered, final_rgb
    gc.collect()
    return record


def _make_contact_sheet(records: list[dict], path: Path) -> None:
    columns, tile_w, tile_h, label_h = 4, 480, 320, 34
    rows = max(1, (len(records) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    if not records:
        draw.text((24, 24), "No DNG files found", fill="white", font=font)
    for index, record in enumerate(records):
        image = Image.open(record["output_path"]).convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        x, y = col * tile_w, row * (tile_h + label_h)
        sheet.paste(image, (x + (tile_w - image.width) // 2, y + label_h + (tile_h - image.height) // 2))
        draw.text((x + 10, y + 8), record["source_filename"], fill="white", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def process_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    debug_stages: bool = False,
    processor: Processor | None = None,
) -> tuple[dict, int]:
    sources = discover_dng_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render = processor or _render_one
    segmenter = SceneSegmenter() if processor is None and sources else None
    successes: list[dict] = []
    failures: list[dict] = []

    for index, source in enumerate(sources, 1):
        try:
            record = render(source, output_dir, debug_stages, segmenter)
            successes.append(record)
            print(f"[{index}/{len(sources)}] OK {source.name} -> {record['output_filename']}", flush=True)
        except Exception as exc:  # batch isolation is intentional
            failures.append({
                "source_filename": source.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"[{index}/{len(sources)}] FAILED {source.name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            gc.collect()

    contact_path = output_dir / "batch_contact_sheet.jpg"
    _make_contact_sheet(successes, contact_path)
    summary = {
        "pipeline": "approved_hdr_c_scene_linear_global_renderer",
        "input_folder": str(input_dir.resolve()),
        "output_folder": str(output_dir.resolve()),
        "total_count": len(sources),
        "success_count": len(successes),
        "failure_count": len(failures),
        "successes": successes,
        "failures": failures,
        "contact_sheet": str(contact_path),
        "debug_stages": bool(debug_stages),
    }
    summary_json = json.dumps(summary, indent=2)
    (output_dir / "batch_summary.json").write_text(summary_json)
    (output_dir / "summary.json").write_text(summary_json)
    print(
        f"Complete: {len(successes)}/{len(sources)} succeeded, {len(failures)} failed. "
        f"Summary: {output_dir / 'batch_summary.json'}",
        flush=True,
    )
    return summary, 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a folder of Lightroom HDR DNG files")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--debug-stages", action="store_true")
    args = parser.parse_args()
    try:
        _, status = process_batch(
            args.input.expanduser(),
            args.output.expanduser(),
            debug_stages=args.debug_stages,
        )
        return status
    except Exception as exc:
        print(f"Batch setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
