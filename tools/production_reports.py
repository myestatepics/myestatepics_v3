#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _crop_box(image: np.ndarray, mode: str, size: int = 700) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    score = cv2.GaussianBlur(gray, (0, 0), 35)
    if mode == "shadow":
        score = 1.0 - score
    elif mode == "material":
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        score = cv2.GaussianBlur(lap, (0, 0), 15)
        score[: h // 3] *= 0.2
    _, _, min_loc, max_loc = cv2.minMaxLoc(score)
    cx, cy = max_loc
    half = min(size // 2, w // 2, h // 2)
    x0 = int(np.clip(cx - half, 0, w - 2 * half)); y0 = int(np.clip(cy - half, 0, h - 2 * half))
    return x0, y0, x0 + 2 * half, y0 + 2 * half


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.batch / "summary.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output / "crops"; crops_dir.mkdir(exist_ok=True)
    chosen = summary["successes"][: min(12, len(summary["successes"]))]
    crop_records, tiles = [], []
    for item in chosen:
        image = np.asarray(Image.open(item["output_path"]).convert("RGB"))
        for mode in ("shadow", "material", "window"):
            box = _crop_box(image, mode)
            crop = Image.fromarray(image).crop(box)
            path = crops_dir / f"{Path(item['source_filename']).stem}_{mode}_100pct.jpg"
            crop.save(path, quality=92)
            crop_records.append({"source": item["source_filename"], "type": mode, "box": box, "path": str(path)})
            tiles.append((f"{item['source_filename']} {mode}", crop))
    tw, th, label, cols = 420, 420, 28, 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, (th + label) * rows), (15, 15, 15))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=15)
    for index, (title, image) in enumerate(tiles):
        image.thumbnail((tw, th), Image.Resampling.LANCZOS)
        x, y = (index % cols) * tw, (index // cols) * (th + label)
        sheet.paste(image, (x, y + label)); draw.text((x + 5, y + 6), title, fill="white", font=font)
    sheet.save(args.output / "representative_100pct_crops.jpg", quality=90)
    performance = {
        "files": summary["success_count"], "failures": summary["failure_count"],
        "total_processing_seconds": summary.get("total_processing_seconds"),
        "mean_processing_seconds": summary.get("mean_processing_seconds"),
        "total_output_mb": summary.get("total_output_mb"),
        "quality_distribution": summary.get("quality_distribution"),
    }
    window_items = [item.get("window_recovery", {}) for item in summary["successes"]]
    window_benchmark = {
        "status_distribution": {
            status: sum(1 for item in window_items if item.get("status") == status)
            for status in sorted({item.get("status", "unknown") for item in window_items})
        },
        "mean_recoverable_percent": float(np.mean([item.get("recoverable_percent", 0.0) for item in window_items])) if window_items else 0.0,
        "mean_sky_completion_percent": float(np.mean([item.get("unrecoverable_sky_percent", 0.0) for item in window_items])) if window_items else 0.0,
        "maximum_blue_spill_percent": float(max((item.get("blue_spill_percent", 0.0) for item in window_items), default=0.0)),
        "files_with_sky_completion": sum(1 for item in window_items if item.get("unrecoverable_sky_percent", 0.0) > 0.0),
        "files_with_no_window_noop": sum(1 for item in window_items if item.get("status") == "no_windows"),
    }
    (args.output / "performance_report.json").write_text(json.dumps(performance, indent=2))
    (args.output / "window_benchmark_report.json").write_text(json.dumps(window_benchmark, indent=2))
    (args.output / "crop_report.json").write_text(json.dumps(crop_records, indent=2))
    print(f"Reports written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
