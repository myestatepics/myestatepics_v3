#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.dng import read_lightroom_hdr_dng


def _metrics(path: Path) -> dict[str, float | str]:
    dng = read_lightroom_hdr_dng(path)
    linear = np.maximum(dng.linear_srgb, 0.0)
    step = max(1, int(np.ceil(max(linear.shape[:2]) / 700)))
    rgb = linear[::step, ::step]
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    positive = y > 1e-7
    upper = y < np.percentile(y[positive], 92) if np.any(positive) else np.ones(y.shape, bool)
    diffuse = positive & upper
    norm = rgb / np.maximum(np.sum(rgb, axis=2, keepdims=True), 1e-7)
    mid = diffuse & (y > np.percentile(y[diffuse], 20)) if np.any(diffuse) else diffuse
    mean = np.mean(norm[mid], axis=0) if np.any(mid) else np.array([1 / 3] * 3)
    bottom = norm[norm.shape[0] * 2 // 3 :]
    top = norm[: norm.shape[0] // 3]
    floor_chroma = np.sqrt(np.sum((bottom - 1 / 3) ** 2, axis=2))
    ceiling_tint = np.sqrt(np.sum((top - 1 / 3) ** 2, axis=2))
    blocks = cv2.resize(norm.astype(np.float32), (8, 6), interpolation=cv2.INTER_AREA)
    return {
        "filename": path.name,
        "room_median": float(np.median(y[diffuse])) if np.any(diffuse) else 0.0,
        "window_highlight_proxy_percent": float(np.mean(y > 1.0) * 100.0),
        "warm_cast": float(mean[0] - mean[2]),
        "green_cast": float(mean[1] - (mean[0] + mean[2]) * 0.5),
        "cyan_cast": float((mean[1] + mean[2]) * 0.5 - mean[0]),
        "floor_chroma": float(np.percentile(floor_chroma, 75)),
        "ceiling_tint": float(np.percentile(ceiling_tint, 75)),
        "black_occupancy": float(np.mean(y < max(float(np.median(y[diffuse])) / 5.0, 1e-5)) * 100.0) if np.any(diffuse) else 0.0,
        "mixed_lighting": float(np.mean(np.std(blocks, axis=(0, 1)))),
        "neutral_score": float(np.linalg.norm(mean - 1 / 3) + abs(float(np.median(y[diffuse])) - 0.18)) if np.any(diffuse) else 99.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    sources = sorted(args.input.glob("*.dng"))
    records = []
    for i, path in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] measuring {path.name}", flush=True)
        try:
            records.append(_metrics(path))
        except Exception as exc:
            records.append({"filename": path.name, "error": f"{type(exc).__name__}: {exc}"})
    valid = [r for r in records if "error" not in r]
    selected: dict[str, list[str]] = {}
    criteria = {
        "darkest_room": ("room_median", False), "highest_window_clipping": ("window_highlight_proxy_percent", True),
        "largest_warm_cast": ("warm_cast", True), "largest_green_cast": ("green_cast", True),
        "largest_cyan_cast": ("cyan_cast", True), "strongest_floor_chroma": ("floor_chroma", True),
        "strongest_ceiling_tint": ("ceiling_tint", True), "black_wall_proxy": ("black_occupancy", True),
        "largest_bright_window_area": ("window_highlight_proxy_percent", True), "most_mixed_lighting": ("mixed_lighting", True),
        "most_neutral_already_good": ("neutral_score", False),
    }
    for reason, (key, reverse) in criteria.items():
        ranked = sorted(valid, key=lambda r: float(r[key]), reverse=reverse)
        for item in ranked[:2]:
            selected.setdefault(item["filename"], []).append(reason)
    for item in sorted(valid, key=lambda r: r["filename"]):
        if len(selected) >= min(args.count, len(valid)):
            break
        selected.setdefault(item["filename"], []).append("diversity_fill")
    chosen = list(selected)[: args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    link_dir = args.output / "input"
    link_dir.mkdir(exist_ok=True)
    for name in chosen:
        link = link_dir / name
        if not link.exists():
            link.symlink_to((args.input / name).resolve())
    report = {"measured_count": len(valid), "selected_count": len(chosen), "selected": [
        {"filename": name, "reasons": selected[name], "metrics": next(r for r in valid if r["filename"] == name)} for name in chosen
    ], "failures": [r for r in records if "error" in r]}
    (args.output / "selection_report.json").write_text(json.dumps(report, indent=2))
    print(f"Selected {len(chosen)} representative DNGs")
    return 0 if len(chosen) >= min(args.count, len(valid)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
