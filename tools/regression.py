#!/usr/bin/env python3
"""
MyEstatePics regression harness (v4.1).

Runs the full correction pipeline (segmentation bypassed via constructed
masks) over a fixed suite of synthetic rooms that encode every historical
failure mode, measures the results, and REJECTS the build on any regression.

Usage:  python tools/regression.py [--out output/regression]
Exit codes: 0 = all gates pass, 1 = regression detected.

Produces in the output folder:
  contact_sheet.jpg      before/after grid with per-scenario verdicts
  regression_report.json brightness / WB / material dE / geometry / timing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.scene import build_scene
from engine.tone_classify import classify_tones
from engine.wb import global_pass1_wb, semantic_white_balance
from engine.tonal import adaptive_per_class_exposure
from engine.materials2 import restore_protected_chroma
from engine.finish import natural_finish
from engine.profiles import detect_room_profile

H, W = 900, 1200


def _lum(a: np.ndarray) -> np.ndarray:
    f = a.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _chroma_mean(a: np.ndarray, m: np.ndarray) -> float:
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    c = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    return float(np.mean(c[m]))


def _hue_median(a: np.ndarray, m: np.ndarray) -> float:
    return float(np.median(cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 0][m])) * 2.0


def _three_band_masks() -> dict[str, np.ndarray]:
    masks = {k: np.zeros((H, W), np.float32) for k in ["ceiling", "wall", "floor"]}
    masks["ceiling"][:270] = 1.0
    masks["wall"][270:630] = 1.0
    masks["floor"][630:] = 1.0
    return masks


def _room(ceiling, wall, floor, cast=(1.0, 1.0, 1.0), seed=3, wall_grad=True):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    c = np.array(cast, np.float32)
    img[:270] = np.array(ceiling, np.float32) * c + rng.normal(0, 0.012, (270, W, 3))
    img[270:630] = np.array(wall, np.float32) * c + rng.normal(0, 0.010, (360, W, 3))
    img[630:] = np.array(floor, np.float32) * c + rng.normal(0, 0.012, (270, W, 3))
    if wall_grad:
        img[270:630] += np.linspace(-0.05, 0.05, W)[None, :, None]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def scenarios() -> list[dict]:
    g = [0.42, 0.42, 0.42]
    return [
        {
            "name": "tungsten_underexposed",
            "rgb": _room([0.42] * 3, [0.34] * 3, [0.30, 0.20, 0.12], cast=(1.18, 1.0, 0.72)),
            "gates": {
                "residual_cast_max": 0.015,
                "image_mean_min": 0.46,
                "ceiling_median_min": 0.72,
                "wall_texture_retained_min": 0.80,
                "floor_hue_drift_max_deg": 4.0,
            },
        },
        {
            "name": "green_cast",
            "rgb": _room(g, [0.34] * 3, [0.30, 0.20, 0.12], cast=(0.88, 1.14, 0.90)),
            "gates": {"residual_cast_max": 0.015, "image_mean_min": 0.46},
        },
        {
            "name": "cyan_cast",
            "rgb": _room(g, [0.34] * 3, [0.30, 0.20, 0.12], cast=(0.84, 1.05, 1.12)),
            "gates": {"residual_cast_max": 0.015, "image_mean_min": 0.46},
        },
        {
            "name": "black_paper_windows",  # windowless dim room: THE priority case
            "rgb": _room([0.30] * 3, [0.24] * 3, [0.26, 0.18, 0.11], seed=11),
            "gates": {
                "image_mean_min": 0.42,
                "ceiling_median_min": 0.62,
                "wall_texture_retained_min": 0.80,
            },
        },
        {
            "name": "black_wall_moody",  # z-66 class
            "rgb": _room([0.34, 0.32, 0.31], [0.045, 0.045, 0.050], [0.16, 0.09, 0.06], seed=7, wall_grad=False),
            "gates": {
                "wall_median_after_max": 0.10,       # black stays black
                "wall_chroma_growth_max": 1.5,       # no pink bloom (abs units)
                "floor_lift_ratio_max": 1.55,        # no fire-red doubling
                "floor_chroma_ratio_max": 1.15,
            },
        },
        {
            # z-66/z-17 real-photo lesson: ceilings carry color BOUNCE from
            # floors/walls. Lifting must CLEAN the tint, never amplify it.
            "name": "bounce_tinted_ceiling",
            "rgb": _room([0.34, 0.295, 0.285], [0.30] * 3, [0.28, 0.15, 0.09], seed=13),
            "gates": {
                "ceiling_chroma_must_drop": True,
                "ceiling_median_min": 0.55,
            },
        },
        {
            "name": "already_good_noop",
            "rgb": _room([0.78] * 3, [0.60] * 3, [0.45, 0.34, 0.24], seed=5),
            "gates": {
                "image_mean_delta_max": 0.06,        # near no-op
                "floor_hue_drift_max_deg": 3.0,
            },
        },
    ]


def run_pipeline(rgb: np.ndarray, targets: dict) -> tuple[np.ndarray, dict, float]:
    masks = _three_band_masks()
    scene = build_scene(masks, (H, W))
    t0 = time.perf_counter()
    pass1, _ = global_pass1_wb(rgb, scene)
    name, p_targets, p_dyn = detect_room_profile(scene, pass1)
    tones, _ = classify_tones(pass1, scene, {**targets, **p_targets})
    wb, _ = semantic_white_balance(pass1, scene, tones)
    analysis = {"material_inherit_factor": 1.0, **p_dyn}
    exposed, _ = adaptive_per_class_exposure(wb, scene, tones, analysis)
    protected, _ = restore_protected_chroma(pass1, exposed, scene)
    final, _ = natural_finish(protected)
    elapsed = time.perf_counter() - t0
    return final, {"profile": name}, elapsed


def evaluate_gates(rgb, final, gates) -> tuple[dict, list[str]]:
    masks = _three_band_masks()
    cm, wm, fm = (masks[k] > 0.5 for k in ["ceiling", "wall", "floor"])
    y0, y1 = _lum(rgb), _lum(final)
    f = final.astype(np.float32) / 255.0
    ceil_rgb = f[:270].reshape(-1, 3).mean(axis=0)

    metrics = {
        "ceiling_chroma_before": _chroma_mean(rgb, cm),
        "ceiling_chroma_after": _chroma_mean(final, cm),
        "residual_cast": float(ceil_rgb.max() - ceil_rgb.min()),
        "image_mean_before": float(y0.mean()),
        "image_mean_after": float(y1.mean()),
        "image_mean_delta": float(abs(y1.mean() - y0.mean())),
        "ceiling_median_after": float(np.median(y1[cm])),
        "wall_median_after": float(np.median(y1[wm])),
        "wall_texture_retained": float(y1[wm].std() / max(y0[wm].std(), 1e-6)),
        "wall_chroma_growth": _chroma_mean(final, wm) - _chroma_mean(rgb, wm),
        "floor_lift_ratio": float(np.median(y1[fm]) / max(np.median(y0[fm]), 1e-6)),
        "floor_chroma_ratio": _chroma_mean(final, fm) / max(_chroma_mean(rgb, fm), 1e-6),
        "floor_hue_drift_deg": abs(_hue_median(final, fm) - _hue_median(rgb, fm)),
        # geometry lock: dimensions identical + zero global shift
        "geometry_dims_ok": final.shape == rgb.shape,
        "geometry_shift_px": float(np.hypot(*cv2.phaseCorrelate(
            _lum(rgb).astype(np.float64), _lum(final).astype(np.float64))[0])),
    }

    failures = []
    checks = {
        "residual_cast_max": lambda v: metrics["residual_cast"] <= v,
        "image_mean_min": lambda v: metrics["image_mean_after"] >= v,
        "image_mean_delta_max": lambda v: metrics["image_mean_delta"] <= v,
        "ceiling_median_min": lambda v: metrics["ceiling_median_after"] >= v,
        "wall_median_after_max": lambda v: metrics["wall_median_after"] <= v,
        "wall_texture_retained_min": lambda v: metrics["wall_texture_retained"] >= v,
        "wall_chroma_growth_max": lambda v: metrics["wall_chroma_growth"] <= v,
        "floor_lift_ratio_max": lambda v: metrics["floor_lift_ratio"] <= v,
        "floor_chroma_ratio_max": lambda v: metrics["floor_chroma_ratio"] <= v,
        "floor_hue_drift_max_deg": lambda v: metrics["floor_hue_drift_deg"] <= v,
        "ceiling_chroma_must_drop": lambda v: (not v) or (
            metrics["ceiling_chroma_after"] < metrics["ceiling_chroma_before"]),
    }
    for gate, value in gates.items():
        if not checks[gate](value):
            failures.append(f"{gate}={value} violated")
    if not metrics["geometry_dims_ok"]:
        failures.append("geometry: dimensions changed")
    if metrics["geometry_shift_px"] > 0.5:
        failures.append(f"geometry: global shift {metrics['geometry_shift_px']:.2f}px")
    return metrics, failures


def contact_sheet(rows: list[tuple[str, np.ndarray, np.ndarray, bool]], path: Path):
    tiles = []
    for name, before, after, ok in rows:
        b = cv2.resize(before, (420, 315))
        a = cv2.resize(after, (420, 315))
        pair = np.hstack([b, a])
        bar = np.zeros((34, pair.shape[1], 3), np.uint8)
        color = (60, 200, 60) if ok else (60, 60, 230)
        cv2.putText(bar, f"{name}  |  {'PASS' if ok else 'FAIL'}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        tiles.append(np.vstack([bar, pair]))
    sheet = np.vstack(tiles)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/regression")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = json.loads(Path("config/settings.json").read_text()).get("targets", {})

    report, rows, all_ok = {"scenarios": {}, "build": "4.1.0-rc1"}, [], True
    for sc in scenarios():
        final, info, elapsed = run_pipeline(sc["rgb"], targets)
        metrics, failures = evaluate_gates(sc["rgb"], final, sc["gates"])
        ok = not failures
        all_ok &= ok
        report["scenarios"][sc["name"]] = {
            "profile_detected": info["profile"],
            "elapsed_seconds": round(elapsed, 3),
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
            "failures": failures,
            "verdict": "PASS" if ok else "FAIL",
        }
        rows.append((f"{sc['name']} [{info['profile']}]", sc["rgb"], final, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {sc['name']:24s} profile={info['profile']:10s} "
              f"{elapsed:.2f}s" + (f"  -> {failures}" if failures else ""))

    contact_sheet(rows, out / "contact_sheet.jpg")
    report["verdict"] = "PASS" if all_ok else "FAIL: BUILD REJECTED"
    (out / "regression_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{report['verdict']}  |  report: {out}/regression_report.json")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
