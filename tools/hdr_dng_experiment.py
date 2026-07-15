#!/usr/bin/env python3
from __future__ import annotations

"""Controlled high-bit-depth DNG versus Phase A JPEG experiment.

This runner is intentionally separate from ``main.py``. It does not change
the production JPEG path or its targets. The DNG working image stays float32
until the final 16-bit PNG and JPEG review exports are written.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.dng import read_lightroom_hdr_dng
from engine.exposure_fusion import ExposureFusionConfig
from engine.scene import build_scene
from engine.utils import load_mask_png


MATCHES = {"input-tiff-9.dng": "z-17.jpg"}


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _encoded_to_linear(rgb: np.ndarray) -> np.ndarray:
    positive = np.maximum(rgb.astype(np.float32), 0.0)
    return np.where(
        positive <= 0.04045,
        positive / 12.92,
        np.power((positive + 0.055) / 1.055, 2.4),
    )


def _linear_to_encoded(linear: np.ndarray) -> np.ndarray:
    positive = np.maximum(linear.astype(np.float32), 0.0)
    return np.where(
        positive <= 0.0031308,
        12.92 * positive,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _robust_illuminant(linear: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    samples = linear[valid]
    if not samples.size:
        return np.ones(3, np.float32), 0
    lo, hi = np.percentile(samples, (10, 90), axis=0)
    trimmed = samples[np.all((samples >= lo) & (samples <= hi), axis=1)]
    if len(trimmed) >= max(200, len(samples) // 5):
        samples = trimmed
    illum = np.power(np.mean(np.power(np.clip(samples, 1e-6, 1.0), 6.0), axis=0), 1.0 / 6.0)
    return illum.astype(np.float32), int(len(samples))


def _gains(illum: np.ndarray) -> np.ndarray:
    target = float(np.exp(np.mean(np.log(np.maximum(illum, 1e-6)))))
    return (target / np.maximum(illum, 1e-6)).astype(np.float32)


def _candidate(rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    clipped = np.clip(rgb, 0.0, 1.0)
    linear = _encoded_to_linear(clipped)
    y = _luma(linear)
    peak, low = clipped.max(axis=2), clipped.min(axis=2)
    saturation = (peak - low) / np.maximum(peak, 1e-6)
    valid = (y > 0.035) & (y < 0.88) & (saturation < 0.42)
    if mask is not None:
        valid &= mask > 0.5
    return valid


def _half_agreement(linear: np.ndarray, valid: np.ndarray) -> float | None:
    h, w = valid.shape
    left, right = valid.copy(), valid.copy()
    left[:, w // 2 :] = False
    right[:, : w // 2] = False
    li, lc = _robust_illuminant(linear, left)
    ri, rc = _robust_illuminant(linear, right)
    if lc < 400 or rc < 400:
        return None
    return float(np.max(np.abs(_gains(li) - _gains(ri))))


def _phase_a_white_balance(rgb: np.ndarray, scene) -> tuple[np.ndarray, dict]:
    linear = _encoded_to_linear(rgb)
    choice = None
    ceiling = scene.masks.get("ceiling")
    if ceiling is not None:
        valid = _candidate(rgb, ceiling)
        agreement = _half_agreement(linear, valid)
        if np.count_nonzero(valid) >= 1200 and scene.coverage.get("ceiling", 0) >= 3 and agreement is not None and agreement <= 0.055:
            choice = (valid, "ceiling", "HIGH", agreement)
    if choice is None and scene.masks.get("wall") is not None:
        upper = scene.masks["wall"].copy()
        upper[int(rgb.shape[0] * 0.45) :] = 0
        valid = _candidate(rgb, upper)
        agreement = _half_agreement(linear, valid)
        if np.count_nonzero(valid) >= 1200 and (agreement is None or agreement <= 0.055 * 1.35):
            choice = (valid, "upper_wall", "MEDIUM", agreement)
    if choice is None and scene.masks.get("structure") is not None:
        valid = _candidate(rgb, scene.masks["structure"])
        if np.count_nonzero(valid) >= 1200:
            choice = (valid, "structure", "LOW", None)
    if choice is None:
        choice = (_candidate(rgb, None), "global_shades_of_gray", "LOW", None)
    valid, reference, confidence, agreement = choice
    illum, count = _robust_illuminant(linear, valid)
    raw = _gains(illum)
    limit = {"HIGH": 0.12, "MEDIUM": 0.10, "LOW": 0.08}[confidence]
    gains = np.clip(raw, 1 - limit, 1 + limit).astype(np.float32)
    applied = bool(np.max(np.abs(gains - 1.0)) > 0.025)
    out = _linear_to_encoded(linear * gains.reshape(1, 1, 3)) if applied else rgb.copy()
    return out, {
        "applied": applied,
        "reference_used": reference,
        "confidence": confidence,
        "reference_pixel_count": count,
        "half_agreement": agreement,
        "raw_gains": raw.tolist(),
        "gains": gains.tolist(),
        "working_precision": "float32",
    }


def _phase_a_wall_consistency(rgb: np.ndarray, scene) -> tuple[np.ndarray, dict]:
    wall = scene.masks.get("wall")
    if wall is None or np.count_nonzero(wall > 0.5) < 2400:
        return rgb.copy(), {"applied": False, "reason": "insufficient_wall"}
    clipped = np.clip(rgb, 0, 1).astype(np.float32)
    lab = cv2.cvtColor(clipped, cv2.COLOR_RGB2LAB)
    l, ab = lab[..., 0] / 100.0, lab[..., 1:3]
    chroma = np.sqrt(np.sum(ab * ab, axis=2))
    valid = (wall > 0.55) & (l > 0.15) & (l < 0.92) & (chroma < 22.0)
    if np.count_nonzero(valid) < 2400:
        return rgb.copy(), {"applied": False, "reason": "no_neutral_wall_family"}
    target = np.median(ab[valid], axis=0).astype(np.float32)
    weight = np.clip(wall, 0, 1) * valid.astype(np.float32)
    h, w = wall.shape
    scale = min(1.0, 640.0 / max(h, w))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small_weight = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA)
    small_ab = cv2.resize(ab, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = max(12.0, max(sh, sw) / 9.0)
    denominator = cv2.GaussianBlur(small_weight, (0, 0), sigma) + 1e-5
    local = np.stack(
        [
            cv2.GaussianBlur(small_ab[..., c] * small_weight, (0, 0), sigma) / denominator
            for c in range(2)
        ], axis=2,
    )
    local = cv2.resize(local, (w, h), interpolation=cv2.INTER_LINEAR)
    divergence = np.sqrt(np.sum((local - target) ** 2, axis=2))
    p90 = float(np.percentile(divergence[valid], 90))
    if p90 < 3:
        return rgb.copy(), {"applied": False, "reason": "wall_chroma_consistent", "divergence_p90": p90}
    gate = _smoothstep(2, 9, divergence)
    strength = min(0.42, 0.28 + max(0, p90 - 4) * 0.020)
    amount = strength * np.clip(wall, 0, 1) * gate * valid.astype(np.float32)
    adjusted = lab.copy()
    adjusted[..., 1:3] = ab - (local - target) * amount[..., None]
    corrected = cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB)
    out = rgb.copy()
    affected = amount > 0.001
    out[affected] = corrected[affected]
    return out, {"applied": True, "strength": strength, "divergence_p90": p90, "affected_percent": float(np.mean(affected) * 100)}


def _scene_exposure_plan(rgb: np.ndarray, scene) -> tuple[float, dict]:
    cfg = ExposureFusionConfig()
    clipped = np.clip(rgb, 0, 1)
    y = _luma(clipped)
    structure, window = scene.masks.get("structure"), scene.masks.get("window")
    room = np.ones(y.shape, bool) if structure is None else structure > 0.45
    if window is not None:
        room &= window < 0.25
    room &= (y > 0.015) & (y < 0.95)
    if np.count_nonzero(room) < 512:
        room = (y > 0.015) & (y < 0.95)
    values = y[room]
    median = float(np.median(values))
    p25, p75 = [float(x) for x in np.percentile(values, (25, 75))]
    shadow = float(np.mean(values < 0.18) * 100)
    def sm(name):
        mask = scene.masks.get(name)
        if mask is None:
            return None
        valid = (mask > 0.55) & (y > 0.015) & (y < 0.95)
        return float(np.median(y[valid])) if np.count_nonzero(valid) >= 256 else None
    ceiling, wall = sm("ceiling"), sm("wall")
    window_area = float(np.mean(window > 0.5) * 100) if window is not None else 0.0
    dark_decor = wall is not None and ceiling is not None and wall < 0.42 * max(ceiling, 1e-4)
    if dark_decor:
        target, max_ev, profile = 0.42, 0.65, "dark_decor"
    elif median < 0.28 or shadow > 32:
        target, max_ev, profile = (0.50 if window_area >= 0.3 else 0.52), 0.75, "underexposed_room"
    elif median < 0.42:
        target, max_ev, profile = 0.50, 0.65, "dim_room"
    else:
        target, max_ev, profile = 0.50, 0.35, "balanced_room"
    required = float(np.log2(max(target, 1e-4) / max(median, 1e-4)))
    ev = float(np.clip(required, 0, max_ev))
    return ev, {"profile": profile, "room_median": median, "room_p25": p25, "room_p75": p75, "shadow_percent": shadow, "ceiling_median": ceiling, "wall_median": wall, "window_area_percent": window_area, "target_median": target, "required_ev": required, "applied_ev": ev, "maximum_ev": max_ev, "config": cfg.__dict__}


def _phase_a_exposure(rgb: np.ndarray, scene) -> tuple[np.ndarray, dict]:
    ev, log = _scene_exposure_plan(rgb, scene)
    if ev <= 0.001:
        return rgb.copy(), log
    cfg = ExposureFusionConfig()
    y = _luma(np.clip(rgb, 0, 1))
    black = _smoothstep(cfg.shadow_anchor_start, cfg.shadow_anchor_end, y)
    guard = 1 - _smoothstep(cfg.highlight_guard_start, cfg.highlight_guard_end, y)
    gain = 1 + (2**ev - 1) * black * guard
    # Extended highlights remain available in float32. Final review exports
    # clip display white, but the experiment metrics use this unclipped result.
    return rgb * gain[..., None], {**log, "working_precision": "float32", "extended_highlights_preserved": True}


def _material_guard(rgb_ref: np.ndarray, corrected: np.ndarray, scene, settings: dict) -> tuple[np.ndarray, dict]:
    cfg = settings.get("material_guardrails", {})
    wall = scene.masks.get("wall", np.zeros(rgb_ref.shape[:2], np.float32))
    cabinet = scene.masks.get("cabinet", np.zeros_like(wall))
    furnishings = scene.masks.get("furnishings", np.zeros_like(wall))
    protected = scene.masks.get("protected", np.zeros_like(wall))
    ref_y, out_y = _luma(np.clip(rgb_ref, 0, None)), _luma(np.clip(corrected, 0, None))
    wall_hold = wall * (1 - _smoothstep(0.12, 0.38, ref_y))
    material_hold = np.maximum.reduce([cabinet, furnishings, protected]) * (1 - _smoothstep(0.16, 0.42, ref_y))
    active = np.maximum(wall_hold, material_hold)
    dark_material = float(cfg.get("dark_material_max_lift", 0.075))
    dark_wall = float(cfg.get("dark_wall_max_lift", 0.055))
    limit = dark_material * (1 - wall_hold) + dark_wall * wall_hold
    excess = np.maximum(out_y - (ref_y + limit), 0)
    target = out_y - excess * _smoothstep(0.05, 0.55, active)
    out = corrected * (target / np.maximum(out_y, 1e-6))[..., None]
    return out.astype(np.float32), {"engine": "material_dark_surface_guardrails_v2_float", "floor_included": False, "held_area_percent": float(np.mean(active > 0.1) * 100), "working_precision": "float32"}


def _load_scene(mask_dir: Path, jpeg_stem: str, shape: tuple[int, int]):
    paths = list(mask_dir.glob(f"{jpeg_stem}_*_*.png"))
    if not paths:
        raise FileNotFoundError(f"No cached masks for {jpeg_stem}")
    masks = {}
    for path in paths:
        name = path.stem.rsplit("_", 1)[-1]
        masks[name] = load_mask_png(path, shape)
    return build_scene(masks, shape)


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)


def _save_rgb(rgb: np.ndarray, jpeg: Path, png16: Path | None = None) -> None:
    jpeg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(jpeg), cv2.cvtColor(_u8(rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if png16 is not None:
        png16.parent.mkdir(parents=True, exist_ok=True)
        u16 = np.clip(rgb * 65535 + 0.5, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(png16), cv2.cvtColor(u16, cv2.COLOR_RGB2BGR))


def _histogram(rgb: np.ndarray) -> list[int]:
    y = np.clip(_luma(np.clip(rgb, 0, 1)), 0, 1)
    return np.histogram(y, bins=256, range=(0, 1))[0].astype(int).tolist()


def _luminance_summary(rgb: np.ndarray) -> dict:
    y = _luma(np.clip(rgb, 0, None))
    return {f"p{p}": float(np.percentile(y, p)) for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)} | {"mean": float(np.mean(y)), "above_display_white_percent": float(np.mean(y > 1) * 100)}


def _material_drift(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> dict:
    b = cv2.cvtColor(np.clip(before, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    a = cv2.cvtColor(np.clip(after, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    valid = mask > 0.25
    if not np.any(valid):
        return {"hue_p95_degrees": 0.0, "saturation_p95": 0.0}
    hue = np.abs(a[..., 0] - b[..., 0]); hue = np.minimum(hue, 360 - hue)
    sat = np.abs(a[..., 1] - b[..., 1])
    return {"hue_p95_degrees": float(np.percentile(hue[valid], 95)), "saturation_p95": float(np.percentile(sat[valid], 95))}


def _comparison(left: np.ndarray, right: np.ndarray, labels: tuple[str, str], path: Path) -> None:
    font = ImageFont.load_default(size=28)
    tiles = []
    for arr in (left, right):
        im = Image.fromarray(_u8(arr), "RGB"); im.thumbnail((1000, 667), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (1000, 667), "black"); tile.paste(im, ((1000-im.width)//2, (667-im.height)//2)); tiles.append(tile)
    canvas = Image.new("RGB", (2000, 739), (15, 15, 15)); canvas.paste(tiles[0], (0, 72)); canvas.paste(tiles[1], (1000, 72))
    draw = ImageDraw.Draw(canvas); draw.text((20, 18), labels[0], font=font, fill="white"); draw.text((1020, 18), labels[1], font=font, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path, quality=95)


def _histogram_chart(series: dict[str, list[int]], path: Path) -> None:
    colors = [(80, 170, 255), (255, 170, 70), (100, 220, 130), (230, 90, 180)]
    canvas = Image.new("RGB", (1400, 700), (20, 20, 24)); draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=22)
    for y in (100, 250, 400, 550): draw.line((70, y, 1340, y), fill=(55, 55, 60), width=1)
    for idx, (name, values) in enumerate(series.items()):
        a=np.asarray(values,np.float64); a=np.log1p(a); a=a/max(float(a.max()),1)
        pts=[(70+i*(1270/255),620-float(v)*520) for i,v in enumerate(a)]
        draw.line(pts,fill=colors[idx],width=3); draw.text((85+idx*315,25),name,font=font,fill=colors[idx])
    draw.text((70,655),"black",font=font,fill="white"); draw.text((1280,655),"white",font=font,fill="white")
    path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--dng-input",required=True); parser.add_argument("--project",default="."); parser.add_argument("--out",default="output/hdr_dng_experiment")
    args=parser.parse_args(); project=Path(args.project).resolve(); folder=Path(args.dng_input).resolve(); out=project/args.out
    out.mkdir(parents=True,exist_ok=True); settings=json.loads((project/"config/settings.json").read_text())
    reports={}; decoded={}
    for path in sorted(folder.glob("*.dng")):
        dng=read_lightroom_hdr_dng(path); reports[path.name]=dng.metadata
        display=np.clip(dng.encoded_srgb,0,1); _save_rgb(display,out/"decoded_inputs"/f"{path.stem}.jpg",out/"decoded_inputs_16bit"/f"{path.stem}.png")
        if path.name in MATCHES: decoded[path.name]=dng
    (out/"dng_decoding_report.json").write_text(json.dumps(reports,indent=2))
    mapping={name:{"jpeg":MATCHES.get(name),"status":"matched_by_visual_scene_content_and_aspect_ratio" if name in MATCHES else "unmatched_current_five_image_development_set"} for name in reports}
    (out/"scene_mapping.json").write_text(json.dumps(mapping,indent=2))

    experiment={"mapping":mapping,"comparisons":{}}
    comparison_paths=[]
    for dng_name,jpeg_name in MATCHES.items():
        dng=decoded[dng_name]; stem=Path(jpeg_name).stem
        scene=_load_scene(project/"output/cache/masks",stem,dng.encoded_srgb.shape[:2])
        dng_input=dng.encoded_srgb
        wb,wb_log=_phase_a_white_balance(dng_input,scene)
        wb2,wall_log=_phase_a_wall_consistency(wb,scene)
        exposed,exposure_log=_phase_a_exposure(wb2,scene)
        final,material_log=_material_guard(wb2,exposed,scene,settings)
        _save_rgb(np.clip(final,0,1),out/"dng_results"/f"{stem}_from_{Path(dng_name).stem}.jpg",out/"dng_results_16bit"/f"{stem}_from_{Path(dng_name).stem}.png")

        jpeg_input=np.asarray(Image.open(project/"input"/jpeg_name).convert("RGB"),dtype=np.float32)/255
        jpeg_final_path=project/"output/phase_a/candidate_scene_wide_bounded/review"/jpeg_name
        if not jpeg_final_path.exists(): jpeg_final_path=project/"output/review"/jpeg_name
        jpeg_final=np.asarray(Image.open(jpeg_final_path).convert("RGB"),dtype=np.float32)/255
        final_display=np.clip(final,0,1)
        compare=out/"comparisons"/f"{stem}_jpeg_vs_hdr_dng.jpg"
        _comparison(jpeg_final,final_display,("CURRENT JPEG RESULT","HDR DNG RESULT — same Phase A targets"),compare); comparison_paths.append(compare)

        protected=scene.masks.get("protected",np.zeros(final.shape[:2],np.float32)); floor=scene.masks.get("floor",np.zeros_like(protected)); material=np.maximum(protected,floor)
        jpeg_scene=_load_scene(project/"output/cache/masks",stem,jpeg_input.shape[:2]); jpeg_material=np.maximum(jpeg_scene.masks.get("protected",np.zeros(jpeg_input.shape[:2],np.float32)),jpeg_scene.masks.get("floor",np.zeros(jpeg_input.shape[:2],np.float32)))
        data={
            "white_balance":wb_log|{"wall_chroma_consistency":wall_log},"exposure":exposure_log,"materials":material_log,
            "luminance":{"jpeg_input":_luminance_summary(jpeg_input),"jpeg_final":_luminance_summary(jpeg_final),"dng_input":_luminance_summary(dng_input),"dng_final":_luminance_summary(final)},
            "histograms":{"jpeg_input":_histogram(jpeg_input),"jpeg_final":_histogram(jpeg_final),"dng_input":_histogram(dng_input),"dng_final":_histogram(final)},
            "material_drift":{"jpeg":_material_drift(jpeg_input,jpeg_final,jpeg_material),"dng":_material_drift(dng_input,final,material)},
        }
        window=scene.masks.get("window",np.zeros(final.shape[:2],np.float32))>0.5
        if np.any(window):
            data["window_highlight_recoverability"]={"dng_linear_luminance_p99":float(np.percentile(_luma(dng.linear_srgb)[window],99)),"dng_linear_pixels_above_display_white_percent":float(np.mean(_luma(dng.linear_srgb)[window]>1)*100)}
        experiment["comparisons"][stem]=data
        _histogram_chart(data["histograms"],out/"histograms"/f"{stem}_histograms.png")

        regions={"darkest_floor":(.05,.58,.52,.98),"deepest_shadow":(0,.08,.27,.92),"ceiling_upper_wall":(.18,0,.88,.32),"window_highlight":(.20,.30,.70,.70),"window_vs_nonwindow_wall":(.12,.12,.95,.58)}
        for label,(x0,y0,x1,y1) in regions.items():
            for path_name,arr in (("jpeg",jpeg_final),("dng",final_display)):
                h,w=arr.shape[:2]; crop=arr[round(y0*h):round(y1*h),round(x0*w):round(x1*w)]
                _save_rgb(crop,out/"crops_100pct"/f"{stem}_{label}_{path_name}_100pct.jpg")
    (out/"experiment_metrics.json").write_text(json.dumps(experiment,indent=2))

    if comparison_paths:
        ims=[Image.open(p).convert("RGB") for p in comparison_paths]
        sheet=Image.new("RGB",(max(i.width for i in ims),sum(i.height for i in ims)),(10,10,10)); y=0
        for im in ims: sheet.paste(im,(0,y)); y+=im.height
        sheet.save(out/"hdr_dng_matched_contact_sheet.jpg",quality=95)
    print(json.dumps({"decoded":len(reports),"matched":len(comparison_paths),"output":str(out)},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
