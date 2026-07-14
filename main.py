#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from engine.segmentation import SceneSegmenter, SegmentationResult
from engine.utils import (
    collect_images, read_rgb, resize_for_analysis, clear_generated_outputs,
    cache_key, remove_stale_cache, analyze_image, save_mask_png,
    load_mask_png, make_comparison, make_scored_comparison, make_contact_sheet,
)
from engine.refine import refine_masks
from engine.scene import build_scene
from engine.tone_classify import classify_tones
from engine.wb import (
    conservative_white_balance,
    semantic_white_balance,
    global_safe_white_balance,
    global_pass1_wb,
)
from engine.profiles import detect_room_profile
from engine.tonal import adaptive_per_class_exposure, global_safe_exposure
from engine.exposure_fusion import exposure_fusion
from engine.mvp_pipeline import process_mvp_core
from engine.windows2 import treat_window_zones
from engine.materials2 import restore_protected_chroma
from engine.finish import natural_finish
from engine.quality2 import evaluate
from engine.exporter import save_jpeg
from engine.scoring import build_checklist, summarize_batch
from engine.debug_stages import DebugStageRecorder, array_sha256


CLASS_NAMES = [
    "wall", "ceiling", "floor", "window", "curtain", "door",
    "cabinet", "mirror", "sofa", "table", "chair", "bed",
    "rug", "plant", "painting", "lamp",
]


def load_settings(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_rgb_jpeg(rgb: np.ndarray, path: Path, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )


def load_or_segment(
    segmenter: SceneSegmenter,
    rgb: np.ndarray,
    image_path: Path,
    project: Path,
    input_root: Path,
    settings: dict,
    no_cache: bool,
) -> tuple[np.ndarray, dict[str, int], dict]:
    key = cache_key(image_path, input_root)
    label_path = project / "output/cache/labels" / f"{key}_labels.png"
    working, working_size = resize_for_analysis(rgb, int(settings["max_side"]))

    if settings.get("use_mask_cache", True) and not no_cache and label_path.exists():
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(label_path)
        if label.shape != rgb.shape[:2]:
            label = cv2.resize(label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        segmenter._load()
        return label, segmenter.name_to_id.copy(), {
            "cache_key": key,
            "label_cache": "hit",
            "working_resolution": list(working_size),
            "inference_seconds": 0.0,
        }

    result = segmenter.segment(working)
    label = result.label_map
    if label.shape != rgb.shape[:2]:
        label = cv2.resize(label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(label_path), label)
    return label, result.name_to_id.copy(), {
        "cache_key": key,
        "label_cache": "miss",
        "working_resolution": list(working_size),
        "inference_seconds": result.inference_seconds,
    }


def load_or_refine(
    rgb: np.ndarray,
    label_map: np.ndarray,
    name_to_id: dict[str, int],
    image_path: Path,
    project: Path,
    input_root: Path,
    settings: dict,
    no_cache: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    key = cache_key(image_path, input_root)
    mask_dir = project / "output/cache/masks"
    expected = {name: mask_dir / f"{key}_{name}.png" for name in CLASS_NAMES if name in name_to_id}
    cache_ok = settings.get("use_mask_cache", True) and not no_cache and expected and all(p.exists() for p in expected.values())
    if cache_ok:
        masks = {name: load_mask_png(path, rgb.shape[:2]) for name, path in expected.items()}
        return masks, {"mask_cache": "hit", "classes": sorted(masks)}

    selected = {name: name_to_id[name] for name in CLASS_NAMES if name in name_to_id}
    masks, log = refine_masks(rgb, label_map, selected)
    for name, mask in masks.items():
        save_mask_png(mask, mask_dir / f"{key}_{name}.png")
    log["mask_cache"] = "miss"
    return masks, log



def main() -> int:
    parser = argparse.ArgumentParser(description="MyEstatePics Semantic Engine v3 Phase 2")
    parser.add_argument("--project", default=".")
    parser.add_argument("--input", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--pipeline",
        choices=("mvp", "legacy"),
        default=None,
        help="Select the new MVP pipeline or the previous legacy pipeline. "
             "Defaults to config workflow.pipeline_mode.",
    )
    parser.add_argument(
        "--debug-stages",
        action="store_true",
        help="Save lossless pipeline stages and protected-material metrics.",
    )
    args = parser.parse_args()

    run_started = time.perf_counter()
    project = Path(args.project).expanduser().resolve()
    settings = load_settings(project / "config/settings.json")
    pipeline_mode = args.pipeline or settings.get("workflow", {}).get("pipeline_mode", "legacy")
    input_root = Path(args.input).expanduser().resolve() if args.input else project / settings["input_folder"]
    images = collect_images(input_root)
    if args.limit is not None:
        images = images[:max(0, args.limit)]
    if not images:
        print(f"No images found in: {input_root}")
        return 1

    clear_generated_outputs(project)
    valid_keys = {cache_key(path, input_root) for path in images}
    remove_stale_cache(project, valid_keys)

    segmenter = SceneSegmenter()
    passed = reviewed = failed = 0
    failed_files = []
    scored_records = []
    comparison_images = []

    for index, path in enumerate(images, 1):
        started = time.perf_counter()
        try:
            original = read_rgb(path)
            original_shape = original.shape[:2]
            label_map, name_to_id, seg_log = load_or_segment(
                segmenter, original, path, project, input_root, settings, args.no_cache
            )
            refined, refine_log = load_or_refine(
                original, label_map, name_to_id, path, project, input_root, settings, args.no_cache
            )
            scene = build_scene(refined, original.shape[:2])
            tone_settings = settings.get("targets", {})

            debug_recorder = None
            if args.debug_stages:
                debug_recorder = DebugStageRecorder(
                    project=project,
                    image_stem=path.stem,
                    scene=scene,
                )
                debug_recorder.save_stage(
                    "00_original",
                    "Original",
                    original,
                )

            if pipeline_mode == "mvp":
                # Sprint 2 Phase 3: one bounded WB pass followed by global
                # exposure fusion. Semantic masks are guardrails only; they do
                # not set exposure targets or locally relight the room.
                mvp = process_mvp_core(original, scene, settings)
                wb = mvp["wb"]
                wb_log = mvp["wb_log"]
                exposed = mvp["exposed"]
                exposure_log = mvp["exposure_log"]
                protected = mvp["protected"]
                materials_log = mvp["materials_log"]
                tones = mvp["tones"]
                tones_log = mvp["tones_log"]
                analysis = mvp["analysis"]
                profile_name = mvp["profile_name"]
                window_log = mvp["window_log"]

                if debug_recorder is not None:
                    debug_recorder.save_stage("01_after_wb", "After white balance", wb)

                if debug_recorder is not None:
                    debug_recorder.save_stage("02_after_exposure", "After exposure fusion", exposed)

                if debug_recorder is not None:
                    debug_recorder.save_stage("03_after_window", "Window stage skipped", exposed)
                    debug_recorder.save_stage(
                        "04_after_material_restore",
                        "After material guardrail",
                        protected,
                    )
                quality_reference = wb
            else:
                # Previous production path retained intact for rollback and
                # A/B comparisons via --pipeline legacy.
                pass1, pass1_log = global_pass1_wb(original, scene)
                profile_name, profile_targets, profile_dynamics = detect_room_profile(scene, pass1)
                tone_settings = {**tone_settings, **profile_targets}
                tones, tones_log = classify_tones(pass1, scene, tone_settings)
                analysis = analyze_image(pass1).to_dict()
                analysis["furnishing_protection"] = float(tone_settings.get("furnishing_factor", 1.0))
                analysis["material_inherit_factor"] = float(settings.get("phase3", {}).get("material_inherit_factor", 1.0))
                analysis.update(profile_dynamics)
                analysis["room_profile"] = profile_name

                if scene.route == "SEMANTIC":
                    wb, wb_log = semantic_white_balance(pass1, scene, tones)
                    wb_log = {**pass1_log, **wb_log}
                    if debug_recorder is not None:
                        debug_recorder.save_stage("01_after_wb", "After white balance", wb)

                    exposed, exposure_log = adaptive_per_class_exposure(wb, scene, tones, analysis)
                    if debug_recorder is not None:
                        debug_recorder.save_stage("02_after_exposure", "After exposure", exposed)

                    windows, window_log = treat_window_zones(exposed, scene)
                    if debug_recorder is not None:
                        debug_recorder.save_stage("03_after_window", "After window treatment", windows)

                    protected, materials_log = restore_protected_chroma(pass1, windows, scene)
                    if debug_recorder is not None:
                        debug_recorder.save_stage(
                            "04_after_material_restore",
                            "After material restoration",
                            protected,
                        )
                else:
                    wb, wb_log = global_safe_white_balance(original)
                    wb_log = {**pass1_log, **wb_log}
                    if debug_recorder is not None:
                        debug_recorder.save_stage("01_after_wb", "After white balance", wb)

                    protected, exposure_log = global_safe_exposure(wb, analysis)
                    if debug_recorder is not None:
                        debug_recorder.save_stage("02_after_exposure", "After exposure", protected)
                        debug_recorder.save_stage("03_after_window", "After window treatment", protected)
                        debug_recorder.save_stage(
                            "04_after_material_restore",
                            "After material restoration",
                            protected,
                        )

                    window_log = {"status": "skipped_global_safe"}
                    materials_log = {"status": "skipped_global_safe"}
                quality_reference = pass1

            final, finish_log = natural_finish(protected)
            if debug_recorder is not None:
                debug_recorder.save_stage("05_after_finish", "After final finish", final)

            if final.shape[:2] != original_shape:
                raise RuntimeError(f"Resolution changed: {original_shape} -> {final.shape[:2]}")

            quality = evaluate(quality_reference, final, scene, tones)

            # Save temporarily so output-quality checks can inspect the actual export settings.
            provisional_path = project / "output/review" / path.name
            export_log = save_jpeg(
                final,
                provisional_path,
                float(settings["maximum_output_mb"]),
                int(settings["jpeg_quality"]),
                expected_shape=original_shape,
            )

            checklist = build_checklist(
                wb_log=wb_log,
                window_log=window_log,
                quality=quality,
                export_log=export_log,
                route=scene.route,
            )

            destination_root = project / (
                "output/final" if checklist["decision"] == "PASS" else "output/review"
            )
            output_path = destination_root / path.name
            if output_path != provisional_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                provisional_path.replace(output_path)

            debug_log = None
            if debug_recorder is not None:
                final_hash_before_debug_report = array_sha256(final)
                final_hash_after_debug_report = array_sha256(final)
                debug_log = debug_recorder.finalize(
                    final_before_debug_hash=final_hash_before_debug_report,
                    final_after_debug_hash=final_hash_after_debug_report,
                )

            comparison = make_scored_comparison(
                original,
                final,
                path.name,
                checklist,
            )
            comparison_path = project / "output/comparisons" / f"{path.stem}_compare.jpg"
            save_rgb_jpeg(comparison, comparison_path, 90)
            comparison_images.append((path.name, comparison))

            report = {
                "filename": path.name,
                "pipeline_mode": pipeline_mode,
                "route": scene.route,
                "route_reasons": scene.reasons,
                "segmentation": seg_log,
                "refinement": refine_log,
                "coverage": scene.coverage,
                "tones": tones_log,
                "analysis": analysis,
                "room_profile": profile_name,
                "white_balance": wb_log,
                "exposure": exposure_log,
                "windows": window_log,
                "materials": materials_log,
                "finish": finish_log,
                "debug_stages": debug_log,
                "quality": quality,
                "checklist": checklist,
                "export": export_log,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            log_path = project / "output/logs" / f"{path.stem}.json"
            log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            scored_records.append(report)
            if checklist["decision"] == "PASS":
                passed += 1
            else:
                reviewed += 1

            wall_info = tones_log.get("wall_components", [])
            wall_text = wall_info[0]["class"].lower() if wall_info else "none"
            before_wall = wall_info[0]["median"] if wall_info else 0.0
            after_wall = quality["achieved_targets"].get("wall", {}).get("after", 0.0)
            ceiling_after = quality["achieved_targets"].get("ceiling", {}).get("after", 0.0)
            print(
                f"[{index}/{len(images)}] {checklist['decision']} score={checklist['overall_score']} route={scene.route} "
                f"wall({wall_text}) {before_wall:.2f}->{after_wall:.2f} "
                f"ceil {ceiling_after:.2f} {path.name} "
                f"({report['elapsed_seconds']:.1f}s)"
            )
        except Exception as exc:
            failed += 1
            failed_files.append(path.name)
            print(f"[{index}/{len(images)}] FAILED {path.name}: {type(exc).__name__}: {exc}")

    score_summary = summarize_batch(scored_records)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": len(images),
        "passed": passed,
        "reviewed": reviewed,
        "failed": failed,
        "failed_files": failed_files,
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "engine_scorecard": score_summary,
    }
    (project / "output/logs/_batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report_dir = project / "output/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "MYESTATEPICS ENGINE RUN SCORECARD",
        "=" * 44,
        f"Images processed: {len(images)}",
        f"Passed: {passed}",
        f"Review: {reviewed}",
        f"Failed: {failed}",
        f"Overall engine score: {score_summary['overall_engine_score']}/100",
        "",
    ]
    for name, score in score_summary.get("averages", {}).items():
        report_lines.append(f"{name}: {score}/100")
    (report_dir / "run_scorecard.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    (report_dir / "run_scorecard.json").write_text(
        json.dumps(score_summary, indent=2),
        encoding="utf-8",
    )

    contact_sheet = make_contact_sheet(comparison_images)
    if contact_sheet is not None:
        save_rgb_jpeg(
            contact_sheet,
            project / "output/comparisons/_batch_contact_sheet.jpg",
            88,
        )

    print("=" * 72)
    print(f"PIPELINE: {pipeline_mode.upper()}")
    print(f"PASS: {passed} | REVIEW: {reviewed} | FAILED: {failed}")
    print(f"ENGINE SCORE: {score_summary['overall_engine_score']}/100")
    print("Review output/comparisons/_batch_contact_sheet.jpg")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
