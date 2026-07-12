
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


STAGE_ORDER = [
    ("00_original", "Original"),
    ("01_after_wb", "After white balance"),
    ("02_after_exposure", "After exposure"),
    ("03_after_window", "After window treatment"),
    ("04_after_material_restore", "After material restoration"),
    ("05_after_finish", "After final finish"),
]

DEFAULT_MATERIALS = [
    "floor",
    "cabinet",
    "furnishings",
    "rug",
]


def array_sha256(rgb: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(rgb)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        raise OSError(f"Could not write debug PNG: {path}")


def _mean_vector(values: np.ndarray, length: int) -> list[float]:
    if values.size == 0:
        return [0.0] * length
    result = np.mean(values, axis=0)
    return [float(x) for x in np.asarray(result).reshape(-1)[:length]]


def _rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    # OpenCV LAB is L [0..255], a/b centered at 128.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _material_mask(scene, name: str, shape: tuple[int, int]) -> np.ndarray:
    masks = scene.masks
    zeros = np.zeros(shape, dtype=np.float32)

    if name == "furnishings":
        return masks.get("furnishings", zeros).astype(np.float32)

    if name == "countertop":
        # ADE20K does not provide a reliable countertop class in this project.
        # Cabinet is the closest available semantic proxy, so countertop is not
        # fabricated or guessed here.
        return zeros

    return masks.get(name, zeros).astype(np.float32)


def compute_material_metrics(
    rgb: np.ndarray,
    mask: np.ndarray,
    original_reference: dict | None = None,
) -> dict:
    selected = mask > 0.5
    pixel_count = int(np.count_nonzero(selected))
    coverage = float(np.mean(selected) * 100.0)

    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "coverage_percent": coverage,
            "mean_rgb": [0.0, 0.0, 0.0],
            "mean_lab": [0.0, 0.0, 0.0],
            "mean_luminance": 0.0,
            "mean_chroma": 0.0,
            "mean_hue_degrees": 0.0,
            "delta_e_from_original": 0.0,
            "hue_rotation_degrees": 0.0,
        }

    rgb_pixels = rgb[selected].astype(np.float32)
    lab = _rgb_to_lab_float(rgb)
    lab_pixels = lab[selected]

    rgb_f = rgb.astype(np.float32) / 255.0
    luminance = (
        0.2126 * rgb_f[..., 0]
        + 0.7152 * rgb_f[..., 1]
        + 0.0722 * rgb_f[..., 2]
    )

    mean_rgb = _mean_vector(rgb_pixels, 3)
    mean_lab = _mean_vector(lab_pixels, 3)

    a_centered = lab_pixels[:, 1] - 128.0
    b_centered = lab_pixels[:, 2] - 128.0
    chroma = np.sqrt(a_centered * a_centered + b_centered * b_centered)
    hue = np.degrees(np.arctan2(b_centered, a_centered))
    hue = np.mod(hue, 360.0)

    metrics = {
        "pixel_count": pixel_count,
        "coverage_percent": coverage,
        "mean_rgb": mean_rgb,
        "mean_lab": mean_lab,
        "mean_luminance": float(np.mean(luminance[selected])),
        "mean_chroma": float(np.mean(chroma)),
        "mean_hue_degrees": float(np.mean(hue)),
        "delta_e_from_original": 0.0,
        "hue_rotation_degrees": 0.0,
    }

    if original_reference is not None:
        original_lab = np.array(original_reference["mean_lab"], dtype=np.float32)
        current_lab = np.array(mean_lab, dtype=np.float32)
        metrics["delta_e_from_original"] = float(
            np.linalg.norm(current_lab - original_lab)
        )

        original_hue = float(original_reference["mean_hue_degrees"])
        current_hue = float(metrics["mean_hue_degrees"])
        rotation = (current_hue - original_hue + 180.0) % 360.0 - 180.0
        metrics["hue_rotation_degrees"] = float(rotation)

    return metrics


@dataclass
class StageRecord:
    key: str
    label: str
    filename: str
    sha256: str
    materials: dict


class DebugStageRecorder:
    """
    Pure observer. It never modifies image arrays or semantic masks.
    """

    def __init__(
        self,
        project: Path,
        image_stem: str,
        scene,
        materials: Iterable[str] = DEFAULT_MATERIALS,
    ):
        self.project = Path(project)
        self.image_stem = image_stem
        self.scene = scene
        self.materials = list(materials)
        self.output_dir = self.project / "output" / "debug" / image_stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[StageRecord] = []
        self.original_metrics: dict[str, dict] = {}

    def save_stage(self, key: str, label: str, rgb: np.ndarray) -> None:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise TypeError(
                f"Debug stage {key} must be RGB uint8 HxWx3; got "
                f"{rgb.dtype} {rgb.shape}"
            )

        filename = f"{key}.png"
        save_rgb_png(rgb, self.output_dir / filename)

        material_results: dict[str, dict] = {}
        for material in self.materials:
            mask = _material_mask(self.scene, material, rgb.shape[:2])
            reference = self.original_metrics.get(material)
            result = compute_material_metrics(rgb, mask, reference)

            if key == "00_original":
                self.original_metrics[material] = result.copy()
                result["delta_e_from_original"] = 0.0
                result["hue_rotation_degrees"] = 0.0

            material_results[material] = result

        self.records.append(
            StageRecord(
                key=key,
                label=label,
                filename=filename,
                sha256=array_sha256(rgb),
                materials=material_results,
            )
        )

    def _rows(self) -> list[dict]:
        rows: list[dict] = []
        for record in self.records:
            for material, metrics in record.materials.items():
                rows.append({
                    "stage_key": record.key,
                    "stage_label": record.label,
                    "material": material,
                    "pixel_count": metrics["pixel_count"],
                    "coverage_percent": metrics["coverage_percent"],
                    "mean_r": metrics["mean_rgb"][0],
                    "mean_g": metrics["mean_rgb"][1],
                    "mean_b": metrics["mean_rgb"][2],
                    "mean_l": metrics["mean_lab"][0],
                    "mean_a": metrics["mean_lab"][1],
                    "mean_lab_b": metrics["mean_lab"][2],
                    "mean_luminance": metrics["mean_luminance"],
                    "mean_chroma": metrics["mean_chroma"],
                    "mean_hue_degrees": metrics["mean_hue_degrees"],
                    "delta_e_from_original": metrics["delta_e_from_original"],
                    "hue_rotation_degrees": metrics["hue_rotation_degrees"],
                    "stage_sha256": record.sha256,
                })
        return rows

    def _first_significant_stage(self, material: str) -> dict:
        for record in self.records[1:]:
            metrics = record.materials.get(material)
            if not metrics or metrics["pixel_count"] == 0:
                continue
            if (
                metrics["delta_e_from_original"] >= 4.0
                or abs(metrics["hue_rotation_degrees"]) >= 4.0
                or abs(
                    metrics["mean_luminance"]
                    - self.original_metrics[material]["mean_luminance"]
                ) >= 0.04
            ):
                return {
                    "stage_key": record.key,
                    "stage_label": record.label,
                    "delta_e": metrics["delta_e_from_original"],
                    "hue_rotation_degrees": metrics["hue_rotation_degrees"],
                    "luminance_change": (
                        metrics["mean_luminance"]
                        - self.original_metrics[material]["mean_luminance"]
                    ),
                }

        return {
            "stage_key": None,
            "stage_label": "No significant divergence detected",
            "delta_e": 0.0,
            "hue_rotation_degrees": 0.0,
            "luminance_change": 0.0,
        }

    def finalize(
        self,
        final_before_debug_hash: str,
        final_after_debug_hash: str,
    ) -> dict:
        rows = self._rows()
        csv_path = self.output_dir / "stage_metrics.csv"
        json_path = self.output_dir / "stage_metrics.json"
        html_path = self.output_dir / "debug_report.html"

        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        divergences = {
            material: self._first_significant_stage(material)
            for material in self.materials
        }

        verification = {
            "final_hash_before_report_generation": final_before_debug_hash,
            "final_hash_after_report_generation": final_after_debug_hash,
            "identical": final_before_debug_hash == final_after_debug_hash,
        }

        payload = {
            "image": self.image_stem,
            "stages": [
                {
                    "key": record.key,
                    "label": record.label,
                    "filename": record.filename,
                    "sha256": record.sha256,
                    "materials": record.materials,
                }
                for record in self.records
            ],
            "first_significant_divergence": divergences,
            "debug_non_mutation_verification": verification,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        html_path.write_text(self._render_html(payload, rows), encoding="utf-8")

        if not verification["identical"]:
            raise RuntimeError(
                "Debug report generation altered the final image array. "
                "This diagnostic build must be non-mutating."
            )

        return {
            "debug_directory": str(self.output_dir),
            "stage_metrics_csv": str(csv_path),
            "stage_metrics_json": str(json_path),
            "debug_report_html": str(html_path),
            "non_mutating": True,
            "first_significant_divergence": divergences,
        }

    def _render_html(self, payload: dict, rows: list[dict]) -> str:
        stage_cards = []
        for record in self.records:
            stage_cards.append(
                f"""
                <section class="stage">
                  <h3>{html.escape(record.label)}</h3>
                  <img src="{html.escape(record.filename)}"
                       alt="{html.escape(record.label)}">
                </section>
                """
            )

        table_rows = []
        for row in rows:
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row['stage_label']))}</td>"
                f"<td>{html.escape(str(row['material']))}</td>"
                f"<td>{int(row['pixel_count'])}</td>"
                f"<td>{row['mean_luminance']:.4f}</td>"
                f"<td>{row['mean_chroma']:.3f}</td>"
                f"<td>{row['mean_hue_degrees']:.2f}</td>"
                f"<td>{row['delta_e_from_original']:.3f}</td>"
                f"<td>{row['hue_rotation_degrees']:.2f}</td>"
                "</tr>"
            )

        findings = []
        for material, result in payload["first_significant_divergence"].items():
            findings.append(
                "<li>"
                f"<strong>{html.escape(material)}</strong>: "
                f"{html.escape(str(result['stage_label']))}"
                f" — ΔE {result['delta_e']:.2f}, "
                f"hue rotation {result['hue_rotation_degrees']:.2f}°, "
                f"luminance change {result['luminance_change']:+.4f}"
                "</li>"
            )

        verified = payload["debug_non_mutation_verification"]["identical"]
        verification_text = (
            "PASS — debug reporting did not alter the final image array."
            if verified
            else "FAIL — debug reporting changed the final image array."
        )

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(self.image_stem)} diagnostic report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       margin: 24px; background: #f4f4f4; color: #1d1d1f; }}
h1, h2, h3 {{ margin-top: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
         gap: 18px; }}
.stage {{ background: white; padding: 14px; border-radius: 10px; }}
.stage img {{ width: 100%; height: auto; display: block; }}
.panel {{ background: white; padding: 18px; margin: 18px 0;
          border-radius: 10px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
  text-align: left;
}}
.ok {{ color: #167c35; font-weight: 700; }}
</style>
</head>
<body>
<h1>MyEstatePics v3.1 Diagnostic Report</h1>
<p><strong>Image:</strong> {html.escape(self.image_stem)}</p>

<div class="panel">
<h2>First significant material change</h2>
<ul>{''.join(findings)}</ul>
<p class="ok">{html.escape(verification_text)}</p>
</div>

<div class="grid">
{''.join(stage_cards)}
</div>

<div class="panel">
<h2>Stage metrics</h2>
<table>
<thead>
<tr>
<th>Stage</th><th>Material</th><th>Pixels</th><th>Luminance</th>
<th>Chroma</th><th>Hue°</th><th>ΔE</th><th>Hue rotation°</th>
</tr>
</thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</div>
</body>
</html>
"""
