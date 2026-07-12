
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha1
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image).copy()


def resize_for_analysis(rgb: np.ndarray, max_side: int = 2048) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return rgb, (w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA), (nw, nh)


def clear_generated_outputs(project: Path) -> None:
    for relative in ["output/final", "output/review", "output/logs", "output/comparisons", "output/reports", "output/debug"]:
        folder = project / relative
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def cache_key(path: Path, input_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = path.name
    stat = path.stat()
    raw = f"{relative}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def remove_stale_cache(project: Path, valid_keys: set[str]) -> None:
    label_dir = project / "output/cache/labels"
    mask_dir = project / "output/cache/masks"
    for folder in [label_dir, mask_dir]:
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if not item.is_file():
                continue
            if not any(item.name.startswith(key + "_") or item.stem == key for key in valid_keys):
                item.unlink()


@dataclass
class ImageAnalysis:
    p35: float
    p50: float
    p95: float
    shadow_lift: float
    midtone_lift: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_image(rgb: np.ndarray) -> ImageAnalysis:
    f = rgb.astype(np.float32) / 255.0
    y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    p35, p50, p95 = [float(x) for x in np.percentile(y, [35, 50, 95])]
    return ImageAnalysis(
        p35=p35,
        p50=p50,
        p95=p95,
        shadow_lift=float(np.clip((0.44 - p35) * 0.85, 0.0, 0.35)),
        midtone_lift=float(np.clip((0.54 - p50) * 0.70, 0.0, 0.28)),
    )


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8))


def load_mask_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return raw.astype(np.float32) / 255.0


def make_comparison(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    max_side = max(before.shape[:2])
    scale = min(1.0, 1800.0 / max_side)
    if scale < 1.0:
        size = (round(before.shape[1] * scale), round(before.shape[0] * scale))
        before = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    divider = np.full((before.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.hstack([before, divider, after])



def make_scored_comparison(
    before: np.ndarray,
    after: np.ndarray,
    filename: str,
    checklist: dict,
) -> np.ndarray:
    comparison = make_comparison(before, after)
    banner_height = 92
    canvas = np.zeros(
        (comparison.shape[0] + banner_height, comparison.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[banner_height:] = comparison
    canvas[:banner_height] = 24

    decision = checklist.get("decision", "REVIEW")
    overall = int(checklist.get("overall_score", 0))

    cv2.putText(
        canvas,
        f"{filename}  |  {decision}  |  Score {overall}/100",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    x = 20
    y = 69
    for check in checklist.get("checks", []):
        text = f"{check['name']}: {check['score']}"
        color = (80, 220, 110) if check["status"] == "PASS" else (255, 180, 70)
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        x += max(180, len(text) * 10)

    return canvas


def make_contact_sheet(
    comparisons: list[tuple[str, np.ndarray]],
    columns: int = 2,
    tile_width: int = 1400,
) -> np.ndarray | None:
    if not comparisons:
        return None

    tiles: list[np.ndarray] = []
    for _, image in comparisons:
        scale = tile_width / image.shape[1]
        tile_height = max(1, round(image.shape[0] * scale))
        tile = cv2.resize(
            image,
            (tile_width, tile_height),
            interpolation=cv2.INTER_AREA,
        )
        tiles.append(tile)

    max_tile_height = max(tile.shape[0] for tile in tiles)
    normalized = []
    for tile in tiles:
        if tile.shape[0] < max_tile_height:
            pad = np.zeros(
                (max_tile_height - tile.shape[0], tile.shape[1], 3),
                dtype=np.uint8,
            )
            tile = np.vstack([tile, pad])
        normalized.append(tile)

    rows = []
    for index in range(0, len(normalized), columns):
        row_tiles = normalized[index:index + columns]
        while len(row_tiles) < columns:
            row_tiles.append(
                np.zeros((max_tile_height, tile_width, 3), dtype=np.uint8)
            )
        rows.append(np.hstack(row_tiles))

    return np.vstack(rows)
