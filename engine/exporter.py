
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def save_jpeg(
    rgb: np.ndarray,
    output_path: Path,
    max_mb: float,
    start_quality: int = 95,
    min_quality: int = 82,
    expected_shape: tuple[int, int] | None = None,
    progressive: bool = True,
) -> dict:
    if expected_shape is not None and rgb.shape[:2] != expected_shape:
        raise ValueError(
            f"Resolution mismatch: output {rgb.shape[:2]} != input {expected_shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    if not 1 <= min_quality <= start_quality <= 95:
        raise ValueError("JPEG quality must satisfy 1 <= min_quality <= start_quality <= 95")

    chosen_quality = min_quality
    chosen_bytes = b""

    qualities = list(range(start_quality, min_quality - 1, -2))
    if qualities[-1] != min_quality:
        qualities.append(min_quality)
    for quality in qualities:
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
            optimize=True,
            progressive=progressive,
            dpi=(300, 300),
        )
        chosen_quality = quality
        chosen_bytes = buffer.getvalue()
        if len(chosen_bytes) <= max_mb * 1024 * 1024:
            break

    oversized = len(chosen_bytes) > max_mb * 1024 * 1024
    output_path.write_bytes(chosen_bytes)
    return {
        "quality": int(chosen_quality),
        "size_mb": float(len(chosen_bytes) / (1024 * 1024)),
        "size_bytes": int(len(chosen_bytes)),
        "width": int(image.width),
        "height": int(image.height),
        "oversized": bool(oversized),
        "exif_stripped": True,
        "resolution_preserved": True,
        "progressive": bool(progressive),
        "optimized": True,
    }
