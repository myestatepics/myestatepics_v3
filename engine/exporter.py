
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
    expected_shape: tuple[int, int] | None = None,
) -> dict:
    if expected_shape is not None and rgb.shape[:2] != expected_shape:
        raise ValueError(
            f"Resolution mismatch: output {rgb.shape[:2]} != input {expected_shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb, mode="RGB")
    chosen_quality = 83
    chosen_bytes = b""

    for quality in range(start_quality, 81, -2):
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=True,
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
        "width": int(image.width),
        "height": int(image.height),
        "oversized": bool(oversized),
        "exif_stripped": True,
        "resolution_preserved": True,
    }
