from __future__ import annotations

"""High-precision ingestion for Lightroom linear HDR DNG experiments.

The production JPEG path does not import this module. Lightroom HDR merges use
JPEG XL-compressed, three-channel LinearRaw planes that current LibRaw builds
may reject. tifffile plus imagecodecs exposes that plane without rendering it
to eight bits.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - exercised only without extras
    tifffile = None
    _TIFF_IMPORT_ERROR = exc
else:
    _TIFF_IMPORT_ERROR = None


_D50_TO_D65 = np.array(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float32,
)
_XYZ_D65_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)


@dataclass
class DNGImage:
    """Decoded DNG data kept in scene-linear and extended sRGB precision."""

    camera_linear: np.ndarray
    xyz_d50: np.ndarray
    linear_srgb: np.ndarray
    encoded_srgb: np.ndarray
    metadata: dict[str, Any]


def _rational_array(value: Any) -> np.ndarray:
    flat = list(value)
    return np.asarray(
        [float(flat[i]) / max(float(flat[i + 1]), 1e-12) for i in range(0, len(flat), 2)],
        dtype=np.float32,
    )


def _xmp_value(xmp: bytes | str, name: str) -> str | None:
    text = xmp.decode("utf-8", errors="replace") if isinstance(xmp, bytes) else str(xmp)
    match = re.search(rf'(?:crs|aux|xmp):{re.escape(name)}="([^"]*)"', text)
    return match.group(1) if match else None


def _apply_orientation(array: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 1:
        return array
    if orientation == 2:
        return np.fliplr(array)
    if orientation == 3:
        return np.rot90(array, 2)
    if orientation == 4:
        return np.flipud(array)
    if orientation == 5:
        return np.transpose(array, (1, 0, 2))
    if orientation == 6:
        return np.rot90(array, 3)
    if orientation == 7:
        return np.flipud(np.transpose(array, (1, 0, 2)))
    if orientation == 8:
        return np.rot90(array, 1)
    return array


def _interpolated_forward_matrix(root, temperature: float | None) -> tuple[np.ndarray, str]:
    fm1 = _rational_array(root.tags["ForwardMatrix1"].value).reshape(3, 3)
    fm2 = _rational_array(root.tags["ForwardMatrix2"].value).reshape(3, 3)
    if temperature is None or not np.isfinite(temperature):
        return fm2, "ForwardMatrix2"
    # Calibration illuminants are Standard Light A and D65 in these files.
    t1, t2 = 2856.0, 6504.0
    weight2 = float(np.clip((1.0 / temperature - 1.0 / t1) / (1.0 / t2 - 1.0 / t1), 0.0, 1.0))
    return fm1 * (1.0 - weight2) + fm2 * weight2, f"ForwardMatrix1/2 reciprocal-temperature interpolation ({weight2:.4f})"


def _linear_to_extended_srgb(linear: np.ndarray) -> np.ndarray:
    positive = np.maximum(linear, 0.0)
    return np.where(
        positive <= 0.0031308,
        12.92 * positive,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _percentiles(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    # Metadata statistics must not dominate production runtime or allocate a
    # second full-resolution RGB plane. Deterministic striding preserves the
    # distribution while bounding percentile work to one million samples.
    if values.size > 1_000_000:
        values = values[:: int(np.ceil(values.size / 1_000_000))]
    return {
        f"p{label}": float(value)
        for label, value in zip(
            ("0_01", "0_1", "1", "50", "99", "99_9", "99_99"),
            np.percentile(values, (0.01, 0.1, 1, 50, 99, 99.9, 99.99)),
        )
    }


def read_lightroom_hdr_dng(path: str | Path) -> DNGImage:
    """Decode the full Lightroom HDR DNG LinearRaw plane as float32.

    No sharpening, denoising, local adjustment, or develop-settings tone map
    is applied. DNG AsShotNeutral, ForwardMatrix, BaselineExposure, crop, and
    orientation metadata are honored. Values above display white are retained.
    """

    if tifffile is None:
        raise RuntimeError(
            "DNG support requires tifffile and imagecodecs (JPEG XL support)"
        ) from _TIFF_IMPORT_ERROR

    source = Path(path)
    with tifffile.TiffFile(source) as tf:
        if not tf.is_dng:
            raise ValueError(f"Not a DNG file: {source}")
        root = tf.pages[0]
        candidates = [series for series in tf.series if len(series.shape) == 3]
        if not candidates:
            raise ValueError(f"No RGB image plane found in {source}")
        series = max(candidates, key=lambda item: int(np.prod(item.shape[:2])))
        page = series.pages[0]
        sample_formats = (
            tuple(page.sampleformat)
            if isinstance(page.sampleformat, (tuple, list))
            else (page.sampleformat,) * int(page.samplesperpixel)
        )
        if page.samplesperpixel != 3 or sample_formats[0].name != "IEEEFP":
            raise ValueError("Expected a three-channel floating-point Lightroom LinearRaw plane")

        raw_storage = series.asarray()
        raw = raw_storage.astype(np.float32)
        orientation = int(root.tags.get("Orientation", 1).value)
        raw = _apply_orientation(raw, orientation)
        black = _rational_array(page.tags["BlackLevel"].value)
        white = np.asarray(page.tags["WhiteLevel"].value, dtype=np.float32)
        normalized = (raw - black.reshape(1, 1, 3)) / np.maximum(
            white.reshape(1, 1, 3) - black.reshape(1, 1, 3), 1e-8
        )

        neutral = _rational_array(root.tags["AsShotNeutral"].value)
        xmp = root.tags.get("XMP")
        xmp_bytes = xmp.value if xmp is not None else b""
        temperature_text = _xmp_value(xmp_bytes, "Temperature")
        temperature = float(temperature_text) if temperature_text else None
        forward, matrix_source = _interpolated_forward_matrix(root, temperature)
        baseline_exposure = float(_rational_array(root.tags["BaselineExposure"].value)[0])

        # Preserve signed scene-linear samples in the immutable master. Small
        # negatives are legitimate around the DNG black estimate and are only
        # clamped at a display-rendering boundary.
        camera_balanced = normalized / np.maximum(
            neutral.reshape(1, 1, 3), 1e-8
        )
        # BaselineExposure is DNG metadata normalization, not a creative edit.
        # Apply it at the color-management boundary so every downstream
        # scene-linear representation and measurement uses the same domain.
        xyz_d50 = (camera_balanced @ forward.T) * float(2.0**baseline_exposure)
        xyz_d65 = xyz_d50 @ _D50_TO_D65.T
        linear_srgb = xyz_d65 @ _XYZ_D65_TO_SRGB.T
        encoded_srgb = _linear_to_extended_srgb(linear_srgb)

        y = (
            0.2126 * linear_srgb[..., 0]
            + 0.7152 * linear_srgb[..., 1]
            + 0.0722 * linear_srgb[..., 2]
        )
        positive = y[y > 1e-8]
        low = float(np.percentile(positive, 0.1)) if positive.size else 0.0
        high = float(np.percentile(positive, 99.9)) if positive.size else 0.0
        dynamic_stops = float(np.log2(high / low)) if low > 0 and high > low else 0.0

        bits_per_sample = (
            list(page.bitspersample)
            if isinstance(page.bitspersample, (tuple, list))
            else [page.bitspersample] * int(page.samplesperpixel)
        )
        metadata = {
            "filename": source.name,
            "decoder": "tifffile + imagecodecs JPEG XL",
            "dng_type": "linear HDR DNG" if int(page.photometric) == 34892 else "non-mosaic RGB DNG",
            "is_merged_hdr": _xmp_value(xmp_bytes, "IsMergedHDR") == "True",
            "storage_dtype": str(raw_storage.dtype),
            "working_dtype": str(linear_srgb.dtype),
            "bits_per_sample": [int(x) for x in bits_per_sample],
            "sample_format": [item.name for item in sample_formats],
            "compression": int(page.compression),
            "dimensions": [int(encoded_srgb.shape[1]), int(encoded_srgb.shape[0])],
            "orientation": orientation,
            "photometric": int(page.photometric),
            "color_space": "extended linear sRGB working space; DNG camera color via ForwardMatrix to XYZ D50/D65",
            "camera_make": str(root.tags.get("Make").value),
            "camera_model": str(root.tags.get("Model").value),
            "software": str(root.tags.get("Software").value),
            "capture_time": str(root.tags["ExifTag"].value.get("DateTimeOriginal")),
            "as_shot_neutral": [float(x) for x in neutral],
            "white_balance_temperature_xmp": temperature,
            "white_balance_tint_xmp": _xmp_value(xmp_bytes, "Tint"),
            "forward_matrix_source": matrix_source,
            "baseline_exposure_ev": baseline_exposure,
            "black_level": [float(x) for x in black],
            "white_level": [float(x) for x in white],
            "raw_normalized_stats": _percentiles(normalized),
            "linear_luminance_stats": _percentiles(y),
            "estimated_dynamic_range_stops_p0_1_to_p99_9": dynamic_stops,
            "encoded_values_above_display_white_percent": float(np.mean(np.max(encoded_srgb, axis=2) > 1.0) * 100.0),
            "automatic_tone_mapping": False,
            "automatic_sharpening": False,
            "automatic_denoising": False,
        }
        return DNGImage(
            camera_linear=camera_balanced.astype(np.float32, copy=False),
            xyz_d50=xyz_d50.astype(np.float32, copy=False),
            linear_srgb=linear_srgb.astype(np.float32, copy=False),
            encoded_srgb=encoded_srgb.astype(np.float32, copy=False),
            metadata=metadata,
        )
