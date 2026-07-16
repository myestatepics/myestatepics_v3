from __future__ import annotations

"""Local, scene-linear window recovery with deterministic restrained sky."""

import hashlib

import cv2
import numpy as np

from .scene import Scene


def _smooth01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(rgb.astype(np.float32), 0.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055).astype(np.float32)


def _luma(linear: np.ndarray) -> np.ndarray:
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


def _seed_value(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "little")


def generate_mls_sky(shape: tuple[int, int], seed: str) -> np.ndarray:
    """Generate one light-blue, low-contrast cloud field for the full frame."""
    h, w = shape
    rng = np.random.default_rng(_seed_value(seed))
    small_h, small_w = max(8, h // 96), max(8, w // 96)
    noise = rng.random((small_h, small_w), dtype=np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.25)
    cloud = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    cloud = (cloud - float(cloud.min())) / max(float(np.ptp(cloud)), 1e-6)
    cloud = _smooth01((cloud - 0.38) / 0.38) * 0.32
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    blue = np.empty((h, w, 3), np.float32)
    blue[..., 0] = 0.67 + 0.10 * yy
    blue[..., 1] = 0.80 + 0.08 * yy
    blue[..., 2] = 0.91 + 0.05 * yy
    return np.clip(blue * (1.0 - cloud[..., None]) + cloud[..., None] * 0.97, 0.0, 1.0)


def _component_upper_prior(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.35).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    prior = np.zeros(mask.shape, np.float32)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 64:
            continue
        ramp = np.linspace(1.0, 0.0, max(h, 1), dtype=np.float32)[:, None]
        prior[y : y + h, x : x + w] = np.maximum(
            prior[y : y + h, x : x + w], np.broadcast_to(ramp, (h, w))
        )
    return prior


def recover_windows(
    scene_linear_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    scene: Scene,
    *,
    seed: str,
) -> tuple[np.ndarray, dict]:
    """Recover real exterior first and synthesize only unrecoverable sky pixels."""
    shape = rendered_rgb.shape[:2]
    zero = np.zeros(shape, np.float32)
    window = scene.masks.get("window", zero)
    mirror = scene.masks.get("mirror", zero)
    curtain = scene.masks.get("curtain", zero)
    eligible = np.clip(window * (1.0 - mirror) * (1.0 - curtain), 0.0, 1.0)
    if np.count_nonzero(eligible > 0.35) < 128:
        return rendered_rgb.copy(), {
            "status": "no_windows", "window_percent": float(np.mean(window > 0.5) * 100.0),
            "recoverable_percent": 0.0, "unrecoverable_sky_percent": 0.0,
            "blue_spill_percent": 0.0, "fallback_route": "no_window_noop",
        }

    source = np.maximum(scene_linear_rgb.astype(np.float32), 0.0)
    source_y = _luma(source)
    valid = eligible > 0.5
    values = source_y[valid & np.isfinite(source_y)]
    reference = float(np.percentile(values, 68)) if values.size else 1.0
    gain = float(np.clip(0.58 / max(reference, 1e-6), 0.02, 16.0))
    exposed_y = source_y * gain
    recovered_y = exposed_y / (exposed_y + 0.72)
    scale = recovered_y / np.maximum(source_y, 1e-8)
    recovered_linear = source * scale[..., None]
    recovered_scale = 1.0 / np.maximum(np.max(recovered_linear, axis=2), 1.0)
    recovered = np.clip(_linear_to_srgb(recovered_linear * recovered_scale[..., None]), 0.0, 1.0)

    rendered_linear = _srgb_to_linear(rendered_rgb)
    rendered_y = _luma(rendered_linear)
    gx = cv2.Sobel(rendered_y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(rendered_y, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    frame_guard = np.maximum(_smooth01((0.48 - rendered_y) / 0.28), _smooth01((edge - 0.045) / 0.12))
    highlight_gate = _smooth01((rendered_y - 0.48) / 0.32)
    recovery_alpha = eligible * highlight_gate * (1.0 - 0.92 * frame_guard)
    recovery_alpha = np.minimum(recovery_alpha, eligible)

    recovered_gray = cv2.cvtColor(np.clip(recovered, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    local_mean = cv2.boxFilter(recovered_gray, cv2.CV_32F, (21, 21), normalize=True)
    local_sq = cv2.boxFilter(recovered_gray * recovered_gray, cv2.CV_32F, (21, 21), normalize=True)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    upper = _component_upper_prior(eligible)
    low_detail = 1.0 - _smooth01((local_std - 0.006) / 0.018)
    bright = _smooth01((np.maximum(recovered_gray, rendered_y) - 0.62) / 0.22)
    sky_need = eligible * upper * low_detail * bright * (1.0 - frame_guard)
    sky_need = np.where(sky_need > 0.35, sky_need, 0.0).astype(np.float32)
    sky_alpha = cv2.GaussianBlur(sky_need, (0, 0), 1.2)
    sky_alpha = np.minimum(sky_alpha, eligible * (1.0 - frame_guard))

    real_first = rendered_rgb * (1.0 - recovery_alpha[..., None]) + recovered * recovery_alpha[..., None]
    sky = generate_mls_sky(shape, seed)
    output = real_first * (1.0 - sky_alpha[..., None]) + sky * sky_alpha[..., None]
    outside = eligible < 1e-5
    output[outside] = rendered_rgb[outside]

    recoverable = (recovery_alpha > 0.15) & (sky_alpha < 0.15)
    sky_pixels = sky_alpha > 0.15
    status = "partially_recoverable" if np.any(sky_pixels) and np.any(recoverable) else (
        "unrecoverable_sky_replaced" if np.any(sky_pixels) else "recoverable"
    )
    return np.clip(output, 0.0, 1.0).astype(np.float32), {
        "status": status,
        "window_percent": float(np.mean(window > 0.5) * 100.0),
        "recoverable_percent": float(np.mean(recoverable) * 100.0),
        "unrecoverable_sky_percent": float(np.mean(sky_pixels) * 100.0),
        "frame_guard_percent": float(np.mean((frame_guard > 0.5) & valid) * 100.0),
        "curtain_excluded_percent": float(np.mean((curtain > 0.5) & (window > 0.5)) * 100.0),
        "mirror_excluded_percent": float(np.mean((mirror > 0.5) & (window > 0.5)) * 100.0),
        "blue_spill_percent": float(np.mean(np.any(np.abs(output - rendered_rgb) > 1e-6, axis=2) & outside) * 100.0),
        "window_exposure_gain": gain,
        "fallback_route": "scene_linear_recovery_then_deterministic_sky",
    }
