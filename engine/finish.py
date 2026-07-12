
from __future__ import annotations

import cv2
import numpy as np


def natural_finish(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[..., 0] / 255.0
    base = cv2.GaussianBlur(l, (0, 0), 9.0)
    detail = l - base
    mid = np.clip(1.0 - np.abs(l - 0.50) / 0.46, 0.0, 1.0)
    l2 = l + detail * (0.10 * mid)
    black_weight = np.clip((0.24 - l2) / 0.24, 0.0, 1.0)
    l2 -= black_weight * 0.006
    lab[..., 0] = np.clip(l2 * 255.0, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(out, (0, 0), 0.8)
    out = cv2.addWeighted(out, 1.06, blur, -0.06, 0)
    return np.clip(out, 0, 255).astype(np.uint8), {
        "local_contrast": 0.10,
        "black_anchor": 0.006,
        "sharpen": 0.06,
    }
