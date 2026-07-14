
from __future__ import annotations

import numpy as np


def natural_finish(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the protected exposure result without stacked enhancement."""
    return rgb.copy(), {
        "local_contrast": 0.0,
        "black_anchor": 0.0,
        "sharpen": 0.0,
        "mode": "no_additional_finish_phase_a",
    }
