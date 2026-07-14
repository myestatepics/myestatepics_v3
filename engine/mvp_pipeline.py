from __future__ import annotations

from typing import Any

import numpy as np

from .exposure_fusion import exposure_fusion
from .materials2 import restore_protected_chroma
from .profiles import detect_room_profile
from .tone_classify import classify_tones
from .utils import analyze_image
from .wb import conservative_white_balance


def process_mvp_core(
    original: np.ndarray,
    scene,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Run the model-independent MVP image core.

    Segmentation/refinement happen before this function. The core uses one
    conservative white-balance pass, bounded global exposure fusion, and
    semantic masks only to protect photographed material chroma.
    """
    tone_settings = settings.get("targets", {})
    wb, wb_log = conservative_white_balance(original, scene)

    profile_name, profile_targets, profile_dynamics = detect_room_profile(scene, wb)
    tone_settings = {**tone_settings, **profile_targets}
    tones, tones_log = classify_tones(wb, scene, tone_settings)
    analysis = analyze_image(wb).to_dict()
    analysis["furnishing_protection"] = float(
        tone_settings.get("furnishing_factor", 1.0)
    )
    analysis["material_inherit_factor"] = float(
        settings.get("phase3", {}).get("material_inherit_factor", 1.0)
    )
    analysis.update(profile_dynamics)
    analysis["room_profile"] = profile_name

    exposed, exposure_log = exposure_fusion(wb)
    protected, materials_log = restore_protected_chroma(wb, exposed, scene)

    return {
        "wb": wb,
        "wb_log": wb_log,
        "exposed": exposed,
        "exposure_log": exposure_log,
        "protected": protected,
        "materials_log": materials_log,
        "tones": tones,
        "tones_log": tones_log,
        "analysis": analysis,
        "profile_name": profile_name,
        "window_log": {
            "status": "skipped_mvp_phase3",
            "reason": "window treatment intentionally disabled in MVP core",
        },
    }
