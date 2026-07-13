
from __future__ import annotations

import numpy as np

from .scene import Scene


# Each profile is an exposure MODEL: target overrides consumed by
# tone_classify (cfg keys) plus dynamics consumed by tonal
# (ratio caps / ambient cap). "Correct the light, never the materials"
# holds in every profile; profiles only decide how much light a room
# of that kind should receive and how gently to deliver it.
PROFILES: dict[str, dict] = {
    "kitchen": {
        # Kitchens sell on materials: cabinets, counters, floor. Bright,
        # clean structure; slightly restrained ambient so dark cabinetry
        # keeps its depth.
        "targets": {"ceiling_target": 0.80, "wall_light_target": 0.63},
        "dynamics": {"ambient_cap_ratio": 1.42, "ceiling_ratio_cap": 1.90},
    },
    "bathroom": {
        # Small, tiled, mirror-heavy. Airy and bright; segmentation is
        # weaker here so a touch more ambient compensates for sparse
        # structure coverage.
        "targets": {"ceiling_target": 0.82, "wall_light_target": 0.66},
        "dynamics": {"ambient_cap_ratio": 1.55, "ceiling_ratio_cap": 2.00},
    },
    "bedroom": {
        # Warm, restful; MLS-bright but never clinical.
        "targets": {"ceiling_target": 0.78, "wall_light_target": 0.61},
        "dynamics": {"ambient_cap_ratio": 1.48, "ceiling_ratio_cap": 1.85},
    },
    "living": {
        "targets": {"ceiling_target": 0.80, "wall_light_target": 0.62},
        "dynamics": {"ambient_cap_ratio": 1.50, "ceiling_ratio_cap": 1.90},
    },
    "dark_room": {
        # Deliberately dark decor (z-66 class): moody stays moody.
        # Gentle everything; black paint keeps reading black.
        "targets": {
            "ceiling_target": 0.70,
            "wall_dark_lift": 0.08,
            "wall_dark_max_lift": 0.10,
        },
        "dynamics": {"ambient_cap_ratio": 1.30, "ceiling_ratio_cap": 1.70},
    },
    "low_light": {
        # Basement / windowless / "black paper over every window":
        # the room must still come out professionally exposed even with
        # zero window light, so this profile gets the deepest reach in the
        # engine -- higher ratio ceilings and the widest ambient cap --
        # still delivered through the same edge-aware, chroma-damped path.
        "targets": {"ceiling_target": 0.78, "wall_light_target": 0.62,
                    "wall_medium_max_lift": 0.20,
                    # In a windowless/dim room every surface reads darker than
                    # its true paint. Shift classification thresholds down so
                    # light-gray paint at L 0.24 is treated as paint under dim
                    # light (medium), not as a dark accent wall to preserve.
                    "wall_light_threshold": 0.38,
                    "wall_medium_threshold": 0.18},
        "dynamics": {"ambient_cap_ratio": 1.70, "ceiling_ratio_cap": 2.20,
                     "wall_ratio_caps": {1: 1.35, 2: 1.80, 3: 2.05}},
    },
    "generic": {
        "targets": {},
        "dynamics": {"ambient_cap_ratio": 1.50, "ceiling_ratio_cap": 1.90},
    },
}


def detect_room_profile(scene: Scene, rgb: np.ndarray) -> tuple[str, dict, dict]:
    """
    Rule-based room classification from semantic composition + light level.
    Returns (profile_name, target_overrides, dynamics). Order matters:
    light-level profiles outrank furniture profiles because exposure need
    dominates decor.
    """
    cov = scene.coverage
    f = rgb.astype(np.float32) / 255.0
    y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]

    structure = scene.masks.get("structure")
    structure_median = (
        float(np.median(y[structure > 0.5]))
        if structure is not None and np.any(structure > 0.5)
        else float(np.median(y))
    )

    name = "generic"
    if structure_median < 0.15:
        # Distinguish dark DECOR from dark LIGHTING: dark paint has low
        # reflectance but the ceiling is usually still light-painted.
        ceiling = scene.masks.get("ceiling")
        ceiling_median = (
            float(np.median(y[ceiling > 0.5]))
            if ceiling is not None and np.any(ceiling > 0.5)
            else structure_median
        )
        wall = scene.masks.get("wall")
        wall_median = (
            float(np.median(y[wall > 0.5]))
            if wall is not None and np.any(wall > 0.5)
            else structure_median
        )
        name = "dark_room" if wall_median < 0.5 * max(ceiling_median, 1e-3) else "low_light"
    elif cov.get("window", 0.0) < 0.3 and structure_median < 0.30:
        name = "low_light"
    elif cov.get("mirror", 0.0) >= 2.0 and cov.get("floor", 0.0) < 5.0:
        name = "bathroom"
    elif cov.get("cabinet", 0.0) >= 10.0:
        name = "kitchen"
    elif cov.get("bed", 0.0) >= 4.0:
        name = "bedroom"
    elif cov.get("sofa", 0.0) >= 2.0:
        name = "living"

    profile = PROFILES[name]
    return name, dict(profile["targets"]), dict(profile["dynamics"])
