import numpy as np

from engine.tonal import (
    _adaptive_ratio_cap,
    global_safe_exposure,
)


def test_adaptive_cap_gives_medium_surfaces_more_reach():
    near_black = _adaptive_ratio_cap(0.06, 1.30, surface="wall")
    medium = _adaptive_ratio_cap(0.30, 1.30, surface="wall")
    assert medium > near_black
    assert near_black <= 1.35
    assert medium >= 2.0


def test_global_safe_does_not_darkens_already_good_image():
    image = np.full((160, 220, 3), 150, np.uint8)
    out, _ = global_safe_exposure(
        image,
        {
            "shadow_lift": 0.0,
            "midtone_lift": 0.0,
            "room_profile": "generic",
        },
    )
    assert out.mean() >= image.mean() - 0.5


def test_global_safe_convergence_brightens_underexposed_room():
    image = np.full((160, 220, 3), 78, np.uint8)
    out, log = global_safe_exposure(
        image,
        {
            "shadow_lift": 0.22,
            "midtone_lift": 0.16,
            "room_profile": "low_light",
        },
    )
    assert out.mean() > image.mean() * 1.25
    assert "convergence" in log


def test_global_safe_preserves_geometry():
    image = np.full((123, 197, 3), 90, np.uint8)
    out, _ = global_safe_exposure(
        image,
        {
            "shadow_lift": 0.18,
            "midtone_lift": 0.12,
            "room_profile": "generic",
        },
    )
    assert out.shape == image.shape
