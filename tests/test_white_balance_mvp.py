import numpy as np

from engine.scene import build_scene
from engine.wb import (
    WhiteBalanceConfig,
    conservative_white_balance,
    global_pass1_wb,
    semantic_white_balance,
)


def _scene(h: int, w: int):
    masks = {name: np.zeros((h, w), np.float32) for name in ("ceiling", "wall", "floor")}
    masks["ceiling"][: h // 3] = 1.0
    masks["wall"][h // 3 : 2 * h // 3] = 1.0
    masks["floor"][2 * h // 3 :] = 1.0
    return build_scene(masks, (h, w))


def _mean_channel_spread(rgb: np.ndarray, mask: np.ndarray) -> float:
    means = rgb[mask].astype(np.float32).mean(axis=0)
    return float(means.max() - means.min())


def test_single_pass_reduces_neutral_ceiling_cast_without_overcorrection():
    h, w = 360, 480
    image = np.empty((h, w, 3), np.uint8)
    image[: h // 3] = [150, 128, 100]
    image[h // 3 : 2 * h // 3] = [110, 96, 82]
    image[2 * h // 3 :] = [88, 55, 34]
    scene = _scene(h, w)
    ceiling = scene.masks["ceiling"] > 0.5

    before = _mean_channel_spread(image, ceiling)
    out, log = conservative_white_balance(image, scene)
    after = _mean_channel_spread(out, ceiling)

    assert log["applied"] is True
    assert log["reference_used"] == "ceiling"
    assert after < before
    assert max(abs(g - 1.0) for g in log["gains"]) <= 0.120001


def test_already_balanced_image_is_noop():
    image = np.full((240, 320, 3), 128, np.uint8)
    out, log = conservative_white_balance(image, _scene(240, 320))
    assert log["applied"] is False
    assert log["reason"] == "within_noop_band"
    assert np.array_equal(out, image)


def test_global_fallback_is_more_tightly_bounded():
    image = np.full((200, 300, 3), [170, 120, 70], np.uint8)
    _, log = conservative_white_balance(
        image,
        None,
        config=WhiteBalanceConfig(low_confidence_limit=0.08),
    )
    assert log["confidence"] == "LOW"
    assert max(abs(g - 1.0) for g in log["gains"]) <= 0.080001


def test_compatibility_second_pass_does_not_compound_gains():
    h, w = 300, 420
    image = np.full((h, w, 3), [145, 120, 95], np.uint8)
    scene = _scene(h, w)
    first, first_log = global_pass1_wb(image, scene)
    second, second_log = semantic_white_balance(first, scene, {})

    assert first_log["pass1_applied"] is True
    assert second_log["reason"] == "single_pass_already_completed"
    assert np.array_equal(second, first)


def test_dtype_and_dimensions_are_preserved():
    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, size=(127, 193, 3), dtype=np.uint8)
    out, _ = conservative_white_balance(image, None)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
