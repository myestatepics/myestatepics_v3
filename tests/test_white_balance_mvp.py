import cv2
import numpy as np

from engine.scene import build_scene
from engine.wb import (
    WhiteBalanceConfig,
    conservative_white_balance,
    gentle_wall_chroma_consistency,
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


def test_mixed_lit_ceiling_is_rejected_as_global_reference():
    h, w = 360, 480
    image = np.full((h, w, 3), 128, np.uint8)
    image[: h // 3, : w // 2] = [170, 130, 90]
    image[: h // 3, w // 2 :] = [100, 130, 170]
    _, log = conservative_white_balance(image, _scene(h, w))
    assert log["reference_used"] != "ceiling"


def test_wall_consistency_is_chroma_only_and_feathered():
    h, w = 240, 360
    scene = _scene(h, w)
    image = np.full((h, w, 3), [130, 130, 130], np.uint8)
    image[h // 3 : 2 * h // 3, : w // 2] = [138, 128, 118]
    image[h // 3 : 2 * h // 3, w // 2 :] = [118, 128, 138]
    before_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    out, log = gentle_wall_chroma_consistency(image, scene)
    after_lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    wall = scene.masks["wall"] > 0.55
    assert log["applied"] is True
    assert np.percentile(np.abs(after_lab[..., 0][wall] - before_lab[..., 0][wall]), 95) <= 1.0
    before_spread = np.std(before_lab[..., 1:3][wall], axis=0).mean()
    after_spread = np.std(after_lab[..., 1:3][wall], axis=0).mean()
    assert after_spread < before_spread
