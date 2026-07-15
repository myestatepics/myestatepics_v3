from pathlib import Path

import numpy as np
from PIL import Image

from engine.exporter import save_jpeg


def test_mls_jpeg_is_progressive_optimized_and_preserves_dimensions(tmp_path: Path):
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(900, 1400, 3), dtype=np.uint8)
    path = tmp_path / "delivery.jpg"

    log = save_jpeg(rgb, path, max_mb=0.35, start_quality=92, min_quality=82, expected_shape=(900, 1400))

    with Image.open(path) as image:
        assert image.size == (1400, 900)
        assert bool(image.info.get("progressive"))
    assert 82 <= log["quality"] <= 92
    assert log["size_bytes"] == path.stat().st_size
    assert log["progressive"] is True
    assert log["optimized"] is True


def test_mls_jpeg_reduces_quality_but_never_below_floor(tmp_path: Path):
    rng = np.random.default_rng(11)
    rgb = rng.integers(0, 256, size=(1000, 1500, 3), dtype=np.uint8)
    path = tmp_path / "bounded.jpg"

    log = save_jpeg(rgb, path, max_mb=0.01, start_quality=92, min_quality=82)

    assert log["quality"] == 82
    assert log["oversized"] is True
