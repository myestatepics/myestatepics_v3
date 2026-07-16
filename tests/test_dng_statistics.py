import numpy as np

from engine.dng import _percentiles


def test_dng_metadata_percentiles_are_deterministic_and_bounded_memory():
    values = np.linspace(-0.25, 8.0, 2_100_003, dtype=np.float32)
    first = _percentiles(values)
    second = _percentiles(values)
    assert first == second
    assert first["p0_01"] < first["p50"] < first["p99_99"]
