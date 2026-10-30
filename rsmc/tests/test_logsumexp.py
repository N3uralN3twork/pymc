"""Check the completed exercise through its native Python entry point."""

import inspect

import numpy as np
import pymc_rust
import pytest

from scipy.special import logsumexp


@pytest.mark.parametrize(
    "values",
    [
        [-3.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1000.0, 1000.0],
        [-1000.0, -1000.0],
        [-1000.0, 0.0, 1000.0],
    ],
)
def test_native_logsumexp_matches_scipy(values: list[float]) -> None:
    inputs = np.array(values, dtype=np.float64)
    inputs.flags.writeable = False
    before = inputs.copy()
    assert inspect.isbuiltin(pymc_rust.logsumexp)
    result = pymc_rust.logsumexp(inputs)
    assert isinstance(result, float)
    assert result == pytest.approx(float(logsumexp(inputs)), rel=1e-12, abs=1e-12)
    np.testing.assert_array_equal(inputs, before)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-np.inf]])
def test_logsumexp_rejects_inputs_outside_the_exercise(values: list[float]) -> None:
    with pytest.raises(ValueError):
        pymc_rust.logsumexp(np.array(values, dtype=np.float64))


def test_logsumexp_layout_and_dtype() -> None:
    with pytest.raises(ValueError, match="C-contiguous"):
        pymc_rust.logsumexp(np.zeros(8, dtype=np.float64)[::2])
    with pytest.raises(TypeError):
        pymc_rust.logsumexp(np.zeros(4, dtype=np.float32))
    with pytest.raises(TypeError):
        pymc_rust.logsumexp(np.zeros((2, 2), dtype=np.float64))
