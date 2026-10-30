"""Check the second exercise through the native Python entry point."""

import inspect

import numpy as np
import pymc_rust
import pytest

from scipy.special import logsumexp


@pytest.mark.parametrize(
    ("values", "old_beta", "beta"),
    [
        ([0.0, np.log(2.0), np.log(3.0), np.log(4.0)], 0.0, 1.0),
        ([1000.0, 999.0, 998.0, 997.0], 0.25, 0.75),
        ([-1000.0, -999.0, -998.0, -997.0], 0.25, 1.5),
        ([-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0], 0.35, 2.0),
        ([0.0, 1.0, 2.0, 3.0], 0.5, 0.5),
        ([-3.0], 0.0, 0.5),
    ],
)
def test_native_weights_at_beta(values: list[float], old_beta: float, beta: float) -> None:
    inputs = np.array(values, dtype=np.float64)
    before = inputs.copy()
    inputs.flags.writeable = False
    unnormalized = (beta - old_beta) * inputs
    expected_normalizer = float(logsumexp(unnormalized))
    expected_log_weights = unnormalized - expected_normalizer
    expected_ess = int(np.exp(-logsumexp(2 * expected_log_weights)))

    assert inspect.isbuiltin(pymc_rust.weights_at_beta)
    log_weights, normalizer, ess = pymc_rust.weights_at_beta(inputs, old_beta, beta)

    assert log_weights.dtype == np.float64
    assert log_weights.shape == inputs.shape
    np.testing.assert_allclose(log_weights, expected_log_weights, rtol=1e-12, atol=1e-12)
    assert normalizer == pytest.approx(expected_normalizer, rel=1e-12, abs=1e-12)
    assert isinstance(ess, int)
    assert ess == expected_ess
    assert np.exp(log_weights).sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(inputs, before)
    assert not np.shares_memory(inputs, log_weights)


@pytest.mark.parametrize(
    ("values", "old_beta", "beta"),
    [
        ([], 0.0, 0.5),
        ([np.nan], 0.0, 0.5),
        ([np.inf], 0.0, 0.5),
        ([-np.inf], 0.0, 0.5),
        ([0.0], -0.1, 0.5),
        ([0.0], 1.0, 1.5),
        ([0.0], np.nan, 0.5),
        ([0.0], 0.0, np.nan),
        ([0.0], 0.0, np.inf),
        ([0.0], 0.5, 0.25),
        ([0.0], 0.0, 2.1),
    ],
)
def test_invalid_weights_at_beta_inputs(values: list[float], old_beta: float, beta: float) -> None:
    with pytest.raises(ValueError):
        pymc_rust.weights_at_beta(np.array(values, dtype=np.float64), old_beta, beta)


def test_weights_at_beta_layout_and_dtype() -> None:
    with pytest.raises(ValueError, match="C-contiguous"):
        pymc_rust.weights_at_beta(np.zeros(8, dtype=np.float64)[::2], 0.0, 0.5)
    with pytest.raises(TypeError):
        pymc_rust.weights_at_beta(np.zeros(4, dtype=np.float32), 0.0, 0.5)
    with pytest.raises(TypeError):
        pymc_rust.weights_at_beta(np.zeros((2, 2), dtype=np.float64), 0.0, 0.5)


def test_weights_at_beta_outputs_have_independent_storage() -> None:
    values = np.array([-10.0, -3.0, -1.0, 0.0], dtype=np.float64)
    first, _, _ = pymc_rust.weights_at_beta(values, 0.0, 0.5)
    second, _, _ = pymc_rust.weights_at_beta(values, 0.0, 0.5)
    assert not np.shares_memory(first, second)
