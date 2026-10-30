"""Opt-in learning checks: these should fail until the three Rust blanks are filled.

Run explicitly with pytest after rebuilding the extension. These compare the
isolated, actual PyMC method; they do not establish full sample_smc integration.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymc_rust
import pytest

from numpy.typing import NDArray

from pymc.smc.kernels import SMC_KERNEL

reference: Callable[[Any], None] = SMC_KERNEL.update_beta_and_weights


@pytest.mark.parametrize("old_beta", [0.0, 0.35])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "likelihoods",
    [
        [0.0] * 8,
        [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0],
        [-1000.0, -999.0, -998.0, -997.0, -996.0, -995.0, -994.0, -993.0],
        [1000.0, 999.0, 998.0, 997.0, 996.0, 995.0, 994.0, 993.0],
        [-3.0],
    ],
)
def test_temperature_matches_actual_pymc_method(
    likelihoods: list[float], old_beta: float, threshold: float
) -> None:
    values: NDArray[np.float64] = np.array(likelihoods, dtype=np.float64)
    before = values.copy()
    values.flags.writeable = False
    state = SimpleNamespace(
        likelihood_logp=values,
        draws=values.size,
        beta=old_beta,
        threshold=threshold,
        iteration=0,
        log_marginal_likelihood=7.0,
    )
    reference(state)

    beta, weights, increment = pymc_rust.update_beta_and_weights(values, old_beta, threshold)

    assert beta == pytest.approx(state.beta, abs=1e-12, rel=0.0)
    np.testing.assert_allclose(weights, state.weights, rtol=1e-11, atol=1e-13)
    assert increment == pytest.approx(state.log_marginal_likelihood - 7.0, abs=1e-10)
    assert old_beta <= beta <= 1.0
    assert weights.shape == values.shape
    assert weights.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(values, before)
    assert not np.shares_memory(values, weights)


def test_temperature_outputs_have_independent_storage() -> None:
    values = np.array([-10.0, -3.0, -1.0, 0.0], dtype=np.float64)
    _, first, _ = pymc_rust.update_beta_and_weights(values, 0.0, 0.5)
    _, second, _ = pymc_rust.update_beta_and_weights(values, 0.0, 0.5)
    assert not np.shares_memory(first, second)
