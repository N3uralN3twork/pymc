"""Check the exercise's completed Python boundary, without requiring its answers."""

import inspect

import numpy as np
import pymc_rust
import pytest


def test_temperature_import_is_native() -> None:
    assert inspect.isbuiltin(pymc_rust.update_beta_and_weights)


@pytest.mark.parametrize(
    ("likelihoods", "beta", "threshold", "message"),
    [
        ([], 0.0, 0.5, "empty"),
        ([np.nan], 0.0, 0.5, "finite"),
        ([np.inf], 0.0, 0.5, "finite"),
        ([-np.inf], 0.0, 0.5, "finite"),
        ([0.0], -0.1, 0.5, "old_beta"),
        ([0.0], 1.0, 0.5, "old_beta"),
        ([0.0], np.nan, 0.5, "old_beta"),
        ([0.0], 0.0, -0.1, "threshold"),
        ([0.0], 0.0, 1.1, "threshold"),
        ([0.0], 0.0, np.nan, "threshold"),
    ],
)
def test_invalid_temperature_inputs(
    likelihoods: list[float], beta: float, threshold: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pymc_rust.update_beta_and_weights(np.array(likelihoods, dtype=np.float64), beta, threshold)


def test_temperature_layout_and_dtype() -> None:
    with pytest.raises(ValueError, match="C-contiguous"):
        pymc_rust.update_beta_and_weights(np.zeros(8, dtype=np.float64)[::2], 0.0, 0.5)
    with pytest.raises(TypeError):
        pymc_rust.update_beta_and_weights(np.zeros(4, dtype=np.float32), 0.0, 0.5)
    with pytest.raises(TypeError):
        pymc_rust.update_beta_and_weights(np.zeros((2, 2), dtype=np.float64), 0.0, 0.5)
