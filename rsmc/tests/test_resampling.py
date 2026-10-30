"""Check the installed native extension against the actual PyMC kernel."""

import inspect

from collections.abc import Callable
from typing import Any

import numpy as np
import pymc_rust
import pytest

from numpy.typing import NDArray

from pymc.smc.kernels import systematic_resampling as untyped_reference

reference: Callable[..., NDArray[np.intp]] = untyped_reference


class FixedUniform:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def random(self, size: int) -> Any:
        assert size == 1
        return np.array([self.offset], dtype=np.float64)


def test_import_is_native() -> None:
    assert inspect.isbuiltin(pymc_rust.systematic_resampling)
    assert pymc_rust.__version__
    assert isinstance(pymc_rust.__debug_assertions__, bool)


@pytest.mark.parametrize("particles", [1, 2, 7, 100, 10000])
@pytest.mark.parametrize("offset", [0.0, 0.37, 0.95])
@pytest.mark.parametrize("seed", [17, 31])
def test_matches_pymc(particles: int, offset: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.lognormal(sigma=3.0, size=particles)
    weights /= weights.sum()
    before = weights.copy()
    weights.flags.writeable = False
    expected = reference(weights, FixedUniform(offset))
    actual = pymc_rust.systematic_resampling(weights, offset)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(weights, before)
    assert actual.dtype == np.dtype(np.intp)
    assert actual.shape == (particles,)


@pytest.mark.parametrize(
    ("weights", "offset"),
    [
        ([0.25, 0.25, 0.25, 0.25], 0.0),
        ([0.0, 1.0, 0.0], 0.0),
        ([0.0, 0.0, 1.0], 0.5),
    ],
)
def test_cumulative_boundaries(weights: list[float], offset: float) -> None:
    values = np.array(weights, dtype=np.float64)
    np.testing.assert_array_equal(
        pymc_rust.systematic_resampling(values, offset),
        reference(values, FixedUniform(offset)),
    )


@pytest.mark.parametrize(
    ("weights", "offset"),
    [
        ([], 0.5),
        ([1.0], -0.1),
        ([1.0], 1.0),
        ([1.0], np.nan),
        ([-1.0, 2.0], 0.5),
        ([np.nan, 1.0], 0.5),
        ([np.inf, 1.0], 0.5),
        ([0.2, 0.2], 0.5),
    ],
)
def test_invalid_values_raise_python_errors(weights: list[float], offset: float) -> None:
    with pytest.raises(ValueError):
        pymc_rust.systematic_resampling(np.array(weights, dtype=np.float64), offset)


def test_layout_and_dtype_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="C-contiguous"):
        pymc_rust.systematic_resampling((np.ones(8, dtype=np.float64) / 4)[::2], 0.5)
    with pytest.raises(TypeError):
        pymc_rust.systematic_resampling(np.array([1.0], dtype=np.float32), 0.5)
    with pytest.raises(TypeError):
        pymc_rust.systematic_resampling(np.array([[1.0]], dtype=np.float64), 0.5)


def test_outputs_do_not_alias_inputs_or_other_calls() -> None:
    weights = np.array([0.25, 0.75], dtype=np.float64)
    first = pymc_rust.systematic_resampling(weights, 0.5)
    second = pymc_rust.systematic_resampling(weights, 0.5)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, weights)
