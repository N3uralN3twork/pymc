"""Deterministic, offline inputs shared by Python and Rust comparisons."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from numpy.typing import NDArray

SEED = 20261029


@dataclass(frozen=True)
class LogSumExpInputs:
    values: NDArray[np.float64]

    @property
    def fingerprint(self) -> str:
        return sha256(self.values.tobytes()).hexdigest()


def logsumexp_inputs(elements: int, case: str = "normal", seed: int = SEED) -> LogSumExpInputs:
    """Use the same finite float64 inputs for ordinary and shifted reductions."""
    if elements < 1:
        raise ValueError("elements must be positive")
    offsets = {"normal": 0.0, "large_positive": 1000.0, "large_negative": -1000.0}
    if case not in offsets:
        raise ValueError(f"Unknown logsumexp case: {case}")
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=3.0, size=elements) + offsets[case]
    values.flags.writeable = False
    return LogSumExpInputs(values)


@dataclass(frozen=True)
class WeightsAtBetaInputs:
    likelihood_logp: NDArray[np.float64]
    old_beta: float
    beta: float

    @property
    def fingerprint(self) -> str:
        temperatures = np.array([self.old_beta, self.beta], dtype=np.float64)
        return sha256(self.likelihood_logp.tobytes() + temperatures.tobytes()).hexdigest()


def weights_at_beta_inputs(
    elements: int,
    case: str = "normal",
    seed: int = SEED,
    *,
    old_beta: float = 0.25,
    beta: float = 0.75,
) -> WeightsAtBetaInputs:
    """Reuse the finite likelihood corpus, with temperatures included in its hash."""
    if not 0.0 <= old_beta < 1.0:
        raise ValueError("old_beta must be finite and in [0, 1)")
    if not old_beta <= beta <= 2.0:
        raise ValueError("beta must be finite and in [old_beta, 2]")
    values = logsumexp_inputs(elements, case, seed).values
    return WeightsAtBetaInputs(values, old_beta, beta)


@dataclass(frozen=True)
class ResamplingInputs:
    weights: NDArray[np.float64]
    offset: float

    @property
    def fingerprint(self) -> str:
        """Identify the exact numeric inputs, including the supplied random offset."""
        payload = self.weights.tobytes() + np.float64(self.offset).tobytes()
        return sha256(payload).hexdigest()


def resampling_inputs(
    particles: int, distribution: str = "uniform", seed: int = SEED
) -> ResamplingInputs:
    """Create normalized, contiguous float64 weights and a uniform offset in [0, 1)."""
    if particles < 1:
        raise ValueError("particles must be positive")
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        weights = np.ones(particles, dtype=np.float64)
    elif distribution == "skewed":
        weights = rng.lognormal(sigma=3.0, size=particles)
    else:
        raise ValueError(f"Unknown weight distribution: {distribution}")
    weights /= weights.sum()
    weights.flags.writeable = False
    return ResamplingInputs(weights, float(rng.random()))


@dataclass(frozen=True)
class RegressionInputs:
    x: NDArray[np.float64]
    y: NDArray[np.float64]

    @property
    def fingerprint(self) -> str:
        return sha256(self.x.tobytes() + self.y.tobytes()).hexdigest()


def regression_inputs(observations: int, seed: int = SEED) -> RegressionInputs:
    """Generate a small continuous model without downloads or global RNG state."""
    if observations < 1:
        raise ValueError("observations must be positive")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(observations, 3))
    y = x @ np.array([0.5, -1.0, 1.5]) + rng.normal(scale=0.5, size=observations)
    x.flags.writeable = False
    y.flags.writeable = False
    return RegressionInputs(x, y)
