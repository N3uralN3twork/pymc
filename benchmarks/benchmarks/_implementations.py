"""Adapters and correctness gates used by ASV and the standalone profile runner."""

import importlib
import inspect
import os

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np

from numpy.typing import NDArray
from scipy.special import logsumexp

from ._workloads import (
    SEED,
    logsumexp_inputs,
    regression_inputs,
    resampling_inputs,
    weights_at_beta_inputs,
)


class ImplementationUnavailable(NotImplementedError):
    """An optional implementation is absent; ASV skips it during setup."""


@dataclass(frozen=True)
class PreparedWorkload:
    run: Callable[[], Any]
    validate: Callable[[Any], None]
    metadata: dict[str, Any]


def optional_module(name: str) -> Any:
    """Skip only an absent module, allowing broken installed implementations to fail."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
            raise ImplementationUnavailable(
                f"Optional implementation {name!r} is not installed"
            ) from exc
        raise


class _FixedUniform:
    """Supply the same pre-generated variate to the existing PyMC kernel."""

    def __init__(self, offset: float) -> None:
        self.value = np.array([offset], dtype=np.float64)
        self.value.flags.writeable = False

    def random(self, size: int) -> NDArray[np.float64]:
        if size != 1:
            raise ValueError("The resampling benchmark supplies exactly one variate")
        return self.value


def _rust_kernel(symbol: str, metadata: dict[str, Any]) -> Callable[..., Any]:
    """Resolve an extension function and record its actual native binary."""
    name = os.environ.get("PYMC_BENCH_RUST_MODULE", "pymc_rust")
    module = optional_module(name)
    kernel: Callable[..., Any] = getattr(module, symbol)
    if not callable(kernel):
        raise TypeError(f"{name}.{symbol} must be callable")
    callable_module = inspect.getmodule(kernel) or module
    module_file = getattr(callable_module, "__file__", None)
    metadata.update(
        rust_module=name,
        rust_module_file=module_file,
        rust_version=getattr(module, "__version__", None),
        rust_debug_assertions=getattr(module, "__debug_assertions__", None),
    )
    if module_file and Path(module_file).is_file():
        metadata["rust_module_sha256"] = sha256(Path(module_file).read_bytes()).hexdigest()
    return kernel


def prepare_logsumexp(
    implementation: str, elements: int, case: str = "normal", seed: int = SEED
) -> PreparedWorkload:
    """Compare the standalone Rust reduction with the SciPy routine used by SMC."""
    inputs = logsumexp_inputs(elements, case, seed)
    baseline: Callable[..., Any] = logsumexp
    reference: float | None = None
    metadata: dict[str, Any] = {
        "workload": "logsumexp",
        "implementation": implementation,
        "baseline": "scipy.special.logsumexp",
        "elements": elements,
        "case": case,
        "seed": seed,
        "dtype": str(inputs.values.dtype),
        "layout": "C",
        "input_sha256": inputs.fingerprint,
        "timing_scope": "Python entry point including input checks, FFI and allocations",
    }
    if implementation == "baseline":
        run = partial(baseline, inputs.values)
    elif implementation == "rust":
        run = partial(_rust_kernel("logsumexp", metadata), inputs.values)
    else:
        raise ValueError(f"Unknown logsumexp implementation: {implementation}")

    def validate(result: Any) -> None:
        nonlocal reference
        if reference is None:
            reference = float(baseline(inputs.values))
        if not isinstance(result, (float, np.floating)) or not np.isfinite(result):
            raise ValueError("logsumexp must return a finite floating-point scalar")
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)
        if inputs.fingerprint != metadata["input_sha256"]:
            raise ValueError("The implementation changed its input values")

    return PreparedWorkload(run, validate, metadata)


def baseline_weights_at_beta(
    likelihood_logp: NDArray[np.float64], old_beta: float, beta: float
) -> tuple[NDArray[np.float64], float, int]:
    """Extract one trial's NumPy/SciPy arithmetic from SMC_KERNEL.update_beta_and_weights."""
    log_weights_un = (beta - old_beta) * likelihood_logp
    log_normalizer = float(logsumexp(log_weights_un))
    log_weights = log_weights_un - log_normalizer
    ess = int(np.exp(-logsumexp(log_weights * 2)))
    return log_weights, log_normalizer, ess


def prepare_weights_at_beta(
    implementation: str,
    elements: int,
    case: str = "normal",
    seed: int = SEED,
    *,
    old_beta: float = 0.25,
    beta: float = 0.75,
) -> PreparedWorkload:
    """Measure one trial temperature, returning all three fields through Python."""
    inputs = weights_at_beta_inputs(elements, case, seed, old_beta=old_beta, beta=beta)
    reference: tuple[NDArray[np.float64], float, int] | None = None
    metadata: dict[str, Any] = {
        "workload": "weights_at_beta",
        "implementation": implementation,
        "baseline": "NumPy/SciPy arithmetic from SMC_KERNEL.update_beta_and_weights",
        "elements": elements,
        "case": case,
        "old_beta": old_beta,
        "beta": beta,
        "seed": seed,
        "dtype": str(inputs.likelihood_logp.dtype),
        "layout": "C",
        "input_sha256": inputs.fingerprint,
        "timing_scope": "One trial beta including Python dispatch, input checks, FFI and allocations",
    }
    if implementation == "baseline":
        kernel: Callable[..., Any] = baseline_weights_at_beta
    elif implementation == "rust":
        kernel = _rust_kernel("weights_at_beta", metadata)
    else:
        raise ValueError(f"Unknown weights_at_beta implementation: {implementation}")
    run = partial(kernel, inputs.likelihood_logp, inputs.old_beta, inputs.beta)

    def validate(result: Any) -> None:
        nonlocal reference
        if reference is None:
            reference = baseline_weights_at_beta(
                inputs.likelihood_logp, inputs.old_beta, inputs.beta
            )
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("weights_at_beta must return (log_weights, log_normalizer, ess)")
        log_weights, log_normalizer, ess = result
        if (
            not isinstance(log_weights, np.ndarray)
            or log_weights.shape != (elements,)
            or log_weights.dtype != np.dtype(np.float64)
            or not np.isfinite(log_weights).all()
        ):
            raise ValueError("Expected finite float64 log weights with shape (elements,)")
        if not isinstance(log_normalizer, (float, np.floating)) or not np.isfinite(log_normalizer):
            raise ValueError("Expected a finite floating-point log normalizer")
        np.testing.assert_allclose(log_weights, reference[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(log_normalizer, reference[1], rtol=1e-12, atol=1e-12)
        if not isinstance(ess, int) or ess != reference[2]:
            raise ValueError(f"Expected integer ESS {reference[2]}, got {ess!r}")
        if inputs.fingerprint != metadata["input_sha256"]:
            raise ValueError("The implementation changed its inputs")

    return PreparedWorkload(run, validate, metadata)


def prepare_resampling(
    implementation: str, particles: int, distribution: str = "uniform", seed: int = SEED
) -> PreparedWorkload:
    """Bind either implementation to identical immutable inputs, outside timing."""
    from pymc.smc.kernels import systematic_resampling

    inputs = resampling_inputs(particles, distribution, seed)
    baseline: Callable[..., NDArray[np.intp]] = systematic_resampling
    reference: NDArray[np.intp] | None = None
    metadata: dict[str, Any] = {
        "workload": "resampling",
        "implementation": implementation,
        "particles": particles,
        "distribution": distribution,
        "seed": seed,
        "dtype": str(inputs.weights.dtype),
        "layout": "C",
        "input_sha256": inputs.fingerprint,
        "rng": "one shared pre-generated uniform offset; RNG generation excluded",
    }
    if implementation == "baseline":
        run = partial(baseline, inputs.weights, _FixedUniform(inputs.offset))
    elif implementation == "rust":
        run = partial(
            _rust_kernel("systematic_resampling", metadata), inputs.weights, inputs.offset
        )
    else:
        raise ValueError(f"Unknown resampling implementation: {implementation}")

    def validate(result: Any) -> None:
        nonlocal reference
        if reference is None:
            reference = baseline(inputs.weights, _FixedUniform(inputs.offset))
        if not isinstance(result, np.ndarray):
            raise TypeError("systematic_resampling must return a NumPy array")
        if result.shape != (particles,) or result.dtype != np.dtype(np.intp):
            raise ValueError("systematic_resampling must return shape (particles,) and dtype intp")
        np.testing.assert_array_equal(result, reference)
        if inputs.fingerprint != metadata["input_sha256"]:
            raise ValueError("The implementation changed its input weights")

    return PreparedWorkload(run, validate, metadata)


def prepare_regression(
    implementation: str,
    observations: int = 200,
    draws: int = 250,
    tune: int = 250,
    seed: int = SEED,
) -> PreparedWorkload:
    """Compare existing PyMC/nutpie workflows, separately from custom Rust kernels."""
    import pymc as pm

    if draws < 1 or tune < 1:
        raise ValueError("draws and tune must be positive")
    if implementation not in ("pymc", "nutpie"):
        raise ValueError(f"Unknown regression implementation: {implementation}")
    if implementation == "nutpie":
        optional_module("nutpie")
    sampler: Literal["pymc", "nutpie"] = "pymc" if implementation == "pymc" else "nutpie"
    inputs = regression_inputs(observations, seed)
    with pm.Model(coords={"feature": ["a", "b", "c"]}) as model:
        beta: Any = pm.Normal("beta", dims="feature")
        pm.Normal("y", mu=inputs.x @ beta, sigma=0.5, observed=inputs.y)

    def run() -> Any:
        with model:
            return pm.sample(
                draws=draws,
                tune=tune,
                chains=2,
                cores=1,
                nuts_sampler=sampler,
                backend="numba",
                random_seed=seed,
                initvals={"beta": np.zeros(3)},
                nuts={"target_accept": 0.9},
                blas_cores=1,
                progressbar=False,
                quiet=True,
                compute_convergence_checks=False,
            )

    def validate(result: Any) -> None:
        values = np.asarray(result.posterior["beta"])
        if values.shape != (2, draws, 3) or not np.isfinite(values).all():
            raise ValueError("Expected finite beta draws with shape (2, draws, 3)")
        if list(result.posterior.coords["feature"].values) != ["a", "b", "c"]:
            raise ValueError("Sampling changed the feature coordinates")

    return PreparedWorkload(
        run,
        validate,
        {
            "workload": "regression",
            "implementation": implementation,
            "backend": "numba",
            "observations": observations,
            "draws": draws,
            "tune": tune,
            "chains": 2,
            "cores": 1,
            "seed": seed,
            "input_sha256": inputs.fingerprint,
            "timing_scope": "pm.sample including compilation/cache lookup, tuning and conversion",
        },
    )


def prepare(
    workload: str,
    implementation: str,
    *,
    particles: int = 10000,
    distribution: str = "uniform",
    elements: int = 10000,
    case: str = "normal",
    old_beta: float = 0.25,
    beta: float = 0.75,
    observations: int = 200,
    draws: int = 250,
    tune: int = 250,
    seed: int = SEED,
) -> PreparedWorkload:
    """Dispatch both measurement tools through the same factories and validators."""
    if workload == "resampling":
        return prepare_resampling(implementation, particles, distribution, seed)
    if workload == "logsumexp":
        return prepare_logsumexp(implementation, elements, case, seed)
    if workload == "weights_at_beta":
        return prepare_weights_at_beta(
            implementation, elements, case, seed, old_beta=old_beta, beta=beta
        )
    if workload == "regression":
        return prepare_regression(implementation, observations, draws, tune, seed)
    raise ValueError(f"Unknown workload: {workload}")
