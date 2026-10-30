"""Check the trial-temperature benchmark's inputs, reference, and correctness gates."""

import json
import subprocess
import sys

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks._implementations import (
    baseline_weights_at_beta,
    prepare,
    prepare_weights_at_beta,
)
from benchmarks._workloads import weights_at_beta_inputs
from numpy.typing import NDArray


@pytest.mark.parametrize("case", ["normal", "large_positive", "large_negative"])
def test_weights_at_beta_inputs_and_adapters(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    inputs = weights_at_beta_inputs(128, case)
    assert inputs.fingerprint == weights_at_beta_inputs(128, case).fingerprint
    assert inputs.fingerprint != weights_at_beta_inputs(128, case, seed=17).fingerprint
    assert inputs.fingerprint != weights_at_beta_inputs(128, case, beta=1.5).fingerprint
    assert inputs.likelihood_logp.dtype == np.float64
    assert inputs.likelihood_logp.flags.c_contiguous
    assert not inputs.likelihood_logp.flags.writeable
    calls: list[tuple[NDArray[np.float64], float, float]] = []
    module = ModuleType("comparison_weights_at_beta_rust")

    def kernel(
        values: NDArray[np.float64], old_beta: float, beta: float
    ) -> tuple[NDArray[np.float64], float, int]:
        calls.append((values, old_beta, beta))
        return baseline_weights_at_beta(values, old_beta, beta)

    setattr(module, "weights_at_beta", kernel)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("PYMC_BENCH_RUST_MODULE", module.__name__)
    baseline = prepare("weights_at_beta", "baseline", elements=128, case=case)
    rust = prepare_weights_at_beta("rust", 128, case)
    assert calls == []
    rust.validate(rust.run())
    baseline.validate(baseline.run())
    np.testing.assert_array_equal(calls[0][0], inputs.likelihood_logp)
    assert calls[0][1:] == (inputs.old_beta, inputs.beta)
    assert rust.metadata["input_sha256"] == baseline.metadata["input_sha256"]


def test_weights_reference_agrees_with_pymc_temperature_update() -> None:
    from pymc.smc.kernels import SMC_KERNEL

    values = np.array([-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0])
    old_beta = 0.35
    state = SimpleNamespace(
        likelihood_logp=values,
        draws=values.size,
        beta=old_beta,
        threshold=0.5,
        iteration=0,
        log_marginal_likelihood=7.0,
    )
    update: Callable[[Any], None] = SMC_KERNEL.update_beta_and_weights
    update(state)
    log_weights, normalizer, _ = baseline_weights_at_beta(values, old_beta, state.beta)
    weights = np.exp(log_weights)
    weights /= weights.sum()
    np.testing.assert_allclose(weights, state.weights, rtol=1e-12, atol=1e-12)
    assert normalizer - np.log(values.size) == pytest.approx(state.log_marginal_likelihood - 7.0)


@pytest.mark.parametrize("wrong_field", ["log_weights", "normalizer", "ess"])
def test_weights_at_beta_rejects_wrong_results(wrong_field: str) -> None:
    workload = prepare_weights_at_beta("baseline", 128)
    log_weights, normalizer, ess = workload.run()
    if wrong_field == "log_weights":
        log_weights = log_weights + 1.0
    elif wrong_field == "normalizer":
        normalizer += 1.0
    else:
        ess += 1
    with pytest.raises((AssertionError, ValueError)):
        workload.validate((log_weights, normalizer, ess))


def test_weights_at_beta_cli_records_temperatures(tmp_path: Path) -> None:
    output = tmp_path / "weights.json"
    driver = Path(__file__).resolve().parents[1] / "profile_workload.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--workload",
            "weights_at_beta",
            "--implementation",
            "baseline",
            "--elements",
            "16",
            "--case",
            "large_negative",
            "--old-beta",
            "0.4",
            "--beta",
            "1.5",
            "--iterations",
            "1",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    workload = report["samples"][0]["workload"]
    assert workload["workload"] == "weights_at_beta"
    assert workload["old_beta"] == 0.4
    assert workload["beta"] == 1.5
    expected = weights_at_beta_inputs(16, "large_negative", old_beta=0.4, beta=1.5)
    assert workload["input_sha256"] == expected.fingerprint
    assert report["median_seconds_per_call"] > 0
