"""Correctness gates and reproducible inputs for the logsumexp benchmark."""

import json
import subprocess
import sys

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from benchmarks._implementations import prepare, prepare_logsumexp
from benchmarks._workloads import logsumexp_inputs
from numpy.typing import NDArray
from scipy.special import logsumexp


@pytest.mark.parametrize("case", ["normal", "large_positive", "large_negative"])
def test_logsumexp_inputs_and_adapters(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    inputs = logsumexp_inputs(128, case)
    assert inputs.fingerprint == logsumexp_inputs(128, case).fingerprint
    assert inputs.fingerprint != logsumexp_inputs(128, case, seed=17).fingerprint
    assert inputs.values.dtype == np.float64
    assert inputs.values.flags.c_contiguous
    assert not inputs.values.flags.writeable
    calls: list[NDArray[np.float64]] = []
    module = ModuleType("comparison_logsumexp_rust")

    def kernel(values: NDArray[np.float64]) -> float:
        calls.append(values)
        return float(logsumexp(values))

    setattr(module, "logsumexp", kernel)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("PYMC_BENCH_RUST_MODULE", module.__name__)
    baseline = prepare("logsumexp", "baseline", elements=128, case=case)
    rust = prepare_logsumexp("rust", 128, case)
    assert calls == []  # Preparation must not warm the kernel.
    rust.validate(rust.run())
    baseline.validate(baseline.run())
    np.testing.assert_array_equal(calls[0], inputs.values)
    assert not calls[0].flags.writeable
    assert rust.metadata["input_sha256"] == baseline.metadata["input_sha256"]
    assert rust.metadata["baseline"] == "scipy.special.logsumexp"


@pytest.mark.parametrize("result", [0.0, np.inf, np.array([1.0])])
def test_logsumexp_rejects_wrong_results(result: Any) -> None:
    workload = prepare_logsumexp("baseline", 128)
    with pytest.raises((AssertionError, ValueError)):
        workload.validate(result)


def test_logsumexp_cli_records_selected_inputs(tmp_path: Path) -> None:
    output = tmp_path / "logsumexp.json"
    driver = Path(__file__).resolve().parents[1] / "profile_workload.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--workload",
            "logsumexp",
            "--implementation",
            "baseline",
            "--elements",
            "16",
            "--case",
            "large_positive",
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
    assert workload["workload"] == "logsumexp"
    assert workload["elements"] == 16
    assert workload["case"] == "large_positive"
    assert workload["input_sha256"] == logsumexp_inputs(16, "large_positive").fingerprint
    assert report["median_seconds_per_call"] > 0
