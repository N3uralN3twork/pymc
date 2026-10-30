"""Behavioral checks for the shared comparison harness, without a Rust dependency."""

import json
import os
import subprocess
import sys

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from benchmarks._implementations import (
    ImplementationUnavailable,
    optional_module,
    prepare_regression,
    prepare_resampling,
)
from benchmarks._workloads import regression_inputs, resampling_inputs

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "profile_workload.py"


@pytest.mark.parametrize("distribution", ["uniform", "skewed"])
def test_shared_inputs_are_reproducible_and_readonly(distribution: str) -> None:
    first = resampling_inputs(100, distribution, seed=17)
    second = resampling_inputs(100, distribution, seed=17)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != resampling_inputs(100, distribution, seed=18).fingerprint
    assert not first.weights.flags.writeable
    assert first.weights.flags.c_contiguous
    assert first.weights.dtype == np.float64
    assert first.weights.sum() == pytest.approx(1.0)
    assert 0 <= first.offset < 1
    model = regression_inputs(20)
    assert model.fingerprint == regression_inputs(20).fingerprint
    assert not model.x.flags.writeable
    assert not model.y.flags.writeable


@pytest.mark.parametrize("distribution", ["uniform", "skewed"])
def test_rust_adapter_receives_the_same_inputs(
    monkeypatch: pytest.MonkeyPatch, distribution: str
) -> None:
    inputs = resampling_inputs(100, distribution)
    calls = []
    module = ModuleType("comparison_test_rust")

    def kernel(weights: Any, offset: float) -> Any:
        calls.append((weights.copy(), offset))
        positions = (offset + np.arange(len(weights))) / len(weights)
        return np.searchsorted(np.cumsum(weights), positions, side="left").astype(np.intp)

    module.systematic_resampling = kernel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("PYMC_BENCH_RUST_MODULE", module.__name__)
    baseline = prepare_resampling("baseline", 100, distribution)
    rust = prepare_resampling("rust", 100, distribution)
    output = rust.run()
    rust.validate(output)
    baseline.validate(baseline.run())
    np.testing.assert_array_equal(output, baseline.run())
    np.testing.assert_array_equal(calls[0][0], inputs.weights)
    assert calls[0][1] == inputs.offset
    assert rust.metadata["input_sha256"] == baseline.metadata["input_sha256"]


def test_wrong_rust_result_fails_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("comparison_test_bad_rust")
    module.systematic_resampling = lambda weights, offset: np.zeros(len(weights), dtype=np.intp)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("PYMC_BENCH_RUST_MODULE", module.__name__)
    workload = prepare_resampling("rust", 50)
    with pytest.raises(AssertionError):
        workload.validate(workload.run())


def test_unavailable_rust_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYMC_BENCH_RUST_MODULE", "comparison_definitely_not_installed")
    with pytest.raises(ImplementationUnavailable, match="not installed"):
        prepare_resampling("rust", 5)


def test_dependency_failure_is_not_treated_as_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_import(name: str) -> None:
        raise ModuleNotFoundError("missing dependency", name="internal_dependency")

    monkeypatch.setattr("benchmarks._implementations.importlib.import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        optional_module("installed_extension")


@pytest.mark.parametrize("phase", ["cold", "warm"])
def test_cli_uses_fresh_workers_and_records_metadata(tmp_path: Path, phase: str) -> None:
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--implementation",
            "baseline",
            "--particles",
            "16",
            "--iterations",
            "1",
            "--repeats",
            "2",
            "--phase",
            phase,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert not report["diagnostic_only"]
    assert len(report["samples"]) == 2
    first, second = report["samples"]
    assert first["environment"]["pid"] != second["environment"]["pid"]
    assert first["workload"]["input_sha256"] == second["workload"]["input_sha256"]
    assert all(value == "1" for value in first["thread_controls"].values())
    assert all(pool["num_threads"] == 1 for pool in first["environment"]["threadpools"])
    assert report["median_seconds_per_call"] > 0


def test_cli_marks_profiles_as_diagnostics(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"
    result = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--implementation",
            "baseline",
            "--particles",
            "16",
            "--iterations",
            "1",
            "--repeats",
            "1",
            "--profile",
            "cprofile",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["diagnostic_only"]
    assert "median_seconds_per_call" not in report
    assert Path(report["samples"][0]["profile_file"]).is_file()


def test_cli_reports_missing_extension(tmp_path: Path) -> None:
    output = tmp_path / "missing.json"
    env = {**os.environ, "PYMC_BENCH_RUST_MODULE": "comparison_definitely_not_installed"}
    result = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--implementation",
            "rust",
            "--particles",
            "16",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads(output.read_text())
    assert report["samples"][0]["status"] == "unavailable"
    assert "median_seconds_per_call" not in report


@pytest.mark.parametrize("implementation", ["pymc", "nutpie"])
def test_regression_runs_the_real_pymc_workflow(implementation: str) -> None:
    if implementation == "nutpie":
        pytest.importorskip("nutpie")
    workload = prepare_regression(implementation, observations=16, draws=8, tune=8)
    workload.validate(workload.run())


def test_preparation_does_not_warm_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    import pymc.smc.kernels as kernels

    original: Any = kernels.systematic_resampling
    calls: list[int] = []

    def counted(weights: Any, rng: Any) -> Any:
        calls.append(len(weights))
        return original(weights, rng)

    monkeypatch.setattr(kernels, "systematic_resampling", counted)
    workload = prepare_resampling("baseline", 8)
    assert calls == []
    value = workload.run()
    assert calls == [8]
    workload.validate(value)
    assert calls == [8, 8]


def test_installed_native_adapter_records_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    from hashlib import sha256
    from importlib.machinery import EXTENSION_SUFFIXES

    pytest.importorskip("pymc_rust")
    monkeypatch.setenv("PYMC_BENCH_RUST_MODULE", "pymc_rust")
    workload = prepare_resampling("rust", 128, "skewed")
    workload.validate(workload.run())
    binary = Path(workload.metadata["rust_module_file"])
    assert any(str(binary).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    assert workload.metadata["rust_module_sha256"] == sha256(binary.read_bytes()).hexdigest()
    assert isinstance(workload.metadata["rust_debug_assertions"], bool)
