"""Check metric normalization, repeat aggregation, and paired execution order."""

from collections.abc import Callable

import pytest

from benchmarks.bench_kernels import ResamplingSpeedup, ResamplingThroughput

from benchmarks import _metrics


def test_metrics_normalize_independently_calibrated_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration: dict[str, list[int]] = {"baseline": [], "rust": []}
    measured: list[tuple[str, int]] = []

    def baseline() -> str:
        return "baseline"

    def rust() -> str:
        return "rust"

    class FakeTimer:
        def __init__(self, function: Callable[[], object]) -> None:
            self.name = str(function())
            self.seconds = 0.004 if self.name == "baseline" else 0.001

        def timeit(self, number: int = 1) -> float:
            history = calibration[self.name]
            if not history or history[-1] * self.seconds < 0.01:
                history.append(number)
            else:
                measured.append((self.name, number))
            return number * self.seconds

    monkeypatch.setattr(_metrics, "Timer", FakeTimer)
    seconds = _metrics.measure_seconds_per_call(baseline, rust)
    assert seconds == pytest.approx((0.004, 0.001))
    assert calibration == {"baseline": [1, 2, 4], "rust": [1, 2, 4, 8, 16]}
    assert measured == [
        ("baseline", 4),
        ("rust", 16),
        ("rust", 16),
        ("baseline", 4),
        ("baseline", 4),
        ("rust", 16),
        ("rust", 16),
        ("baseline", 4),
        ("baseline", 4),
        ("rust", 16),
        ("rust", 16),
        ("baseline", 4),
    ]


def test_measured_batches_use_the_median(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTimer:
        def __init__(self, function: Callable[[], object]) -> None:
            self.values = iter([0.02, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01])

        def timeit(self, number: int = 1) -> float:
            return next(self.values)

    monkeypatch.setattr(_metrics, "Timer", FakeTimer)
    assert _metrics.measure_seconds_per_call(lambda: None) == pytest.approx((0.01,))


def test_throughput_and_speedup_units(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks._implementations import PreparedWorkload

    workload = PreparedWorkload(lambda: None, lambda result: None, {})
    throughput = ResamplingThroughput()
    throughput.workload = workload
    monkeypatch.setattr(
        "benchmarks.bench_kernels.measure_seconds_per_call", lambda *functions: (0.002,)
    )
    assert throughput.track_million_particles_per_second("baseline", 1000, "uniform") == 0.5
    assert throughput.unit == "million particles/s"

    speedup = ResamplingSpeedup()
    speedup.baseline = workload
    speedup.rust = workload
    monkeypatch.setattr(
        "benchmarks.bench_kernels.measure_seconds_per_call", lambda *functions: (0.008, 0.002)
    )
    assert speedup.track_rust_speedup(1000, "uniform") == 4
    assert speedup.unit == "x"
