"""Python-entry-point kernel comparisons; FFI and output allocations are timed."""

from typing import Any

from threadpoolctl import threadpool_limits

from ._implementations import (
    PreparedWorkload,
    prepare_logsumexp,
    prepare_resampling,
    prepare_weights_at_beta,
)
from ._metrics import measure_seconds_per_call


def _validated_logsumexp(implementation: str, elements: int, case: str) -> PreparedWorkload:
    workload = prepare_logsumexp(implementation, elements, case)
    with threadpool_limits(limits=1):
        workload.validate(workload.run())
    return workload


class LogSumExp:
    params = (
        ["baseline", "rust"],
        [100, 10000, 100000],
        ["normal", "large_positive", "large_negative"],
    )
    param_names = ["implementation", "elements", "case"]
    repeat = 5
    rounds = 2
    warmup_time = 0.1
    timeout = 120

    def setup(self, implementation: str, elements: int, case: str) -> None:
        self.workload = _validated_logsumexp(implementation, elements, case)

    def time_logsumexp(self, implementation: str, elements: int, case: str) -> Any:
        return self.workload.run()

    def peakmem_logsumexp(self, implementation: str, elements: int, case: str) -> Any:
        return self.workload.run()


class LogSumExpThroughput:
    params = LogSumExp.params
    param_names = LogSumExp.param_names
    unit = "million elements/s"
    timeout = 120

    def setup(self, implementation: str, elements: int, case: str) -> None:
        self.workload = _validated_logsumexp(implementation, elements, case)

    def track_million_elements_per_second(
        self, implementation: str, elements: int, case: str
    ) -> float:
        (seconds,) = measure_seconds_per_call(self.workload.run)
        return float(f"{elements / seconds / 1e6:.3g}")


class LogSumExpSpeedup:
    params = (LogSumExp.params[1], LogSumExp.params[2])
    param_names = ["elements", "case"]
    unit = "x"
    timeout = 120

    def setup(self, elements: int, case: str) -> None:
        self.baseline = _validated_logsumexp("baseline", elements, case)
        self.rust = _validated_logsumexp("rust", elements, case)

    def track_rust_speedup(self, elements: int, case: str) -> float:
        baseline_seconds, rust_seconds = measure_seconds_per_call(self.baseline.run, self.rust.run)
        return float(f"{baseline_seconds / rust_seconds:.3g}")


def _validated_weights_at_beta(implementation: str, elements: int, case: str) -> PreparedWorkload:
    workload = prepare_weights_at_beta(implementation, elements, case)
    with threadpool_limits(limits=1):
        workload.validate(workload.run())
    return workload


class WeightsAtBeta:
    params = LogSumExp.params
    param_names = LogSumExp.param_names
    repeat = 5
    rounds = 2
    warmup_time = 0.1
    timeout = 120

    def setup(self, implementation: str, elements: int, case: str) -> None:
        self.workload = _validated_weights_at_beta(implementation, elements, case)

    def time_weights_at_beta(self, implementation: str, elements: int, case: str) -> Any:
        return self.workload.run()

    def peakmem_weights_at_beta(self, implementation: str, elements: int, case: str) -> Any:
        return self.workload.run()


class WeightsAtBetaThroughput:
    params = WeightsAtBeta.params
    param_names = WeightsAtBeta.param_names
    unit = "million elements/s"
    timeout = 120

    def setup(self, implementation: str, elements: int, case: str) -> None:
        self.workload = _validated_weights_at_beta(implementation, elements, case)

    def track_million_elements_per_second(
        self, implementation: str, elements: int, case: str
    ) -> float:
        (seconds,) = measure_seconds_per_call(self.workload.run)
        return float(f"{elements / seconds / 1e6:.3g}")


class WeightsAtBetaSpeedup:
    params = (WeightsAtBeta.params[1], WeightsAtBeta.params[2])
    param_names = ["elements", "case"]
    unit = "x"
    timeout = 120

    def setup(self, elements: int, case: str) -> None:
        self.baseline = _validated_weights_at_beta("baseline", elements, case)
        self.rust = _validated_weights_at_beta("rust", elements, case)

    def track_rust_speedup(self, elements: int, case: str) -> float:
        baseline_seconds, rust_seconds = measure_seconds_per_call(self.baseline.run, self.rust.run)
        return float(f"{baseline_seconds / rust_seconds:.3g}")


def _validated_resampling(
    implementation: str, particles: int, distribution: str
) -> PreparedWorkload:
    workload = prepare_resampling(implementation, particles, distribution)
    with threadpool_limits(limits=1):
        workload.validate(workload.run())
    return workload


class Resampling:
    params = (["baseline", "rust"], [100, 10000, 100000], ["uniform", "skewed"])
    param_names = ["implementation", "particles", "distribution"]
    repeat = 5
    rounds = 2
    warmup_time = 0.1
    timeout = 120

    def setup(self, implementation: str, particles: int, distribution: str) -> None:
        self.workload = _validated_resampling(implementation, particles, distribution)

    def time_resampling(self, implementation: str, particles: int, distribution: str) -> Any:
        return self.workload.run()

    def peakmem_resampling(self, implementation: str, particles: int, distribution: str) -> Any:
        return self.workload.run()


class ResamplingThroughput:
    params = Resampling.params
    param_names = Resampling.param_names
    unit = "million particles/s"
    timeout = 120

    def setup(self, implementation: str, particles: int, distribution: str) -> None:
        self.workload = _validated_resampling(implementation, particles, distribution)

    def track_million_particles_per_second(
        self, implementation: str, particles: int, distribution: str
    ) -> float:
        (seconds,) = measure_seconds_per_call(self.workload.run)
        return float(f"{particles / seconds / 1e6:.3g}")


class ResamplingSpeedup:
    params = (Resampling.params[1], Resampling.params[2])
    param_names = ["particles", "distribution"]
    unit = "x"
    timeout = 120

    def setup(self, particles: int, distribution: str) -> None:
        self.baseline = _validated_resampling("baseline", particles, distribution)
        self.rust = _validated_resampling("rust", particles, distribution)

    def track_rust_speedup(self, particles: int, distribution: str) -> float:
        baseline_seconds, rust_seconds = measure_seconds_per_call(self.baseline.run, self.rust.run)
        return float(f"{baseline_seconds / rust_seconds:.3g}")
