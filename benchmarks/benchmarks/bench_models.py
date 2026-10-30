"""Warm repeated PyMC API calls; this is not isolated sampler-kernel timing."""

from typing import Any

from threadpoolctl import threadpool_limits

from ._implementations import prepare_regression


class Regression:
    params = (["pymc", "nutpie"], [200, 2000])
    param_names = ["implementation", "observations"]
    number = 1
    repeat = 3
    rounds = 2
    warmup_time = 0
    timeout = 300

    def setup(self, implementation: str, observations: int) -> None:
        self.workload = prepare_regression(implementation, observations)
        with threadpool_limits(limits=1):
            self.workload.validate(self.workload.run())

    def time_sample(self, implementation: str, observations: int) -> Any:
        with threadpool_limits(limits=1):
            return self.workload.run()

    def peakmem_sample(self, implementation: str, observations: int) -> Any:
        with threadpool_limits(limits=1):
            return self.workload.run()
