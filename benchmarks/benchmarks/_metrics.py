"""Bounded, repeated timings for scalar ASV throughput and speedup metrics."""

from collections.abc import Callable
from statistics import median
from timeit import Timer

_MIN_SAMPLE_SECONDS = 0.01
_SAMPLE_ROUNDS = 6


def measure_seconds_per_call(*functions: Callable[[], object]) -> tuple[float, ...]:
    """Measure warmed callables with comparable timing budgets and alternating order.

    Calibrate each callable independently so even short Rust calls occupy enough
    time to measure. Validation and fixture creation belong outside this helper.
    Six measured batches give each member of a pair three first-in-order runs.
    """
    timers = [Timer(function) for function in functions]
    batch_sizes = []
    for timer in timers:
        number = 1
        while timer.timeit(number=number) < _MIN_SAMPLE_SECONDS:
            number *= 2
        batch_sizes.append(number)

    samples: list[list[float]] = [[] for _ in timers]
    for sample in range(_SAMPLE_ROUNDS):
        order = range(len(timers)) if sample % 2 == 0 else reversed(range(len(timers)))
        for index in order:
            elapsed = timers[index].timeit(number=batch_sizes[index])
            samples[index].append(elapsed / batch_sizes[index])
    return tuple(median(values) for values in samples)
