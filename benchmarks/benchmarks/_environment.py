"""Apply the same thread budget before importing either numerical implementation."""

import os

from pathlib import Path

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def configure_threads() -> dict[str, str]:
    """Use a fixed single-thread budget for the initial comparison suite."""
    controls = dict.fromkeys(THREAD_VARIABLES, "1")
    os.environ.update(controls)
    cache = Path(__file__).resolve().parents[1] / "results" / "cache"
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    if not any(
        part.split("=", 1)[0].strip() in {"compiledir", "base_compiledir"}
        for part in flags.split(",")
    ):
        flag = f"base_compiledir={cache.as_posix()}/pytensor"
        os.environ["PYTENSOR_FLAGS"] = f"{flags},{flag}" if flags else flag
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    return controls
