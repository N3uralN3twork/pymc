"""Python interface to the native Rust kernels."""

import numpy as np

from numpy.typing import NDArray

__version__: str
__debug_assertions__: bool

def logsumexp(values: NDArray[np.float64]) -> float: ...
def systematic_resampling(weights: NDArray[np.float64], offset: float) -> NDArray[np.intp]: ...
def weights_at_beta(
    likelihood_logp: NDArray[np.float64], old_beta: float, beta: float
) -> tuple[NDArray[np.float64], float, int]: ...
def update_beta_and_weights(
    likelihood_logp: NDArray[np.float64], old_beta: float, threshold: float
) -> tuple[float, NDArray[np.float64], float]: ...
