//! Thin NumPy/Python interface; array conversion costs belong in benchmarks.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::{resampling, temperature};

/// Stable logsumexp of a nonempty C-contiguous float64 vector of finite values.
///
/// Borrow inputs while holding the GIL. Do not concurrently mutate an alias.
#[pyfunction]
fn logsumexp(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let values = values
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be a C-contiguous float64 vector"))?;
    if values.is_empty() {
        return Err(PyValueError::new_err("values must not be empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    temperature::logsumexp(values).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Resample normalized float64 weights using one supplied offset in [0, 1).
///
/// Input must be one-dimensional and C-contiguous. The returned vector has dtype
/// numpy.intp. Inputs are borrowed without copying and are never modified.
/// The caller must not concurrently mutate an alias of the input.
#[pyfunction]
fn systematic_resampling<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f64>,
    offset: f64,
) -> PyResult<Bound<'py, PyArray1<isize>>> {
    let weights = weights
        .as_slice()
        .map_err(|_| PyValueError::new_err("weights must be a C-contiguous float64 vector"))?;
    // Keep the GIL while borrowing NumPy memory. Detaching requires a separate
    // ownership/concurrency design; do not send this borrowed slice to workers.
    let indices = resampling::systematic_resampling(weights, offset)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(indices.into_pyarray(py))
}

/// Return (normalized log weights, log normalizer, integer ESS) at one trial beta.
///
/// Inputs are nonempty finite contiguous float64 likelihoods, old_beta in
/// [0, 1), and beta in [old_beta, 2]. Scaled likelihoods must remain finite.
/// Hold the GIL while borrowing input; do not concurrently mutate an alias.
#[pyfunction]
fn weights_at_beta<'py>(
    py: Python<'py>,
    likelihood_logp: PyReadonlyArray1<'py, f64>,
    old_beta: f64,
    beta: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, usize)> {
    let likelihood_logp = likelihood_logp.as_slice().map_err(|_| {
        PyValueError::new_err("likelihood_logp must be a C-contiguous float64 vector")
    })?;
    if likelihood_logp.is_empty() {
        return Err(PyValueError::new_err("likelihood_logp must not be empty"));
    }
    if likelihood_logp.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("likelihood_logp must be finite"));
    }
    if !old_beta.is_finite() || !(0.0..1.0).contains(&old_beta) {
        return Err(PyValueError::new_err(
            "old_beta must be finite and in [0, 1)",
        ));
    }
    if !beta.is_finite() || !(old_beta..=2.0).contains(&beta) {
        return Err(PyValueError::new_err(
            "beta must be finite and in [old_beta, 2]",
        ));
    }
    let stats = temperature::weights_at_beta(likelihood_logp, old_beta, beta)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        stats.log_weights.into_pyarray(py),
        stats.log_normalizer,
        stats.ess,
    ))
}

/// Guided exercise: return (beta, weights, log marginal likelihood increment).
///
/// Inputs must be a nonempty C-contiguous float64 vector of finite log
/// likelihoods, old_beta in [0, 1), and threshold in [0, 1]. The input is
/// borrowed while holding the GIL; do not concurrently mutate an alias.
#[pyfunction]
fn update_beta_and_weights<'py>(
    py: Python<'py>,
    likelihood_logp: PyReadonlyArray1<'py, f64>,
    old_beta: f64,
    threshold: f64,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, f64)> {
    let likelihood_logp = likelihood_logp.as_slice().map_err(|_| {
        PyValueError::new_err("likelihood_logp must be a C-contiguous float64 vector")
    })?;
    let result = temperature::update_beta_and_weights(likelihood_logp, old_beta, threshold)
        .map_err(|error| match error {
            temperature::TemperatureError::ExerciseIncomplete(_) => {
                PyNotImplementedError::new_err(error.to_string())
            }
            _ => PyValueError::new_err(error.to_string()),
        })?;
    Ok((
        result.beta,
        result.weights.into_pyarray(py),
        result.log_marginal_likelihood_increment,
    ))
}

/// Initial bindings require a GIL-enabled interpreter.
#[pymodule(gil_used = true)]
mod pymc_rust {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::logsumexp;

    #[pymodule_export]
    use super::systematic_resampling;

    #[pymodule_export]
    use super::weights_at_beta;

    #[pymodule_export]
    use super::update_beta_and_weights;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        module.add("__debug_assertions__", cfg!(debug_assertions))?;
        Ok(())
    }
}
