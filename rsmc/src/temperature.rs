//! Guided port of `SMC_KERNEL.update_beta_and_weights`.
//!
//! Complete EXERCISES 1-3 in order. The unfinished functions return an error
//! instead of panicking or pretending to produce usable sampling results.
//! See `../exercises/temperature.md` for hints and commands.

use core::f64;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemperatureError {
    EmptyLikelihoods,
    NonFiniteLikelihood,
    InvalidBeta,
    InvalidThreshold,
    ExerciseIncomplete(&'static str),
}

impl fmt::Display for TemperatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLikelihoods => formatter.write_str("likelihood_logp must not be empty"),
            Self::NonFiniteLikelihood => {
                formatter.write_str("this exercise requires finite likelihood_logp values")
            }
            Self::InvalidBeta => formatter.write_str("old_beta must be finite and in [0, 1)"),
            Self::InvalidThreshold => formatter.write_str("threshold must be finite and in [0, 1]"),
            Self::ExerciseIncomplete(step) => write!(formatter, "exercise incomplete: {step}"),
        }
    }
}

impl Error for TemperatureError {}

/// Values to apply to the Python sampler after one successful update.
#[derive(Debug, PartialEq)]
pub struct TemperatureUpdate {
    pub beta: f64,
    pub weights: Vec<f64>,
    /// Add this to the previous log marginal likelihood; it is not a total.
    pub log_marginal_likelihood_increment: f64,
}

/// Intermediate values used by the temperature search.
#[derive(Debug)]
pub struct WeightStats {
    pub log_weights: Vec<f64>,
    pub log_normalizer: f64,
    pub ess: usize,
}

/// Return `ln(sum(exp(values)))` without directly exponentiating large values.
///
/// Exercise helper: callers supply a nonempty slice of finite values.
pub fn logsumexp(values: &[f64]) -> Result<f64, TemperatureError> {
    // EXERCISE 1: use a maximum shift before exponentiating and summing.
    // Keep the Result signature: the caller uses `?` to propagate this error.
    // Replace the next two lines with your implementation.
    let maximum: f64 = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let total: f64 = values.iter().map(|x| (x - maximum).exp()).sum();

    Ok(maximum + total.ln())
}

/// Evaluate normalized log weights and integer ESS at one trial temperature.
///
/// Exercise helper: inputs are validated by `update_beta_and_weights`.
/// The search may evaluate trial temperatures above one before its final cap.
pub fn weights_at_beta(
    likelihood_logp: &[f64],
    old_beta: f64,
    beta: f64,
) -> Result<WeightStats, TemperatureError> {
    // Provided: create the unnormalized log weights and call exercise 1.
    let log_weights_un: Vec<f64> = likelihood_logp
        .iter()
        .map(|value| (beta - old_beta) * value)
        .collect();
    let log_normalizer = logsumexp(&log_weights_un)?;

    let log_weight: Vec<f64> = log_weights_un
        .iter()
        .map(|value| value - log_normalizer)
        .collect();

    let doubled_log_weights: Vec<f64> = log_weight
        .iter()
        .map(|value| 2.0 * value)
        .collect();

    let log_ess_sum = logsumexp(&doubled_log_weights)?;

    let ess = (-1.0 * log_ess_sum).exp() as usize;
    
    Ok(WeightStats {
        log_weights: log_weight,
        log_normalizer,
        ess,
    })
}



fn choose_next_beta(
    likelihood_logp: &[f64],
    old_beta: f64,
    threshold: f64,
) -> Result<f64, TemperatureError> {
    // EXERCISE 3: port the reference bisection using weights_at_beta(...)? .
    // Start at [old_beta, 2.0]; target = trunc(particle_count * threshold).
    // Stop when ESS equals target or the bracket width is <= 1e-6.
    // Return the last trial beta capped at 1.0. See the guide for branch hints.
    // Replace the next two lines with your implementation.
    let _ = (likelihood_logp, old_beta, threshold);
    Err(TemperatureError::ExerciseIncomplete(
        "3: temperature search",
    ))
}

/// Compute the next SMC temperature, weights, and evidence increment.
///
/// This starter supports nonempty finite float64 likelihoods, old_beta in
/// [0, 1), and threshold in [0, 1]. It does not mutate inputs or sampler state.
/// Non-finite likelihoods (including -infinity) are outside this exercise's
/// scope. Supporting them requires a separate numerical contract.
pub fn update_beta_and_weights(
    likelihood_logp: &[f64],
    old_beta: f64,
    threshold: f64,
) -> Result<TemperatureUpdate, TemperatureError> {
    if likelihood_logp.is_empty() {
        return Err(TemperatureError::EmptyLikelihoods);
    }
    if likelihood_logp.iter().any(|value| !value.is_finite()) {
        return Err(TemperatureError::NonFiniteLikelihood);
    }
    if !old_beta.is_finite() || !(0.0..1.0).contains(&old_beta) {
        return Err(TemperatureError::InvalidBeta);
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(TemperatureError::InvalidThreshold);
    }

    let beta = choose_next_beta(likelihood_logp, old_beta, threshold)?;
    // Recompute at the final capped beta, just as the reference does.
    let stats = weights_at_beta(likelihood_logp, old_beta, beta)?;
    let mut weights: Vec<f64> = stats.log_weights.iter().map(|value| value.exp()).collect();
    // Preserve PyMC's extra normalization to correct accumulated roundoff.
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }

    Ok(TemperatureUpdate {
        beta,
        weights,
        log_marginal_likelihood_increment: stats.log_normalizer
            - (likelihood_logp.len() as f64).ln(),
    })
}

#[cfg(test)]
mod tests {
    use super::{TemperatureError, logsumexp, update_beta_and_weights, weights_at_beta};

    #[test]
    fn invalid_inputs_are_rejected_before_the_exercises() {
        assert_eq!(
            update_beta_and_weights(&[], 0.0, 0.5),
            Err(TemperatureError::EmptyLikelihoods)
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                update_beta_and_weights(&[value], 0.0, 0.5),
                Err(TemperatureError::NonFiniteLikelihood)
            );
        }
        for beta in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                update_beta_and_weights(&[0.0], beta, 0.5),
                Err(TemperatureError::InvalidBeta)
            );
        }
        for threshold in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                update_beta_and_weights(&[0.0], 0.0, threshold),
                Err(TemperatureError::InvalidThreshold)
            );
        }
    }

    #[test]
    #[ignore = "learning checkpoint: complete EXERCISE 1, then run with --ignored"]
    fn exercise_1_stable_logsumexp() {
        for offset in [0.0, 1000.0, -1000.0] {
            let actual = logsumexp(&[offset, offset]).expect("complete exercise 1");
            assert!((actual - (offset + 2.0_f64.ln())).abs() < 1e-12);
        }
        assert_eq!(logsumexp(&[-3.0]).unwrap(), -3.0);
    }

    #[test]
    #[ignore = "learning checkpoint: complete EXERCISES 1-2, then run with --ignored"]
    fn exercise_2_log_weights_and_ess() {
        let likelihoods = [1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln(), 4.0_f64.ln()];
        let stats = weights_at_beta(&likelihoods, 0.0, 1.0).expect("complete exercises 1-2");
        for (log_weight, expected) in stats.log_weights.iter().zip([0.1, 0.2, 0.3, 0.4]) {
            assert!((log_weight.exp() - expected).abs() < 1e-12);
        }
        assert!((stats.log_normalizer - 10.0_f64.ln()).abs() < 1e-12);
        assert_eq!(stats.ess, 3); // ESS is about 3.33; PyMC truncates to an integer.
    }

    #[test]
    #[ignore = "learning checkpoint: complete EXERCISES 1-3, then run with --ignored"]
    fn exercise_3_temperature_search() {
        let flat = update_beta_and_weights(&[0.0; 8], 0.0, 0.5).expect("complete exercises 1-3");
        assert_eq!(flat.beta, 1.0);
        for weight in flat.weights {
            assert!((weight - 0.125).abs() < 1e-12);
        }
        assert!(flat.log_marginal_likelihood_increment.abs() < 1e-12);

        let result = update_beta_and_weights(
            &[-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0],
            0.0,
            0.5,
        )
        .unwrap();
        // At beta=0.125, integer ESS is 4, exactly the reference target.
        assert_eq!(result.beta, 0.125);
        assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
