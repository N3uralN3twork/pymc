//! Sequential systematic resampling with the same boundary convention as PyMC.

use std::error::Error;
use std::fmt;

/// Invalid inputs are errors rather than native panics or silent normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingError {
    EmptyWeights,
    InvalidOffset,
    InvalidWeight,
    NotNormalized,
    InsufficientMass,
}

impl fmt::Display for ResamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyWeights => "weights must not be empty",
            Self::InvalidOffset => "offset must be finite and in [0, 1)",
            Self::InvalidWeight => "weights must be finite and nonnegative",
            Self::NotNormalized => "weights must sum to 1 within absolute tolerance 1e-8",
            Self::InsufficientMass => "cumulative weights do not reach the final sampling position",
        };
        formatter.write_str(message)
    }
}

impl Error for ResamplingError {}

/// Return one index per particle using positions `(offset + i) / n`.
///
/// Equality selects the first cumulative weight equal to a position, matching
/// PyMC's `while position > cumulative_weight`. Do not replace this with `>=`.
/// Summation is sequential and inputs are not renormalized. Validation and
/// output allocation are part of this public operation and its measured cost.
pub fn systematic_resampling(weights: &[f64], offset: f64) -> Result<Vec<isize>, ResamplingError> {
    if weights.is_empty() {
        return Err(ResamplingError::EmptyWeights);
    }
    if !offset.is_finite() || !(0.0..1.0).contains(&offset) {
        return Err(ResamplingError::InvalidOffset);
    }
    let mut total = 0.0;
    for &weight in weights {
        if !weight.is_finite() || weight < 0.0 {
            return Err(ResamplingError::InvalidWeight);
        }
        total += weight;
    }
    if (total - 1.0).abs() > 1e-8 {
        return Err(ResamplingError::NotNormalized);
    }

    let count = weights.len();
    let mut indices = Vec::with_capacity(count);
    let mut index = 0;
    let mut cumulative_weight = weights[0];
    for particle in 0..count {
        let position = (offset + particle as f64) / count as f64;
        while position > cumulative_weight {
            index += 1;
            cumulative_weight += weights
                .get(index)
                .ok_or(ResamplingError::InsufficientMass)?;
        }
        indices.push(index as isize);
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::{ResamplingError, systematic_resampling};

    #[test]
    fn uniform_particles_keep_their_indices() {
        assert_eq!(
            systematic_resampling(&[0.25; 4], 0.5).unwrap(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn equality_and_zero_weights_match_pymc() {
        assert_eq!(
            systematic_resampling(&[0.25; 4], 0.0).unwrap(),
            vec![0, 0, 1, 2]
        );
        assert_eq!(
            systematic_resampling(&[0.0, 1.0, 0.0], 0.0).unwrap(),
            vec![0, 1, 1]
        );
    }

    #[test]
    fn concentrated_weights_select_the_mass() {
        assert_eq!(
            systematic_resampling(&[0.0, 0.0, 1.0], 0.5).unwrap(),
            vec![2, 2, 2]
        );
    }

    #[test]
    fn invalid_inputs_return_errors() {
        assert_eq!(
            systematic_resampling(&[], 0.5),
            Err(ResamplingError::EmptyWeights)
        );
        for offset in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                systematic_resampling(&[1.0], offset),
                Err(ResamplingError::InvalidOffset)
            );
        }
        for weights in [[-1.0, 2.0], [f64::NAN, 1.0], [f64::INFINITY, 1.0]] {
            assert_eq!(
                systematic_resampling(&weights, 0.5),
                Err(ResamplingError::InvalidWeight)
            );
        }
        assert_eq!(
            systematic_resampling(&[0.2, 0.2], 0.5),
            Err(ResamplingError::NotNormalized)
        );
    }

    #[test]
    fn insufficient_cumulative_mass_is_an_error_not_a_panic() {
        assert_eq!(
            systematic_resampling(&[0.5, 0.499999999], 0.99999999999),
            Err(ResamplingError::InsufficientMass)
        );
    }
}
