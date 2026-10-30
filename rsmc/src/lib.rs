//! Experimental computation kernels, with optional Python bindings.
//!
//! Keep numeric algorithms independent of Python so Rust-only tests do not need
//! an interpreter. Maturin enables the `python` feature for extension builds.

pub mod resampling;
pub mod temperature;

#[cfg(feature = "python")]
mod python;
