//! # qp-core
//!
//! Shared numerical core of the four Rust engines in this repository
//! (`eq-options-engine`, `fx-options-engine`, `eq-var-engine`,
//! `fx-var-engine`). It depends on `std` only.
//!
//! The engines re-export what they use from here, so their public paths
//! did not change. Each piece was moved, not rewritten: the arithmetic is
//! in the same order as before and every engine golden and bit-equality
//! test passes unchanged. Where two engines had drifted apart (a
//! coefficient literal, a density written as a divide or a multiply, two
//! xoshiro scramblers), both variants are here under separate names and
//! each engine keeps the one it had.
//!
//! | Module | Contents | Used by |
//! |---|---|---|
//! | [`rng`] | SplitMix64, xoshiro256++ with cached Box-Muller, xoshiro256**, uniform maps, Marsaglia-Tsang gamma loop | all four |
//! | [`special`] | Cody erf/erfc (two tables), normal CDF/PDF, Acklam inverse normal, Halley step | all four |
//! | [`iv`] | Normalised-Black implied-vol core, generic over the erf table | both options engines |
//! | [`roots`] | Brent's method with scipy `brentq` semantics | FX options |
//! | [`linalg`] | Row-major Cholesky kernel | both VaR engines |
//! | [`par`] | Fixed-chunk scoped-thread helper; results do not depend on the thread count | FX VaR |
//! | [`pool`] | Persistent worker pool for `'static` jobs, results in index order | new, opt-in |
//! | [`philox`] | Philox4x32-10 counter-based generator (Random123 known answers) | new, opt-in |
//! | [`quantile`] | O(n) selection and type-7 quantile, same bits as a sort | new, opt-in |
//! | [`gemm`] | Cache-blocked GEMM and `D^T D`, same bits as the naive loops | eq VaR covariance |
//! | [`holidays`] | Rule-based settlement calendars (2000-2099), FX spot dates, spot-date forwards, NDF value | new, opt-in |
//! | [`rates_vol`] | SABR implied vol (Hagan, Obloj, shifted) and the Hull-White trinomial Bermudan | new, opt-in |
//! | [`exotics`] | Closed-form barriers, touches, double barriers, Asians, eSSVI/SVI slices; twin of Python `qp_core.exotics` | new, opt-in |
//! | [`vmath`] | Branch-free batch exp/erfc/normal CDF with documented tolerances | new, opt-in |
//! | [`Error`] | Invalid-input / numerical-failure error shared by the fallible functions here | all four (via `From`) |
//!
//! What stayed in the engines: each VaR engine's special functions (the
//! two use different erfc and incomplete-gamma algorithms), the PSD repair
//! code (it is written against each engine's own `Matrix` type), and the
//! engine error enums (their variant names differ and downstream code may
//! implement traits on each separately).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;

pub mod bench;
pub mod black_scholes;
pub mod exotics;
pub mod gemm;
pub mod holidays;
pub mod iv;
pub mod linalg;
pub mod par;
pub mod philox;
pub mod pool;
pub mod quantile;
pub mod rates_vol;
pub mod rng;
pub mod roots;
pub mod special;
pub mod vmath;

/// Error returned by the fallible functions in this crate.
///
/// The `Display` text matches the engines' error enums (`invalid input:
/// ...`, `numerical failure: ...`), and each engine converts it with
/// `From`, so a message that crosses the boundary reads the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Invalid caller input; Python's `ValueError`.
    InvalidInput(String),
    /// Numerical failure on well-formed input.
    Numerical(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::Numerical(msg) => write!(f, "numerical failure: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// `Ok(())` when `value` is finite, otherwise
/// `Error::InvalidInput("{name} must be finite, got {value}")`.
///
/// # Examples
///
/// ```
/// assert!(qp_core::require_finite(1.0, "S").is_ok());
/// let e = qp_core::require_finite(f64::NAN, "S").unwrap_err();
/// assert_eq!(e.to_string(), "invalid input: S must be finite, got NaN");
/// ```
#[inline]
pub fn require_finite(value: f64, name: &str) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )))
    }
}
