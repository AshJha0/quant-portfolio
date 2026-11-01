//! # eq_var_engine: equity market-risk engine (VaR / ES / backtesting)
//!
//! Linear-book port of the Python research library
//! `python/equity/03-var-es-engine` (package `eq_var`), alongside the C++
//! engine `cpp/equity-var-engine` (namespace `eqvar`). The Rust engine
//! revalues books linearly (dollar exposures times factor returns); the
//! Python library also does full revaluation. The shared conventions are
//! checked against the same golden vectors (see
//! `tests/test_cross_language.rs` and `docs/VALIDATION.md`).
//!
//! ## Pipeline
//!
//! factor returns → covariance ([`matrix`]) → historical / parametric /
//! Monte Carlo VaR ([`historical`], [`parametric`], [`monte_carlo`]) →
//! Expected Shortfall ([`expected_shortfall`]) → Kupiec / Christoffersen /
//! Basel traffic-light backtests ([`backtest`]).
//!
//! ## Conventions
//!
//! * P&L arrays are in currency units with **loss < 0**.
//! * `alpha` is the **tail probability**: `alpha = 0.01` → 99 % VaR.
//! * VaR and ES are reported as **positive** numbers for a loss:
//!   `VaR_alpha = -Q_alpha(pnl)`.
//!   Every public function rejects `alpha` outside `(0, 0.5)`.
//! * Empirical VaR is the order statistic `L(m)`, `m = ceil(n alpha - 1e-9)`,
//!   of the losses sorted descending; empirical ES is the Acerbi-Tasche
//!   estimator with the exact fractional boundary weight ([`tail`]). There
//!   is no interpolation on any VaR path.
//! * All stochastic components take an explicit `u64` seed and repeat
//!   exactly for a fixed seed on the same build and platform ([`rng`]).
//!   Results are not claimed to match bit for bit across platforms, since
//!   `ln`, `sin` and `cos` come from the platform libm.
//!
//! ## Design
//!
//! * **Zero external dependencies**: `std` only. The deterministic RNG,
//!   the special functions (`erfc`, inverse normal CDF, regularized
//!   incomplete beta/gamma) and the dense linear algebra are implemented
//!   in-crate and validated to documented tolerances ([`stats`]).
//! * **`Result` everywhere**: invalid inputs return
//!   [`EqVarError::InvalidInput`] instead of panicking; numerical failures
//!   (e.g. an indefinite covariance that defeats the Cholesky jitter
//!   fallback) return [`EqVarError::Numerical`]. Library code never panics
//!   on user input.
//!
//! ## Quickstart
//!
//! ```
//! use eq_var_engine::prelude::*;
//!
//! // Deterministic 3-factor return panel (60 days).
//! let mut ret = Matrix::zeros(60, 3);
//! for t in 0..60 {
//!     for j in 0..3 {
//!         let (tf, jf) = (t as f64, j as f64);
//!         ret.set(t, j, 0.01 * (tf + jf).sin() + 0.005 * (2.0 * tf - jf).cos());
//!     }
//! }
//! let cov = sample_covariance(&ret).unwrap();
//! let exposures = [2.0e6, -1.0e6, 5.0e5];
//!
//! let var99 = parametric_var(&exposures, &cov, 0.01, TailModel::Normal).unwrap();
//! let es975 = parametric_es(&exposures, &cov, 0.025, TailModel::Normal).unwrap();
//! let mc = monte_carlo_var(&exposures, &cov, 0.01, 20_000, TailModel::Normal, 42).unwrap();
//! assert!(es975 > 0.0 && var99 > 0.0 && mc > 0.0);
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]
// `!(x > 0.0)` is the NaN-rejecting form on purpose: `x <= 0.0` is false
// for NaN and would let it through. The lint asks for the opposite.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// Index loops mirror the matrix formulas they implement.
#![allow(clippy::needless_range_loop)]

pub mod backtest;
pub mod expected_shortfall;
pub mod fast;
pub mod historical;
pub mod matrix;
pub mod monte_carlo;
pub mod parametric;
pub mod psd;
pub mod rng;
pub mod stats;
pub mod tail;

use std::fmt;

/// Errors returned by the engine.
///
/// * [`EqVarError::InvalidInput`]: the caller passed something the model
///   cannot accept (empty P&L, `alpha` outside `(0, 0.5)`, dimension
///   mismatch, `df <= 2`, …). The message states what was wrong and what
///   was received, mirroring the `ValueError`s of the Python reference.
/// * [`EqVarError::Numerical`]: the inputs were well-formed but a numerical
///   procedure failed (e.g. Cholesky on an indefinite matrix even after the
///   jitter escalation).
#[derive(Debug, Clone, PartialEq)]
pub enum EqVarError {
    /// Invalid caller input; equivalent to Python's `ValueError`.
    InvalidInput(String),
    /// Numerical failure on well-formed input.
    Numerical(String),
}

impl fmt::Display for EqVarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EqVarError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            EqVarError::Numerical(msg) => write!(f, "numerical failure: {msg}"),
        }
    }
}

impl std::error::Error for EqVarError {}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, EqVarError>;

/// Tail model for parametric and Monte Carlo VaR / ES.
///
/// `StudentT` is **variance-matched**: the t quantile / scale matrix is
/// rescaled by `sqrt((df - 2) / df)` so the portfolio sigma is identical to
/// the normal case and only the tail shape fattens (df must be > 2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TailModel {
    /// Gaussian tails.
    Normal,
    /// Variance-matched Student-t tails with `df` degrees of freedom (> 2).
    StudentT {
        /// Degrees of freedom; must exceed 2 for a finite variance.
        df: f64,
    },
}

pub(crate) fn validate_alpha(alpha: f64) -> Result<()> {
    tail::validate_alpha(alpha).map(|_| ())
}

/// Convenience re-exports of the full public API.
pub mod prelude {
    pub use crate::backtest::{
        acerbi_szekely_z2, basel_traffic_light, christoffersen_cc, christoffersen_independence,
        exceptions_from_pnl, kupiec_pof, BaselInput, BaselZone, ChristoffersenResult,
        ConditionalCoverageResult, KupiecInput, KupiecResult, TrafficLightResult,
    };
    pub use crate::expected_shortfall::{
        expected_shortfall, normal_es, parametric_es, parametric_es_full, student_t_es,
    };
    pub use crate::fast::{
        empirical_var_es_ws, expected_shortfall_ws, historical_var_ws, parametric_es_fast,
        parametric_var_fast, portfolio_sigma_fast, var_bootstrap_se_ws, FastResult, McStats,
        McWorkspace, VarFault,
    };
    pub use crate::historical::{
        age_weighted_var, brw_weights, ewma_volatility, ewma_volatility_forecast,
        fhs_factor_scenarios, filtered_historical_var_factors, historical_var,
        overlapping_horizon_pnl, scale_var_sqrt_time, MIN_OBS,
    };
    pub use crate::matrix::{covariance_from_vols, ewma_covariance, sample_covariance, Matrix};
    pub use crate::monte_carlo::{
        monte_carlo_es, monte_carlo_pnl, monte_carlo_run, monte_carlo_var, portfolio_pnl,
        simulate_factor_returns, simulate_factor_returns_jitter, var_bootstrap_se, var_kde_se,
        McRun,
    };
    pub use crate::parametric::{
        cornish_fisher_valid, cornish_fisher_var, cornish_fisher_z, parametric_var,
        parametric_var_full, portfolio_sigma,
    };
    pub use crate::psd::{
        cholesky_repair, clip_psd, higham_nearest, symmetric_eigen, CholeskyFactor, PsdRepair,
    };
    pub use crate::rng::Rng;
    pub use crate::stats::{
        binomial_cdf, chi2_sf, erfc, excess_kurtosis, mean, normal_cdf, normal_pdf, normal_ppf,
        regularized_beta, regularized_gamma_p, regularized_gamma_q, skewness, stdev, student_t_cdf,
        student_t_pdf, student_t_ppf,
    };
    pub use crate::tail::{empirical_var, empirical_var_es, tail_rank, weighted_var_es};
    /// Deprecated entry points, kept in the prelude for one release.
    #[allow(deprecated)]
    pub use crate::{
        backtest::{basel_traffic_light_250, kupiec_pof_positional},
        historical::{filtered_historical_var, linear_quantile},
        monte_carlo::var_order_statistic_se,
        parametric::cornish_fisher_domain_ok,
    };
    pub use crate::{EqVarError, TailModel};
}
