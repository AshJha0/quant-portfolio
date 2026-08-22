//! # fx_var_engine — multi-currency FX market-risk engine (VaR / ES / stress)
//!
//! Rust twin of the Python research package `python/fx/03-var-es-engine`
//! (package `fx_var`) and the C++ engine `cpp/fx-var-engine` (namespace
//! `fxvar`). All three stacks share the same semantics and are
//! cross-validated against the same golden vectors (see
//! `tests/test_golden_python.rs` and `docs/VALIDATION.md`).
//!
//! ## Pipeline
//!
//! multi-currency [`book`] (spot / forward / cash, USD-triangulated,
//! CIP-consistent forwards) → factor-return history ([`returns`]) →
//! historical / parametric / Monte Carlo VaR ([`historical`],
//! [`parametric`], [`monte_carlo`]) → [`expected_shortfall`] → canned and
//! reverse [`stress`] scenarios → Kupiec / Christoffersen / Basel
//! [`backtest`].
//!
//! ## FX conventions (see `CONVENTIONS.md` at the portfolio root)
//!
//! * Pairs are quoted BASE/QUOTE: `EURUSD` = USD per 1 EUR.
//! * Every currency is mapped to a single **USD factor**: `"FX:CCY"` is the
//!   daily *log return* of the USD price of 1 unit of CCY (the log return
//!   of CCYUSD). USD itself has no FX factor — its USD price is identically
//!   1. Cross pairs (EURJPY) are *triangulated* through their two USD legs,
//!   so the factor set is arbitrage-consistent by construction — there is
//!   no separate cross factor.
//! * `"IR:CCY"` is an *absolute* shock (decimal p.a.) to the continuously
//!   compounded, ACT/365 zero rate of CCY (flat curve per currency).
//! * CIP forward: `F = S * exp((r_d - r_f) * T)` with `r_d` the
//!   quote-currency (domestic) rate and `r_f` the base-currency (foreign)
//!   rate.
//! * P&L arrays are profit (+) / loss (-) in the book's base currency; VaR
//!   and ES are reported as **positive** loss numbers.
//! * `alpha` is a **confidence level** (`0.99` = 99 % VaR), matching the
//!   C++/Python FX engines — note this is the opposite convention from the
//!   sibling equity engines, which use `alpha` as a tail probability.
//!
//! ## Design
//!
//! * **Zero external dependencies** — `std` only. The deterministic RNG
//!   ([`rng`]), special functions and dense linear algebra ([`matrix`],
//!   [`stats`]) are implemented in-crate and validated to documented
//!   tolerances.
//! * **`Result` everywhere** — invalid caller input returns
//!   [`FxVarError::Invalid`]; well-formed input that defeats a numerical
//!   procedure (e.g. a covariance that resists Cholesky even with jitter)
//!   returns [`FxVarError::Numerical`]. Library code never panics on user
//!   input.
//! * **NaN policy: refuse** — every history/backtest input that could
//!   silently corrupt a quantile is checked for NaN and rejected with an
//!   informative message rather than dropped or imputed.
//!
//! ## Quickstart
//!
//! ```
//! use fx_var_engine::prelude::*;
//!
//! // Market: EURUSD = 1.10, USDJPY spot implied by JPY = 0.0090 USD.
//! let market = Market::new(
//!     [("EUR", 1.10), ("JPY", 0.0090)],
//!     [("USD", 0.050), ("EUR", 0.030), ("JPY", 0.001)],
//! ).unwrap();
//!
//! // Long 10m EURUSD plus a 6-month ATM USDJPY forward.
//! let book = Book::new(
//!     vec![
//!         Position::Spot(SpotPosition::new("EURUSD", 10.0e6, None)),
//!         Position::Forward(ForwardPosition::new("USDJPY", 5.0e6, 0.5, None)),
//!     ],
//!     "USD",
//! ).unwrap();
//!
//! // 300-day deterministic synthetic factor history.
//! let compiled = CompiledBook::new(&book, &market).unwrap();
//! let factors = compiled.factors().to_vec();
//! let n = 300usize;
//! let mut data = Matrix::zeros(n, factors.len());
//! for t in 0..n {
//!     for (j, f) in factors.iter().enumerate() {
//!         let scale = if f.starts_with("FX:") { 0.006 } else { 0.0004 };
//!         data.set(t, j, scale * (0.1 * t as f64 + j as f64).sin());
//!     }
//! }
//! let returns = ReturnsMatrix::new(factors, data).unwrap();
//!
//! let hs = historical_var(&book, &market, &returns, &HistoricalOptions::default()).unwrap();
//! let pv = parametric_var(&book, &market, &returns, &ParametricOptions::default()).unwrap();
//! assert!(hs.var > 0.0 && pv.var > 0.0);
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod backtest;
pub mod book;
pub mod expected_shortfall;
pub mod historical;
pub mod matrix;
pub mod monte_carlo;
pub mod parametric;
pub mod returns;
pub mod rng;
pub mod stats;
pub mod stress;

use std::fmt;

/// Errors returned by the engine.
///
/// * [`FxVarError::Invalid`] — the caller passed something the model
///   cannot accept (empty book, `alpha` outside `(0, 1)`, dimension
///   mismatch, missing factor columns, NaNs, …). The message states what
///   was wrong, mirroring the `ValueError`s of the Python reference and the
///   `std::invalid_argument`s of the C++ engine.
/// * [`FxVarError::Numerical`] — the inputs were well-formed but a
///   numerical procedure failed (e.g. Cholesky on an indefinite matrix even
///   after the jitter escalation).
#[derive(Debug, Clone, PartialEq)]
pub enum FxVarError {
    /// Invalid caller input; equivalent to Python's `ValueError` / C++'s
    /// `std::invalid_argument`.
    Invalid(String),
    /// Numerical failure on well-formed input.
    Numerical(String),
}

impl FxVarError {
    /// Build an [`FxVarError::Invalid`] from any message-like value.
    pub fn invalid(msg: impl Into<String>) -> Self {
        FxVarError::Invalid(msg.into())
    }

    /// Build an [`FxVarError::Numerical`] from any message-like value.
    pub fn numerical(msg: impl Into<String>) -> Self {
        FxVarError::Numerical(msg.into())
    }
}

impl fmt::Display for FxVarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FxVarError::Invalid(msg) => write!(f, "invalid input: {msg}"),
            FxVarError::Numerical(msg) => write!(f, "numerical failure: {msg}"),
        }
    }
}

impl std::error::Error for FxVarError {}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, FxVarError>;

/// Tail distribution for parametric VaR/ES ([`parametric`],
/// [`expected_shortfall`]). Mirrors C++ `fxvar::TailDist`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailDist {
    /// Gaussian tails (RiskMetrics classic).
    Normal,
    /// Standardised (unit-variance) Student-t tails — fatter than normal at
    /// equal sigma; the `df` parameter is supplied where used (must be > 2
    /// for finite variance).
    StudentT,
}

/// Simulation distribution for Monte Carlo VaR ([`monte_carlo`]). Mirrors
/// C++ `fxvar::McDist`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McDist {
    /// Multivariate normal factor returns.
    Normal,
    /// Covariance-matched multivariate Student-t (fat tails at equal
    /// sigma).
    StudentT,
    /// Normal diffusion plus a Bernoulli common-jump overlay — the
    /// peg-break / devaluation stress-in-a-distribution.
    Jump,
}

/// Convenience re-exports of the full public API.
pub mod prelude {
    pub use crate::backtest::{
        basel_traffic_light, christoffersen_independence, conditional_coverage,
        evaluate_var_backtest, kupiec_pof, BacktestResult, LrTest, TrafficLight, Zone,
    };
    pub use crate::book::{
        fx_factor, ir_factor, split_pair, Book, CashPosition, CompiledBook, ForwardPosition,
        Market, PairLegs, Position, SpotPosition,
    };
    pub use crate::expected_shortfall::{
        empirical_es, empirical_var, empirical_var_es, normal_es, normal_var, student_t_es,
        student_t_var,
    };
    pub use crate::historical::{historical_var, HistoricalOptions, HistoricalResult, HsMethod};
    pub use crate::matrix::Matrix;
    pub use crate::monte_carlo::{
        monte_carlo_var, simulate_factor_returns, var_standard_error, var_standard_error_bootstrap,
        JumpSpec, MonteCarloOptions, MonteCarloResult,
    };
    pub use crate::parametric::{
        cornish_fisher_domain_ok, cornish_fisher_var, cornish_fisher_z, parametric_var,
        portfolio_sigma, var_covar, CovMethod, ParametricOptions, ParametricResult, VarEs,
    };
    pub use crate::returns::{
        ewma_cov, ewma_volatility, flag_peg_factors, sample_cov, validate_returns, EwmaVolatility,
        FactorCov, ReturnsMatrix, PEG_VOL_THRESHOLD, TRADING_DAYS_PER_YEAR,
    };
    pub use crate::rng::Rng;
    pub use crate::stats::{
        binom_cdf, chi2_sf, ln_gamma, moments, norm_cdf, norm_pdf, inv_norm_cdf, reg_inc_beta,
        reg_lower_gamma, reg_upper_gamma, sample_std, student_t_cdf, student_t_pdf,
        student_t_quantile, validate_alpha, validate_horizon, Moments,
    };
    pub use crate::stress::{
        historical_scenarios, peg_break_scenario, reverse_stress_for_loss,
        reverse_stress_linear, reverse_stress_numerical, run_stress, simple_to_log,
        usd_broad_move, ReverseStress, Scenario, StressRow,
    };
    pub use crate::{FxVarError, McDist, TailDist};
}
