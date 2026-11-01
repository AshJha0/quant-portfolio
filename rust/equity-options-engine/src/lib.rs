//! # eq-options-engine
//!
//! Rust equity options pricing engine with no runtime dependencies, a port
//! of the Python reference package `eq_options`
//! (`python/equity/01-options-pricing/`) and sibling of the C++ engine
//! (`cpp/equity-options-engine/`). Same models and conventions, checked
//! against golden vectors computed in mpmath (per-output tolerances) and
//! against the Python and C++ engines on seeded random inputs.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |---|---|
//! | [`black_scholes`] | European call/put with continuous dividend yield `q`; `d1`/`d2`; erfc-based normal CDF (Cody's rational approximation); [`PricingError`] |
//! | [`greeks`] | Analytic delta/gamma/vega/theta/rho + vanna/volga; generic central finite-difference Greeks over any pricer closure |
//! | [`implied_vol`] | Out-of-the-money-side solver on normalised Black ([`iv_core`]), at most 10 evaluations; rejects quotes at or outside the bounds |
//! | [`iv_core`] | Shared implied-vol core (same algorithm and constants as Python and C++) |
//! | [`binomial`] | CRR tree, European + American, O(n) memory, exercise values precomputed per level |
//! | [`black76`] | Options on forwards/futures: price + Greeks; exactly equals BSM at `F = S e^{(r-q)T}` |
//! | [`monte_carlo`] | GBM exact scheme; antithetic pairs + control variate fitted on pair averages; explicit `u64` seed |
//! | [`rng`] | SplitMix64-seeded xoshiro256++ with Box-Muller normal sampling (std-only, deterministic) |
//! | [`golden`] | GENERATED mpmath golden vectors (do not edit; run `tools/gen_golden_rs.py`) |
//!
//! ## Conventions (identical to the Python reference)
//!
//! Continuously compounded annualised `r`/`q` (ACT/365F); `t` in years;
//! `sigma` annualised log-return vol; theta per **year**, vega per
//! **unit** of vol, rho per **unit** of rate. Invalid inputs return
//! [`PricingError`] (where Python raises `ValueError` and C++ throws
//! `std::invalid_argument`).
//!
//! ## Quickstart
//!
//! ```
//! use eq_options_engine::{
//!     bs_price, bs_greeks, implied_vol, crr_price, mc_price,
//!     Exercise, OptionType,
//! };
//!
//! let (s, k, t, r, sigma, q) = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0);
//! let ot = OptionType::Call;
//!
//! let price = bs_price(s, k, t, r, sigma, q, ot).unwrap();
//! assert!((price - 10.450583572185565).abs() < 1e-12);
//!
//! let greeks = bs_greeks(s, k, t, r, sigma, q, ot).unwrap();
//! assert!((greeks.delta - 0.6368306511756191).abs() < 1e-12);
//!
//! let iv = implied_vol(price, s, k, t, r, q, ot).unwrap();
//! assert!((iv - sigma).abs() < 1e-8);
//!
//! let amer = crr_price(s, k, t, r, sigma, q, OptionType::Put,
//!                      Exercise::American, 500).unwrap();
//! assert!(amer > 0.0);
//!
//! let mc = mc_price(s, k, t, r, sigma, q, ot, 100_000, true, true, 42).unwrap();
//! assert!(mc.contains(price) || (mc.price - price).abs() < 3.0 * mc.std_error);
//! ```

#![deny(missing_docs)]
#![warn(rust_2018_idioms)]
// Rational-approximation coefficients are kept exactly as published.
#![allow(clippy::excessive_precision)]

pub mod binomial;
pub mod black76;
pub mod black_scholes;
pub mod golden;
pub mod greeks;
pub mod implied_vol;
pub mod iv_core;
pub mod monte_carlo;
pub mod rng;

pub use binomial::{crr_price, early_exercise_premium, Exercise};
pub use black76::{b76_d1_d2, black76_greeks, black76_price, Black76Greeks};
pub use black_scholes::{
    bs_price, d1_d2, forward_price, intrinsic_value, OptionType, PricingError,
};
pub use greeks::{bs_greeks, fd_greeks, BsGreeks};
pub use implied_vol::implied_vol;
pub use monte_carlo::{mc_price, McResult};
