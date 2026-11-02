//! Black-Scholes-Merton pricing for European equity options.
//!
//! # Conventions
//!
//! * Rates `r` and dividend yields `q` are continuously compounded,
//!   annualised (ACT/365F).
//! * `t` is the time to expiry in years; `sigma` is the annualised
//!   volatility of log-returns.
//! * Prices are in the same currency units as `s` and `k`.
//!
//! # Edge-case policy (documented + unit tested, identical to the Python
//! reference `eq_options.black_scholes`)
//!
//! * `t == 0`     -> intrinsic value `max(S - K, 0)` / `max(K - S, 0)`.
//! * `sigma == 0` -> discounted intrinsic on the forward,
//!   `exp(-rT) * max(±(F - K), 0)` with `F = S exp((r - q) T)`.
//! * `k == 0`     -> call is a forward on the stock, `S exp(-qT)`; put is 0.
//! * `s == 0`     -> call is 0; put is `K exp(-rT)`.
//! * Negative `s`, `k`, `t` or `sigma`, and any non-finite input
//!   (NaN or infinity, including in `r`/`q`), return
//!   [`PricingError::InvalidInput`]. Negative `r` and `q` are fully
//!   supported.

use std::fmt;

use crate::fast::ErrorCode;
use qp_core::black_scholes::{carry_d1_d2, carry_value, ErfcNormal};

/// Option payoff direction.
///
/// # Examples
///
/// ```
/// use eq_options_engine::OptionType;
/// assert_eq!(OptionType::Call.sign(), 1.0);
/// assert_eq!(OptionType::Put.sign(), -1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    /// Right to buy at the strike: payoff `max(S - K, 0)`.
    Call,
    /// Right to sell at the strike: payoff `max(K - S, 0)`.
    Put,
}

impl OptionType {
    /// Payoff sign: `+1.0` for calls, `-1.0` for puts.
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptionType::Call => write!(f, "call"),
            OptionType::Put => write!(f, "put"),
        }
    }
}

/// Errors returned by the pricing and calibration routines.
///
/// This mirrors the Python reference, where the same conditions raise
/// `ValueError` (and the C++ engine, where they throw
/// `std::invalid_argument`): Rust surfaces them as a `Result` instead of
/// unwinding.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, OptionType, PricingError};
/// let err = bs_price(-1.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap_err();
/// assert!(matches!(err, PricingError::InvalidInput(_)));
/// assert!(err.to_string().contains("S"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PricingError {
    /// An input violated its domain (negative price/vol/expiry, NaN, ...).
    InvalidInput(String),
    /// An iterative solver failed to converge or to bracket a root.
    NoConvergence(String),
    /// A quoted premium sits outside the static no-arbitrage bounds.
    ArbitrageBound(String),
}

impl fmt::Display for PricingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PricingError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            PricingError::NoConvergence(msg) => write!(f, "no convergence: {msg}"),
            PricingError::ArbitrageBound(msg) => write!(f, "arbitrage bound violated: {msg}"),
        }
    }
}

impl std::error::Error for PricingError {}

/// Validate common Black-Scholes inputs.
///
/// Requires `s >= 0`, `k >= 0`, `t >= 0`, `sigma >= 0`, all finite (no
/// NaN, no infinity); negative `r`/`q` are allowed (and therefore not
/// checked here: see [`validate_rates`] for the finiteness check on
/// rates).
///
/// All non-finite inputs are rejected, as in the Python and C++ engines,
/// so garbage cannot propagate silently into a risk number.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] naming the offending parameter.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::validate_inputs;
/// assert!(validate_inputs(100.0, 100.0, 1.0, 0.2).is_ok());
/// assert!(validate_inputs(100.0, 100.0, -1.0, 0.2).is_err());
/// assert!(validate_inputs(f64::NAN, 100.0, 1.0, 0.2).is_err());
/// assert!(validate_inputs(f64::INFINITY, 100.0, 1.0, 0.2).is_err());
/// ```
pub fn validate_inputs(s: f64, k: f64, t: f64, sigma: f64) -> Result<(), PricingError> {
    for (name, value) in [("S", s), ("K", k), ("T", t), ("sigma", sigma)] {
        if value.is_nan() {
            return Err(PricingError::InvalidInput(format!(
                "{name} must not be NaN"
            )));
        }
        if value.is_infinite() {
            return Err(PricingError::InvalidInput(format!(
                "{name} must be finite, got {value}"
            )));
        }
        if value < 0.0 {
            return Err(PricingError::InvalidInput(format!(
                "{name} must be >= 0, got {value}"
            )));
        }
    }
    Ok(())
}

/// Validate that the rate inputs `r` and `q` are finite.
///
/// Negative rates are fully supported (and common post-2015), so only
/// NaN and infinity are rejected. Called by every public pricing entry
/// point so that a NaN rate can never silently produce a NaN price.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] naming the offending parameter.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::validate_rates;
/// assert!(validate_rates(-0.01, 0.02).is_ok());
/// assert!(validate_rates(f64::NAN, 0.0).is_err());
/// assert!(validate_rates(0.05, f64::INFINITY).is_err());
/// ```
pub fn validate_rates(r: f64, q: f64) -> Result<(), PricingError> {
    for (name, value) in [("r", r), ("q", q)] {
        if !value.is_finite() {
            return Err(PricingError::InvalidInput(format!(
                "{name} must be finite, got {value}"
            )));
        }
    }
    Ok(())
}

/// Complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Rust's `std` has no `erfc`, so this is W. J. Cody's rational Chebyshev
/// approximation (SIAM 1969; the netlib `CALERF` algorithm; glibc and most
/// libms use a different, fdlibm-derived approximation): three regimes (`|x| <= 0.46875`,
/// `0.46875 < |x| <= 4`, `|x| > 4`) with the `exp(-x^2)` factor split as
/// `exp(-y16^2) exp(-(x-y16)(x+y16))` for tail accuracy. Verified against
/// reference values to better than `2e-15` relative error over
/// `[-27, 27]` (see the unit tests), comfortably inside the 1e-12
/// requirement for pricing.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::erfc;
/// assert!((erfc(0.0) - 1.0).abs() < 1e-15);
/// assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-14);
/// assert!((erfc(-1.0) - 1.8427007929497148).abs() < 1e-14);
/// ```
#[inline]
pub fn erfc(x: f64) -> f64 {
    qp_core::special::erfc(x)
}

/// Error function `erf(x)`.
///
/// For `|x| <= 0.46875` this is Cody's direct rational approximation
/// `x P(x^2)/Q(x^2)` (full relative precision near zero, unlike
/// `1 - erfc(x)`); beyond that `1 - erfc(x)` loses nothing because
/// `erfc(x) <= 0.51`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::erf;
/// assert!((erf(1e-10) - 1.1283791670955126e-10).abs() < 1e-25);
/// assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-15);
/// ```
#[inline]
pub fn erf(x: f64) -> f64 {
    qp_core::special::erf(x)
}

/// Standard normal cumulative distribution function `Phi(x)`.
///
/// Computed as `0.5 * erfc(-x / sqrt(2))`: the erfc route keeps full
/// relative accuracy deep in the lower tail, where the naive
/// `0.5 (1 + erf)` form suffers catastrophic cancellation.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::norm_cdf;
/// assert!((norm_cdf(0.0) - 0.5).abs() < 1e-15);
/// assert!((norm_cdf(1.959963984540054) - 0.975).abs() < 1e-12);
/// assert!(norm_cdf(-37.0) > 0.0); // keeps tail mass until ~ -37.5
/// ```
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    qp_core::special::norm_cdf(x)
}

/// Standard normal probability density function `phi(x)`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::norm_pdf;
/// assert!((norm_pdf(0.0) - 0.3989422804014327).abs() < 1e-15);
/// ```
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    qp_core::special::norm_pdf_div(x)
}

/// Intrinsic (exercise-now) value `max(S - K, 0)` or `max(K - S, 0)`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{black_scholes::intrinsic_value, OptionType};
/// assert_eq!(intrinsic_value(105.0, 100.0, OptionType::Call), 5.0);
/// assert_eq!(intrinsic_value(105.0, 100.0, OptionType::Put), 0.0);
/// ```
#[inline]
pub fn intrinsic_value(s: f64, k: f64, option_type: OptionType) -> f64 {
    (option_type.sign() * (s - k)).max(0.0)
}

/// Equity forward `F = S * exp((r - q) * T)`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::forward_price;
/// let f = forward_price(100.0, 1.0, 0.05, 0.02);
/// assert!((f - 100.0 * (0.03f64).exp()).abs() < 1e-12);
/// ```
#[inline]
pub fn forward_price(s: f64, t: f64, r: f64, q: f64) -> f64 {
    s * ((r - q) * t).exp()
}

/// Black-Scholes `d1` and `d2` terms.
///
/// `d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))` and
/// `d2 = d1 - sigma sqrt(T)`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] unless `s`, `k`, `t` and `sigma` are all
/// strictly positive (the terms are singular at the boundary).
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::d1_d2;
/// let (d1, d2) = d1_d2(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
/// assert!((d1 - 0.35).abs() < 1e-12);
/// assert!((d2 - 0.15).abs() < 1e-12);
/// assert!(d1_d2(100.0, 100.0, 0.0, 0.05, 0.2, 0.0).is_err());
/// ```
pub fn d1_d2(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
) -> Result<(f64, f64), PricingError> {
    validate_inputs(s, k, t, sigma)?;
    validate_rates(r, q)?;
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return Err(PricingError::InvalidInput(format!(
            "d1/d2 require strictly positive S, K, T and sigma; \
             got S={s}, K={k}, T={t}, sigma={sigma}"
        )));
    }
    Ok(d1_d2_core(s, k, t, r, sigma, q))
}

/// Black-Scholes-Merton price of a European option with dividend yield.
///
/// # Arguments
///
/// * `s`: spot price (currency units), `s >= 0`.
/// * `k`: strike price (currency units), `k >= 0`.
/// * `t`: time to expiry in years (ACT/365F), `t >= 0`.
/// * `r`: continuously compounded annualised risk-free rate; negative
///   rates are supported.
/// * `sigma`: annualised log-return volatility, `sigma >= 0`.
/// * `q`: continuously compounded annualised dividend yield.
/// * `option_type`: call or put.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] if `s`, `k`, `t` or `sigma` is negative
/// or non-finite, or if `r` or `q` is NaN/infinite.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, OptionType};
/// let c = bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
/// assert!((c - 10.450583572185565).abs() < 1e-12); // golden value
///
/// // T = 0 returns intrinsic:
/// let expired = bs_price(105.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
/// assert_eq!(expired, 5.0);
/// ```
pub fn bs_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> Result<f64, PricingError> {
    validate_inputs(s, k, t, sigma)?;
    validate_rates(r, q)?;
    finite_result(bs_price_core(s, k, t, r, sigma, q, option_type), "bs_price")
}

/// Check the four non-negative inputs without allocating.
///
/// Rejects exactly the inputs [`validate_inputs`] rejects.
#[inline]
pub(crate) fn check_inputs(s: f64, k: f64, t: f64, sigma: f64) -> Result<(), ErrorCode> {
    for value in [s, k, t, sigma] {
        if value.is_nan() {
            return Err(ErrorCode::NanInput);
        }
        if value.is_infinite() {
            return Err(ErrorCode::InfiniteInput);
        }
        if value < 0.0 {
            return Err(ErrorCode::NegativeInput);
        }
    }
    Ok(())
}

/// Check the two rates without allocating. Rejects exactly the inputs
/// [`validate_rates`] rejects.
#[inline]
pub(crate) fn check_rates(r: f64, q: f64) -> Result<(), ErrorCode> {
    if r.is_finite() && q.is_finite() {
        Ok(())
    } else {
        Err(ErrorCode::NonFiniteRate)
    }
}

/// Black-Scholes price for inputs that already passed validation.
///
/// Shared by [`bs_price`] and the non-allocating API, so both return the
/// same bits. The result can be non-finite; callers check it.
#[inline]
pub(crate) fn bs_price_core(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> f64 {
    if t == 0.0 {
        return intrinsic_value(s, k, option_type);
    }
    let disc_s = s * (-q * t).exp();
    let disc_k = k * (-r * t).exp();
    if k == 0.0 {
        // Zero-strike call is a (dividend-adjusted) forward on the stock.
        match option_type {
            OptionType::Call => disc_s,
            OptionType::Put => 0.0,
        }
    } else if s == 0.0 {
        match option_type {
            OptionType::Call => 0.0,
            OptionType::Put => disc_k,
        }
    } else if sigma == 0.0 {
        // e^{-rT} max(+-(F - K), 0) written as S e^{-qT} - K e^{-rT}: the
        // forward itself overflows when (r - q) T > 709.
        (option_type.sign() * (disc_s - disc_k)).max(0.0)
    } else {
        // Shared carry-form kernel (qp_core::black_scholes). The + 0.0 turns
        // the -0.0 the phi form gives for a put worth exactly zero into the
        // +0.0 this function has always returned; no other value moves.
        let (d1, d2) = d1_d2_core(s, k, t, r, sigma, q);
        carry_value::<ErfcNormal>(disc_s, disc_k, d1, d2, option_type.sign()) + 0.0
    }
}

/// `d1`, `d2` for strictly positive, validated inputs (same expression
/// as [`d1_d2`]).
#[inline]
pub(crate) fn d1_d2_core(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> (f64, f64) {
    carry_d1_d2(s, k, t, r, q, sigma)
}

/// Return `value`, or [`PricingError::InvalidInput`] if it is not finite
/// (a discount factor or forward overflowed for finite inputs).
pub(crate) fn finite_result(value: f64, what: &str) -> Result<f64, PricingError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PricingError::InvalidInput(format!(
            "{what}: result is not representable as a finite double              (a discount factor or forward overflows)"
        )))
    }
}
