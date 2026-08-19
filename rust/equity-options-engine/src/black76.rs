//! Black-76 pricing and Greeks for options on forwards/futures.
//!
//! Use case: equity *index futures* options (and index options quoted off
//! the forward). Marking off the forward absorbs both the financing rate
//! and the (hard-to-observe) dividend yield into one observable input `f`,
//! which is why index vol desks quote and risk-manage in Black-76 terms.
//!
//! # Conventions (identical to the Python reference)
//!
//! * `f` is the forward/futures price for expiry `t` (years, ACT/365F).
//! * `r` is the continuously compounded annualised discount rate applied
//!   to the premium (for daily-margined futures options with no premium
//!   discounting, pass `r = 0`).
//! * `sigma` is the annualised volatility of the forward's log-returns.
//! * Equivalence with Black-Scholes: with `F = S exp((r - q) T)`, Black-76
//!   reproduces the Black-Scholes-Merton price exactly.
//!
//! Greeks are with respect to the forward `f` (delta, gamma) and per unit
//! of vol / per year / per unit of rate (vega, theta, rho). Rho here is
//! the sensitivity of the *discounting only* (`f` held fixed):
//! `rho = -T * price`.

use crate::black_scholes::{
    norm_cdf, norm_pdf, validate_inputs, validate_rates, OptionType, PricingError,
};

/// Analytic Black-76 Greeks (with respect to the forward `f`).
///
/// # Examples
///
/// ```
/// use eq_options_engine::{black76_greeks, OptionType};
/// let g = black76_greeks(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call).unwrap();
/// assert!((g.rho + 1.0 * g.price).abs() < 1e-14); // rho = -T * V
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Black76Greeks {
    /// Present value, currency units.
    pub price: f64,
    /// dV/dF, dimensionless.
    pub delta: f64,
    /// d2V/dF2, per currency unit.
    pub gamma: f64,
    /// dV/dsigma, per unit of annualised vol.
    pub vega: f64,
    /// dV/dt = -dV/dT, per year (calendar decay with F fixed).
    pub theta: f64,
    /// dV/dr with F held fixed: pure discounting sensitivity `-T * V`.
    pub rho: f64,
}

/// Black-76 `d1`/`d2`: `d1 = [ln(F/K) + sigma^2 T/2] / (sigma sqrt(T))`,
/// `d2 = d1 - sigma sqrt(T)`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] unless `f`, `k`, `t`, `sigma` are all
/// strictly positive.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black76::b76_d1_d2;
/// let (d1, d2) = b76_d1_d2(100.0, 100.0, 1.0, 0.2).unwrap();
/// assert!((d1 - 0.1).abs() < 1e-12);
/// assert!((d2 + 0.1).abs() < 1e-12);
/// ```
pub fn b76_d1_d2(f: f64, k: f64, t: f64, sigma: f64) -> Result<(f64, f64), PricingError> {
    validate_inputs(f, k, t, sigma)?;
    if f <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return Err(PricingError::InvalidInput(format!(
            "b76_d1_d2 requires strictly positive F, K, T, sigma; \
             got F={f}, K={k}, T={t}, sigma={sigma}"
        )));
    }
    let sqrt_t = t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    Ok((d1, d1 - sigma * sqrt_t))
}

/// Black-76 present value of a European option on a forward/futures price.
///
/// `t == 0` returns intrinsic; `sigma == 0` (or a degenerate `f`/`k`)
/// returns the discounted intrinsic `exp(-rT) max(±(F - K), 0)`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] on negative/NaN inputs.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{black76_price, bs_price, OptionType};
/// // Black-76 on the forward equals Black-Scholes-Merton on spot:
/// let (s, t, r, q, sigma): (f64, f64, f64, f64, f64) = (100.0, 1.0, 0.05, 0.02, 0.2);
/// let fwd = s * ((r - q) * t).exp();
/// let b76 = black76_price(fwd, 100.0, t, r, sigma, OptionType::Call).unwrap();
/// let bsm = bs_price(s, 100.0, t, r, sigma, q, OptionType::Call).unwrap();
/// assert!((b76 - bsm).abs() < 1e-12);
/// ```
pub fn black76_price(
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: OptionType,
) -> Result<f64, PricingError> {
    validate_inputs(f, k, t, sigma)?;
    validate_rates(r, 0.0)?;
    let sign = option_type.sign();
    if t == 0.0 {
        return Ok((sign * (f - k)).max(0.0));
    }
    let df = (-r * t).exp();
    if sigma == 0.0 || k == 0.0 || f == 0.0 {
        return Ok(df * (sign * (f - k)).max(0.0));
    }
    let (d1, d2) = b76_d1_d2(f, k, t, sigma)?;
    Ok(df * sign * (f * norm_cdf(sign * d1) - k * norm_cdf(sign * d2)))
}

/// Analytic Black-76 Greeks with respect to the forward.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] unless `f`, `k`, `t`, `sigma` are
/// strictly positive (finite Greeks require the interior of the domain).
///
/// # Examples
///
/// ```
/// use eq_options_engine::{black76_greeks, OptionType};
/// let c = black76_greeks(105.0, 100.0, 0.5, 0.03, 0.25, OptionType::Call).unwrap();
/// let p = black76_greeks(105.0, 100.0, 0.5, 0.03, 0.25, OptionType::Put).unwrap();
/// // Call and put share gamma and vega:
/// assert!((c.gamma - p.gamma).abs() < 1e-15);
/// assert!((c.vega - p.vega).abs() < 1e-15);
/// ```
pub fn black76_greeks(
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: OptionType,
) -> Result<Black76Greeks, PricingError> {
    validate_rates(r, 0.0)?;
    let (d1, d2) = b76_d1_d2(f, k, t, sigma)?;
    let df = (-r * t).exp();
    let sqrt_t = t.sqrt();
    let pdf_d1 = norm_pdf(d1);
    let sign = option_type.sign();

    let price = df * sign * (f * norm_cdf(sign * d1) - k * norm_cdf(sign * d2));
    let delta = df * sign * norm_cdf(sign * d1);
    let gamma = df * pdf_d1 / (f * sigma * sqrt_t);
    let vega = df * f * pdf_d1 * sqrt_t;
    // theta = dV/dt at fixed F: r*V minus decay of the time value.
    let theta = r * price - df * f * pdf_d1 * sigma / (2.0 * sqrt_t);
    let rho = -t * price;
    Ok(Black76Greeks {
        price,
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}
