//! Garman–Kohlhagen pricing for European FX options.
//!
//! Garman–Kohlhagen (1983) is Black–Scholes with the continuous dividend
//! yield replaced by the foreign interest rate: holding the foreign
//! currency pays the foreign risk-free rate, exactly as a dividend-paying
//! stock pays its yield.
//!
//! Conventions: pair BASE/QUOTE (EURUSD = USD per EUR); `s` and prices in
//! domestic (quote) currency per unit foreign (base) notional; `r_d` =
//! quote currency rate, `r_f` = base currency rate; rates continuously
//! compounded, annualised, ACT/365F.
//!
//! # Formulae
//!
//! ```text
//! d1 = [ln(S/K) + (r_d - r_f + sigma^2/2) T] / (sigma sqrt(T))
//! d2 = d1 - sigma sqrt(T)
//! call = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2)
//! put  = K e^{-r_d T} N(-d2) - S e^{-r_f T} N(-d1)
//! ```
//!
//! Limits handled explicitly: `T = 0` returns intrinsic value;
//! `sigma = 0` (or `sigma*sqrt(T) = 0`) returns the discounted intrinsic
//! on the forward, `e^{-r_d T} max(phi (F - K), 0)`.

use crate::{invalid, norm_cdf, validate_inputs, FxResult, OptionType};

/// Threshold below which `sigma * sqrt(T)` is treated as zero vol.
pub(crate) const MIN_VOL: f64 = 1e-12;

/// Garman–Kohlhagen `d1`.
///
/// # Arguments
///
/// * `s`, `k` — spot (domestic per foreign) and strike, both `> 0`.
/// * `t` — time to expiry in years, `> 0`.
/// * `r_d`, `r_f` — domestic / foreign continuously compounded rates.
/// * `sigma` — annualised volatility, `> 0`.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] if inputs are invalid or
/// `sigma * sqrt(T)` is zero.
///
/// ```
/// use fx_options_engine::{d1, d2};
/// let x1 = d1(1.10, 1.10, 0.5, 0.0425, 0.0290, 0.0925).unwrap();
/// let x2 = d2(1.10, 1.10, 0.5, 0.0425, 0.0290, 0.0925).unwrap();
/// let v = 0.0925 * 0.5_f64.sqrt();
/// assert!((x1 - x2 - v).abs() < 1e-15);
/// ```
pub fn d1(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    validate_inputs(s, k, t, r_d, r_f, sigma)?;
    let vol_sqrt_t = sigma * t.sqrt();
    if vol_sqrt_t <= MIN_VOL {
        return invalid(format!(
            "d1 undefined for sigma*sqrt(T)={vol_sqrt_t}; need sigma>0, T>0"
        ));
    }
    Ok(((s / k).ln() + (r_d - r_f + 0.5 * sigma * sigma) * t) / vol_sqrt_t)
}

/// Garman–Kohlhagen `d2 = d1 - sigma*sqrt(T)`.  See [`d1`].
///
/// # Errors
///
/// Same as [`d1`].
pub fn d2(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    Ok(d1(s, k, t, r_d, r_f, sigma)? - sigma * t.sqrt())
}

/// Garman–Kohlhagen price of a European FX option.
///
/// Price is in domestic (quote) currency per unit of foreign (base)
/// notional, e.g. USD per EUR for EURUSD.
///
/// # Arguments
///
/// * `s` — spot FX rate, domestic per unit foreign, `> 0`.
/// * `k` — strike in the same quotation, `> 0`.
/// * `t` — time to expiry in years, `>= 0` (`t = 0` returns intrinsic).
/// * `r_d` — domestic (quote-currency) continuously compounded rate.
/// * `r_f` — foreign (base-currency) continuously compounded rate.
/// * `sigma` — annualised volatility, `>= 0` (`sigma = 0` returns
///   discounted forward intrinsic).
/// * `option_type` — [`OptionType::Call`] (call on the base currency) or
///   [`OptionType::Put`].
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs.
///
/// ```
/// use fx_options_engine::{gk_price, OptionType};
/// // Two-rate put-call parity: C - P = S e^{-r_f T} - K e^{-r_d T}.
/// let (s, k, t, rd, rf, vol) = (1.10, 1.05, 0.75, 0.0425, 0.0290, 0.0925);
/// let c = gk_price(s, k, t, rd, rf, vol, OptionType::Call).unwrap();
/// let p = gk_price(s, k, t, rd, rf, vol, OptionType::Put).unwrap();
/// let parity = s * (-rf * t).exp() - k * (-rd * t).exp();
/// assert!((c - p - parity).abs() < 1e-15);
/// ```
pub fn gk_price(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
) -> FxResult<f64> {
    let phi = option_type.phi();
    validate_inputs(s, k, t, r_d, r_f, sigma)?;
    if t == 0.0 {
        return Ok((phi * (s - k)).max(0.0));
    }
    if sigma * t.sqrt() <= MIN_VOL {
        let forward = s * ((r_d - r_f) * t).exp();
        return Ok((-r_d * t).exp() * (phi * (forward - k)).max(0.0));
    }
    let d1v = d1(s, k, t, r_d, r_f, sigma)?;
    let d2v = d1v - sigma * t.sqrt();
    Ok(phi
        * (s * (-r_f * t).exp() * norm_cdf(phi * d1v)
            - k * (-r_d * t).exp() * norm_cdf(phi * d2v)))
}

/// Convenience wrapper: `gk_price(..., OptionType::Call)`.
///
/// # Errors
///
/// Same as [`gk_price`].
pub fn gk_call(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    gk_price(s, k, t, r_d, r_f, sigma, OptionType::Call)
}

/// Convenience wrapper: `gk_price(..., OptionType::Put)`.
///
/// # Errors
///
/// Same as [`gk_price`].
pub fn gk_put(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    gk_price(s, k, t, r_d, r_f, sigma, OptionType::Put)
}
