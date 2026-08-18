//! Black-76 pricing off the FX forward.
//!
//! FX desks think in forwards: the smile is marked against the forward,
//! and Black-76 prices directly off it,
//!
//! ```text
//! call = e^{-r_d T} [F N(d1) - K N(d2)],
//! d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T).
//! ```
//!
//! With the covered-interest-parity forward `F = S e^{(r_d - r_f) T}`,
//! Black-76 is *algebraically identical* to Garman–Kohlhagen —
//! substituting `F` recovers the GK d1/d2 and price exactly.  The
//! practical difference is the input: quoting off `F` removes the need to
//! know the rate *pair*, only the domestic discount factor, which is how
//! forward-space market data arrives.

use crate::forwards::cip_forward;
use crate::{invalid, norm_cdf, require_finite, FxResult, OptionType};

/// Black-76 price of a European FX option on the forward.
///
/// # Arguments
///
/// * `f` — outright forward rate (domestic per unit foreign), `> 0`.
/// * `k` — strike, `> 0`.
/// * `t` — time to expiry in years, `>= 0`.
/// * `r_d` — domestic continuously compounded rate (discounting only).
/// * `sigma` — annualised volatility, `>= 0`.
/// * `option_type` — call or put on the base currency.
///
/// Returns the premium in domestic currency per unit foreign notional.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs.
///
/// ```
/// use fx_options_engine::{black76_price, OptionType};
/// // At-the-forward call and put have equal value (parity at K = F).
/// let c = black76_price(1.11, 1.11, 0.5, 0.0425, 0.10, OptionType::Call).unwrap();
/// let p = black76_price(1.11, 1.11, 0.5, 0.0425, 0.10, OptionType::Put).unwrap();
/// assert!((c - p).abs() < 1e-15);
/// ```
pub fn black76_price(
    f: f64,
    k: f64,
    t: f64,
    r_d: f64,
    sigma: f64,
    option_type: OptionType,
) -> FxResult<f64> {
    let phi = option_type.phi();
    require_finite(f, "F")?;
    require_finite(k, "K")?;
    require_finite(t, "T")?;
    require_finite(r_d, "r_d")?;
    require_finite(sigma, "sigma")?;
    if f <= 0.0 {
        return invalid(format!("Forward F must be positive, got {f}"));
    }
    if k <= 0.0 {
        return invalid(format!("Strike K must be positive, got {k}"));
    }
    if t < 0.0 {
        return invalid(format!("Time to expiry T must be non-negative, got {t}"));
    }
    if sigma < 0.0 {
        return invalid(format!("Volatility sigma must be non-negative, got {sigma}"));
    }

    let df = (-r_d * t).exp();
    let v = sigma * t.sqrt();
    if t == 0.0 || v <= 1e-12 {
        return Ok(df * (phi * (f - k)).max(0.0));
    }
    let d1 = ((f / k).ln() + 0.5 * v * v) / v;
    let d2 = d1 - v;
    Ok(phi * df * (f * norm_cdf(phi * d1) - k * norm_cdf(phi * d2)))
}

/// Black-76 with the forward built from spot via CIP.
///
/// Equals [`crate::gk_price`] to machine precision — tested to 1e-12 in
/// the suite.
///
/// # Errors
///
/// Same as [`black76_price`].
pub fn black76_from_spot(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
) -> FxResult<f64> {
    let f = cip_forward(s, t, r_d, r_f)?;
    black76_price(f, k, t, r_d, sigma, option_type)
}
