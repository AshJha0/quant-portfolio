//! FX forwards via covered interest parity (CIP).
//!
//! Covered interest parity: `F = S * exp((r_d - r_f) * T)`.  A domestic
//! investor can replicate the forward by borrowing domestic cash, buying
//! spot foreign currency and depositing it at `r_f`; absence of arbitrage
//! forces the forward to the CIP level (abstracting from the
//! cross-currency basis — see docs/METHODOLOGY.md assumption register).
//!
//! Forward points are quoted as `(F - S)` scaled by the pair's pip factor
//! (1e4 for most pairs, 1e2 for JPY-quoted pairs).

use crate::{invalid, require_finite, validate_inputs, FxResult};

/// Standard pip scaling for most pairs (pip = 0.0001, e.g. EURUSD).
pub const PIP_FACTOR_DEFAULT: f64 = 1e4;

/// Pip scaling for JPY-quoted pairs (pip = 0.01, e.g. USDJPY).
pub const PIP_FACTOR_JPY: f64 = 1e2;

/// Covered-interest-parity forward rate `F = S * exp((r_d - r_f) T)`.
///
/// `F > S` when the domestic rate exceeds the foreign rate (forward
/// premium on the base currency), the classic carry relationship.
///
/// # Arguments
///
/// * `s` — spot rate, domestic per unit foreign, `> 0`.
/// * `t` — time to delivery in years, `>= 0`.
/// * `r_d`, `r_f` — domestic / foreign continuously compounded rates
///   (ACT/365F).
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs.
///
/// ```
/// use fx_options_engine::cip_forward;
/// let f = cip_forward(1.10, 1.0, 0.0425, 0.0290).unwrap();
/// assert!(f > 1.10); // r_d > r_f: base currency at a forward premium
/// ```
pub fn cip_forward(s: f64, t: f64, r_d: f64, r_f: f64) -> FxResult<f64> {
    validate_inputs(s, s, t, r_d, r_f, 0.0)?;
    Ok(s * ((r_d - r_f) * t).exp())
}

/// Forward points `(F - S) * pip_factor`.
///
/// # Arguments
///
/// * `s`, `t`, `r_d`, `r_f` — as in [`cip_forward`].
/// * `pip_factor` — [`PIP_FACTOR_DEFAULT`] (1e4) for e.g. EURUSD,
///   [`PIP_FACTOR_JPY`] (1e2) for USDJPY.
///
/// Positive when the base currency trades at a forward premium.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs or non-positive
/// `pip_factor`.
pub fn forward_points(s: f64, t: f64, r_d: f64, r_f: f64, pip_factor: f64) -> FxResult<f64> {
    if !(pip_factor > 0.0) {
        return invalid(format!("pip_factor must be positive, got {pip_factor}"));
    }
    Ok((cip_forward(s, t, r_d, r_f)? - s) * pip_factor)
}

/// Forward implied by put–call parity (a "synthetic forward").
///
/// Conversion/parity: `C - P = e^{-r_d T} (F - K)`, hence
/// `F = K + (C - P) e^{r_d T}`.  Long call + short put at the same strike
/// replicates a forward purchase of the base currency; desks call the
/// position a *synthetic forward* (or a "conversion" when run against an
/// actual forward to lock in the mispricing).
///
/// # Arguments
///
/// * `call_price`, `put_price` — European premiums (domestic ccy per unit
///   foreign) at strike `k`.
/// * `k` — common strike, `> 0`.
/// * `t` — time to expiry in years, `>= 0`.
/// * `r_d` — domestic continuously compounded rate.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs.
///
/// ```
/// use fx_options_engine::{cip_forward, gk_call, gk_put,
///                         synthetic_forward_from_options};
/// let (s, k, t, rd, rf, vol) = (1.10, 1.05, 0.75, 0.0425, 0.0290, 0.0925);
/// let c = gk_call(s, k, t, rd, rf, vol).unwrap();
/// let p = gk_put(s, k, t, rd, rf, vol).unwrap();
/// let f_syn = synthetic_forward_from_options(c, p, k, t, rd).unwrap();
/// let f_cip = cip_forward(s, t, rd, rf).unwrap();
/// assert!((f_syn - f_cip).abs() < 1e-12);
/// ```
pub fn synthetic_forward_from_options(
    call_price: f64,
    put_price: f64,
    k: f64,
    t: f64,
    r_d: f64,
) -> FxResult<f64> {
    validate_inputs(k, k, t, r_d, 0.0, 0.0)?;
    require_finite(call_price, "call_price")?;
    require_finite(put_price, "put_price")?;
    Ok(k + (call_price - put_price) * (r_d * t).exp())
}
