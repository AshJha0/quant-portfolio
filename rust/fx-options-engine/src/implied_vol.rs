//! Implied Garman–Kohlhagen volatility from a domestic-currency premium.
//!
//! Strategy: Newton–Raphson from a moneyness-aware initial guess (fast
//! quadratic convergence when vega is healthy), falling back to bracketed
//! Brent (guaranteed convergence) if Newton stalls or wanders outside the
//! no-arbitrage bracket.  Mirrors the Python
//! `fx_options.garman_kohlhagen.implied_vol` and the C++ engine exactly:
//! same bounds checks, same tolerances, same degenerate-price behaviour.

use crate::garman_kohlhagen::{d1 as d1_fn, gk_price};
use crate::{brentq, invalid, norm_pdf, require_finite, validate_inputs, FxResult, OptionType};

/// Absolute tolerance on the vol root (round trips hold to 1e-10).
const TOL: f64 = 1e-12;
/// Newton iteration budget before the Brent fallback.
const MAX_ITER: usize = 100;

/// GK vega (dV/dsigma), same for calls and puts.
fn gk_vega(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    let d1v = d1_fn(s, k, t, r_d, r_f, sigma)?;
    Ok(s * (-r_f * t).exp() * norm_pdf(d1v) * t.sqrt())
}

/// Implied Garman–Kohlhagen volatility from a domestic-currency premium.
///
/// # Arguments
///
/// * `price` — observed premium, domestic ccy per unit foreign notional.
/// * `s`, `k`, `t`, `r_d`, `r_f` — as in [`crate::gk_price`]; requires
///   `t > 0`.
/// * `option_type` — call or put.
///
/// Returns the implied volatility (annualised) to absolute tolerance
/// 1e-12 (price round trips reproduce the input vol to better than
/// 1e-10 across the tested strike/vol grid).
///
/// A price whose time value is below double-precision resolution returns
/// `0.0` (the `sigma -> 0` limit; vol unrecoverable — documented in
/// docs/VALIDATION.md).
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] if the price violates the
/// no-arbitrage bounds `[discounted intrinsic on the forward, discounted
/// forward bound]` or `t = 0`.
///
/// ```
/// use fx_options_engine::{gk_price, implied_vol, OptionType};
/// let (s, k, t, rd, rf) = (1.10, 1.15, 0.5, 0.0425, 0.0290);
/// let px = gk_price(s, k, t, rd, rf, 0.0925, OptionType::Call).unwrap();
/// let iv = implied_vol(px, s, k, t, rd, rf, OptionType::Call).unwrap();
/// assert!((iv - 0.0925).abs() < 1e-10);
/// ```
pub fn implied_vol(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    option_type: OptionType,
) -> FxResult<f64> {
    let phi = option_type.phi();
    validate_inputs(s, k, t, r_d, r_f, 0.0)?;
    if t <= 0.0 {
        return invalid("implied_vol requires T > 0");
    }
    require_finite(price, "price")?;

    let df_d = (-r_d * t).exp();
    let df_f = (-r_f * t).exp();
    let forward = s * df_f / df_d;
    let lower = df_d * (phi * (forward - k)).max(0.0); // sigma -> 0 limit
    let upper = if phi > 0.0 { s * df_f } else { k * df_d }; // sigma -> inf
    if price < lower - 1e-14 || price > upper + 1e-14 {
        return invalid(format!(
            "price {price} outside no-arbitrage bounds [{lower}, {upper}]"
        ));
    }
    if price - lower <= 1e-16 * lower.max(1.0) {
        // Time value below double-precision resolution: vol unrecoverable,
        // return the sigma -> 0 limit (documented in docs/VALIDATION.md).
        return Ok(0.0);
    }

    let objective = |sig: f64| -> FxResult<f64> {
        Ok(gk_price(s, k, t, r_d, r_f, sig, option_type)? - price)
    };

    // Newton with a moneyness-aware start (Brenner-Subrahmanyam flavoured).
    let mut sigma = (2.0 * (forward / k).ln().abs() / t).sqrt().max(0.05);
    let lo = 1e-10;
    let hi_cap = 10.0;
    for _ in 0..MAX_ITER {
        let diff = objective(sigma)?;
        let vega = gk_vega(s, k, t, r_d, r_f, sigma)?;
        if vega < 1e-12 {
            // Flat objective: an absolute price tolerance here would be
            // meaningless (a wing option's vega can be ~1e-9, so even a
            // sub-ULP price residual implies a sigma error orders of
            // magnitude above `TOL` — this is what let short-dated wing
            // round trips return early ~1e-7 off).  Only accept if the
            // price residual is already at the double-precision noise
            // floor; otherwise hand off to bracketed Brent, which solves
            // in the sigma domain and is immune to the flat vega.
            if diff.abs() < 1e-14 {
                return Ok(sigma);
            }
            break;
        }
        let new_sigma = sigma - diff / vega;
        if !(new_sigma > lo && new_sigma < hi_cap) {
            break;
        }
        if (new_sigma - sigma).abs() < TOL {
            sigma = new_sigma;
            break;
        }
        sigma = new_sigma;
    }

    // Brent fallback.  The objective cannot fail inside [lo, hi_cap]:
    // sigma > 0, t > 0 and the inputs are validated.
    let obj_infallible = |sig: f64| {
        gk_price(s, k, t, r_d, r_f, sig, option_type)
            .map(|p| p - price)
            .unwrap_or(f64::NAN)
    };

    // First try a bracket centred tightly on the last Newton iterate and
    // expand it geometrically.  For flat-vega wings (short-dated, deep
    // ITM/OTM) the GK call/put formula loses precision to cancellation
    // (S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2), both O(1), difference
    // O(price)); the objective is then a *staircase* in sigma with ULP-
    // sized flats, and Brent converges to the nearest edge of that flat
    // fastest, and with least residual bias, when seeded from a narrow
    // window around a value already close to the root — a wide bracket
    // approaches the same flat from farther away and can settle on the
    // wrong edge.  Falls back to the old wide search from 1.0 if the
    // narrow window never straddles a sign change (e.g. Newton broke out
    // on iteration 0 with a poor start).
    let mut width = (sigma * 1e-4).max(1e-6);
    while width < 2.0 {
        let blo = (sigma - width).max(lo);
        let bhi = (sigma + width).min(hi_cap);
        let flo = obj_infallible(blo);
        let fhi = obj_infallible(bhi);
        if flo == 0.0 {
            return Ok(blo);
        }
        if fhi == 0.0 {
            return Ok(bhi);
        }
        if flo.is_finite() && fhi.is_finite() && flo * fhi < 0.0 {
            return brentq(obj_infallible, blo, bhi, TOL, 200);
        }
        width *= 8.0;
    }

    let mut hi = 1.0;
    while obj_infallible(hi) < 0.0 && hi < 50.0 {
        hi *= 2.0;
    }
    if obj_infallible(hi) < 0.0 {
        return invalid(format!("implied vol > {hi}: price {price} unattainably high"));
    }
    brentq(obj_infallible, lo, hi, TOL, 200)
}
