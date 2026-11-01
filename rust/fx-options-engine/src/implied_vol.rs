//! Implied Garman-Kohlhagen volatility from a domestic-currency premium.
//!
//! The quote is mapped to normalised Black form and solved on the
//! out-of-the-money side (an in-the-money quote is converted by parity)
//! with the shared core in [`crate::iv_core`]: safeguarded Halley
//! iteration on transformed objectives with a relative stopping rule. The
//! same algorithm, constants and bounds tolerances run in the Python
//! `fx_options.implied_vol` and the C++ engine.

use crate::iv_core::black_implied_vol;
use crate::{finite_result, invalid, require_finite, validate_inputs, FxError, FxResult, OptionType};

/// Implied Garman-Kohlhagen volatility from a domestic-currency premium.
///
/// # Arguments
///
/// * `price`: observed premium, domestic ccy per unit foreign notional.
/// * `s`, `k`, `t`, `r_d`, `r_f`: as in [`crate::gk_price`]; requires
///   `t > 0`.
/// * `option_type`: call or put.
///
/// Accuracy: within `max(eps * price / vega, 1e-12 * sigma)` of the true
/// root, in at most 10 evaluations (typically 2-4).
///
/// Bounds use scale-aware tolerances: `tau_lo = 4 eps df_d max(F, K)` at
/// the intrinsic bound when the option is in the money (zero otherwise)
/// and `tau = 4 eps upper` at the `sigma -> inf` bound. A price within
/// `tau_lo` of the intrinsic bound returns `0.0`; a price within `tau` of
/// the upper bound is an error, because no finite vol is identifiable.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] if the price lies outside
/// `(lower - tau_lo, upper - tau)`, if `t = 0` or an input is invalid;
/// [`crate::FxError::Numerical`] if the solver does not converge.
///
/// ```
/// use fx_options_engine::{gk_price, implied_vol, OptionType};
/// let (s, k, t, rd, rf) = (1.10, 1.15, 0.5, 0.0425, 0.0290);
/// let px = gk_price(s, k, t, rd, rf, 0.0925, OptionType::Call).unwrap();
/// let iv = implied_vol(px, s, k, t, rd, rf, OptionType::Call).unwrap();
/// assert!((iv - 0.0925).abs() < 1e-14);
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

    let df_d = finite_result((-r_d * t).exp(), "implied_vol")?;
    let df_f = finite_result((-r_f * t).exp(), "implied_vol")?;
    let forward = finite_result(s * df_f / df_d, "forward")?;
    let lower = df_d * (phi * (forward - k)).max(0.0); // sigma -> 0 limit
    let upper = if phi > 0.0 { s * df_f } else { k * df_d }; // sigma -> inf
    // Rounding of the sigma -> 0 bound df (F - K): a few eps of df max(F, K)
    // when the option is in the money; exactly zero otherwise.
    let tau_lo = if lower > 0.0 {
        4.0 * f64::EPSILON * df_d * forward.max(k)
    } else {
        0.0
    };
    let tau = 4.0 * f64::EPSILON * upper;
    if price < lower - tau_lo || price > upper + tau {
        return invalid(format!(
            "price {price} outside no-arbitrage bounds [{lower}, {upper}]"
        ));
    }
    if price - lower <= tau_lo {
        // Time value below the rounding of the bound: return the
        // sigma -> 0 limit (documented in docs/VALIDATION.md).
        return Ok(0.0);
    }
    if upper - price <= tau {
        return invalid(format!(
            "price {price} is within {tau:e} of the sigma->inf bound {upper}; \
             no finite implied volatility is identifiable"
        ));
    }
    let mut evals = 0;
    black_implied_vol(price, forward, k, t, df_d, phi > 0.0, &mut evals).map_err(|msg| {
        if msg.starts_with("price") {
            FxError::InvalidInput(msg.into())
        } else {
            FxError::Numerical(msg.into())
        }
    })
}
