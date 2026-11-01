//! Implied Black-Scholes volatility, solved on the out-of-the-money side.
//!
//! The quote is mapped to normalised Black form and an in-the-money quote
//! is converted to its out-of-the-money counterpart by parity, so the
//! solver always works on time value. The shared core ([`crate::iv_core`])
//! runs safeguarded Halley iteration on transformed objectives with a
//! relative stopping rule; the same algorithm runs in the Python and C++
//! engines.

use crate::black_scholes::{bs_price, finite_result, validate_inputs, validate_rates, OptionType, PricingError};
use crate::iv_core::black_implied_vol;

/// Former absolute price tolerance of the bisection solver. Unused.
#[deprecated(note = "the implied-vol solver no longer uses a price tolerance")]
pub const IV_PRICE_TOL: f64 = 1e-10;

/// Implied Black-Scholes volatility from an observed premium.
///
/// Accuracy: within `max(eps * price / vega, 1e-12 * sigma)` of the true
/// root over `T` in `[1e-7, 50]`, `sigma` in `[1e-4, 5]` and
/// `|ln(F/K)| <= 5`, in at most 10 evaluations (typically 2-4). For a
/// deep in-the-money quote the price carries the information limit: with
/// S=100, K=90, T=0.02, sigma=10% one ulp of the call price moves the vol
/// by about 5e-4. The returned vol then reprices the quote to a few ulps.
///
/// # Arguments
///
/// * `price`: observed option premium, currency units; must lie strictly
///   between the no-arbitrage bounds.
/// * `s`, `k`, `t`, `r`, `q`, `option_type`: as in [`bs_price`];
///   `t`, `s`, `k` must be strictly positive.
///
/// # Errors
///
/// * [`PricingError::InvalidInput`] for NaN/negative inputs, `t == 0`, or
///   a discount factor that overflows.
/// * [`PricingError::ArbitrageBound`] for prices at or outside the bounds.
/// * [`PricingError::NoConvergence`] if the solver does not converge in
///   100 iterations (not observed on the test grid).
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, implied_vol, OptionType};
/// let ot = OptionType::Call;
/// let price = bs_price(100.0, 110.0, 0.5, 0.03, 0.25, 0.01, ot).unwrap();
/// let iv = implied_vol(price, 100.0, 110.0, 0.5, 0.03, 0.01, ot).unwrap();
/// assert!((iv - 0.25).abs() < 1e-14);
///
/// // Sub-intrinsic prices are rejected:
/// assert!(implied_vol(0.0, 100.0, 50.0, 1.0, 0.05, 0.0, ot).is_err());
/// ```
pub fn implied_vol(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
) -> Result<f64, PricingError> {
    validate_inputs(s, k, t, 0.0)?;
    validate_rates(r, q)?;
    if !price.is_finite() {
        return Err(PricingError::InvalidInput(format!(
            "price must be finite, got {price}"
        )));
    }
    if t <= 0.0 {
        return Err(PricingError::InvalidInput(
            "implied_vol requires T > 0 (option already expired)".into(),
        ));
    }
    if s <= 0.0 || k <= 0.0 {
        return Err(PricingError::InvalidInput(
            "implied_vol requires S > 0 and K > 0".into(),
        ));
    }

    // Static no-arbitrage bounds: sigma -> 0 (discounted forward intrinsic)
    // and sigma -> inf.
    let lower = bs_price(s, k, t, r, 0.0, q, option_type)?;
    let upper = match option_type {
        OptionType::Call => finite_result(s * (-q * t).exp(), "implied_vol")?,
        OptionType::Put => k * (-r * t).exp(),
    };
    if price <= lower {
        return Err(PricingError::ArbitrageBound(format!(
            "price {price} is at or below the sigma->0 bound {lower}; \
             implied vol is undefined"
        )));
    }
    if price >= upper {
        return Err(PricingError::ArbitrageBound(format!(
            "price {price} is at or above the sigma->inf bound {upper}; \
             implied vol is undefined"
        )));
    }

    let df = finite_result((-r * t).exp(), "implied_vol")?;
    let forward = finite_result(s * ((r - q) * t).exp(), "implied_vol")?;
    let mut evals = 0;
    black_implied_vol(price, forward, k, t, df, option_type == OptionType::Call, &mut evals)
        .map_err(|msg| {
            if msg.starts_with("price") {
                PricingError::ArbitrageBound(msg.into())
            } else {
                PricingError::NoConvergence(msg.into())
            }
        })
}
