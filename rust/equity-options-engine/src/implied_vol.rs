//! Implied Black-Scholes volatility: bracketed Newton with bisection fallback.
//!
//! Newton iterations use analytic vega; every step is kept inside a
//! maintained bracket, and whenever Newton stalls (tiny vega deep ITM/OTM,
//! or a step that would leave the bracket) the algorithm bisects instead.
//! This makes it robust across moneyness 0.5x-2.0x and expiries from days
//! to years, mirroring the Python reference `eq_options.implied_vol`.

use crate::black_scholes::{
    bs_price, d1_d2, norm_pdf, validate_inputs, validate_rates, OptionType, PricingError,
};

/// Absolute price tolerance for the implied-vol solver.
pub const IV_PRICE_TOL: f64 = 1e-10;

/// Analytic vega used inside the Newton iteration.
fn bs_vega(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> Result<f64, PricingError> {
    let (d1, _) = d1_d2(s, k, t, r, sigma, q)?;
    Ok(s * (-q * t).exp() * norm_pdf(d1) * t.sqrt())
}

/// Implied Black-Scholes volatility from an observed premium.
///
/// Bracketed Newton (analytic vega) with bisection fallback; converges to
/// an absolute price tolerance of [`IV_PRICE_TOL`] (`1e-10`). Prices at or
/// below the `sigma -> 0` arbitrage bound (the discounted intrinsic on the
/// forward) or at/above the `sigma -> inf` bound are rejected — a
/// sub-intrinsic quote has no implied volatility.
///
/// # Arguments
///
/// * `price` — observed option premium, currency units; must lie strictly
///   between the no-arbitrage bounds.
/// * `s`, `k`, `t`, `r`, `q`, `option_type` — as in [`bs_price`];
///   `t`, `s`, `k` must be strictly positive.
///
/// # Errors
///
/// * [`PricingError::InvalidInput`] for NaN/negative inputs or `t == 0`.
/// * [`PricingError::ArbitrageBound`] for sub-intrinsic or above-bound prices.
/// * [`PricingError::NoConvergence`] if the root cannot be bracketed in
///   `[1e-9, 1e3]`. Once bracketed, the solver always terminates (Newton
///   with a bisection fallback and a final bracket-bisection refinement),
///   so this is never returned for an admissible, bracketable input.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, implied_vol, OptionType};
/// let ot = OptionType::Call;
/// let price = bs_price(100.0, 110.0, 0.5, 0.03, 0.25, 0.01, ot).unwrap();
/// let iv = implied_vol(price, 100.0, 110.0, 0.5, 0.03, 0.01, ot).unwrap();
/// assert!((iv - 0.25).abs() < 1e-8);
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
    const SIGMA_LO: f64 = 1e-9;
    const SIGMA_HI: f64 = 10.0;
    const MAX_ITER: usize = 200;

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
        OptionType::Call => s * (-q * t).exp(),
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

    let objective =
        |sig: f64| -> Result<f64, PricingError> { Ok(bs_price(s, k, t, r, sig, q, option_type)? - price) };

    let (mut lo, mut hi) = (SIGMA_LO, SIGMA_HI);
    let f_lo = objective(lo)?;
    let mut f_hi = objective(hi)?;
    // Expand the top of the bracket if needed (extremely high premiums).
    while f_hi < 0.0 && hi < 1e3 {
        hi *= 2.0;
        f_hi = objective(hi)?;
    }
    if f_lo > 0.0 || f_hi < 0.0 {
        return Err(PricingError::NoConvergence(
            "failed to bracket implied volatility in [1e-9, 1e3]".into(),
        ));
    }

    // Bracketed Newton: start from the midpoint, never leave [lo, hi];
    // fall back to bisection whenever vega is tiny or the step escapes.
    // Note this loop only ever `break`s (never returns `sigma` directly):
    // reaching `|diff| < IV_PRICE_TOL` is not by itself a safe stopping
    // rule -- see the bisection refinement below.
    let mut sigma = 0.5 * (lo + hi);
    for _ in 0..MAX_ITER {
        let diff = objective(sigma)?;
        if diff.abs() < IV_PRICE_TOL {
            break;
        }
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        let vega = bs_vega(s, k, t, r, sigma, q)?;
        let mut candidate = if vega > 1e-14 {
            sigma - diff / vega
        } else {
            f64::NAN
        };
        if !(candidate > lo && candidate < hi) {
            candidate = 0.5 * (lo + hi); // bisection fallback
        }
        if (candidate - sigma).abs() < 1e-16 {
            break;
        }
        sigma = candidate;
    }

    // Final safeguard: bisect the maintained bracket down to
    // double-precision width rather than trusting the `IV_PRICE_TOL` price
    // residual alone. In a flat-vega region (very long-dated + very high
    // vol, where d1/d2 blow up and vega ~ s sqrt(t) phi(d1) underflows
    // towards zero near the bracket's arbitrage bound) a tiny price
    // residual can map through that near-zero vega to a sigma residual of
    // whole vol points, so exiting the Newton loop the moment
    // `|diff| < IV_PRICE_TOL` risks returning an under-refined sigma.
    // Bisecting the bracket itself down to double-precision width is the
    // only stopping rule that is safe in that regime too, and costs at
    // most ~50 extra evaluations elsewhere. A valid bracket always
    // converges here, so this never fails to produce an answer -- the
    // `NoConvergence` variant remains reachable only from the earlier
    // bracket-expansion check.
    for _ in 0..200 {
        if hi - lo <= 1e-15 * hi.max(1.0) {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let f_mid = objective(mid)?;
        if f_mid > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}
