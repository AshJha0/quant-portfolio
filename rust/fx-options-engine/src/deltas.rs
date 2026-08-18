//! FX delta conventions: spot, forward, premium-adjusted, and inversions.
//!
//! Why four deltas?  FX options are quoted interbank in (delta, vol)
//! space, so the *meaning* of delta is part of the quote.  Two independent
//! choices:
//!
//! 1. **Spot vs forward hedge**: hedge with spot (delta includes the
//!    foreign discount factor) or with an outright forward.
//! 2. **Premium adjustment**: if the premium is paid in the *base*
//!    (foreign) currency — standard for USDJPY and most EM/non-EURUSD
//!    pairs — the premium itself is a position in the underlying and must
//!    be subtracted from the hedge.
//!
//! With `phi` = +1 call / -1 put and `F = S e^{(r_d - r_f)T}`:
//!
//! | Convention          | Delta                              |
//! |---------------------|------------------------------------|
//! | spot                | `phi e^{-r_f T} N(phi d1)`         |
//! | forward             | `phi N(phi d1)`                    |
//! | spot premium-adj    | `phi e^{-r_f T} (K/F) N(phi d2)`   |
//! | forward premium-adj | `phi (K/F) N(phi d2)`              |
//!
//! Relations: `delta_forward = delta_spot * e^{r_f T}` and
//! `delta_pa = delta_unadjusted - premium/S` (spot form).
//!
//! **Strike from delta**: analytic for the unadjusted conventions.  For
//! premium-adjusted *calls* the map K -> delta is **not monotone** (it
//! rises then falls; `(K/F)N(d2) -> 0` both as K -> 0 and K -> inf), so
//! the equation has zero, one, or two solutions.  Market convention takes
//! the solution on the *right* (decreasing) branch, i.e. the larger
//! strike — implemented here by locating the peak of `K N(d2)` and
//! Brent-solving on `[K_peak, K_max]`.  Premium-adjusted put deltas are
//! monotone in K and Brent-solve directly.
//!
//! **ATM conventions**:
//!
//! * ATM-forward: `K = F`.
//! * ATM delta-neutral straddle (DNS): strike where call delta + put delta
//!   = 0 under the pair's delta convention: `K = F e^{+sigma^2 T/2}` for
//!   unadjusted deltas (d1 = 0), `K = F e^{-sigma^2 T/2}` for
//!   premium-adjusted deltas (d2 = 0).

use crate::garman_kohlhagen::d1 as d1_fn;
use crate::{
    brentq, invalid, norm_cdf, norm_pdf, norm_ppf, require_finite, validate_inputs, FxResult,
    OptionType,
};

/// FX delta quoting convention (see the module docs for the formulae).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeltaConvention {
    /// Spot delta: hedge with spot, premium in domestic (quote) ccy.
    Spot,
    /// Forward delta: hedge with an outright forward.
    Forward,
    /// Premium-adjusted spot delta (premium paid in base ccy).
    SpotPa,
    /// Premium-adjusted forward delta (premium paid in base ccy).
    ForwardPa,
}

impl DeltaConvention {
    /// All four conventions, for grid tests and iteration.
    pub const ALL: [DeltaConvention; 4] = [
        DeltaConvention::Spot,
        DeltaConvention::Forward,
        DeltaConvention::SpotPa,
        DeltaConvention::ForwardPa,
    ];

    /// `true` for the premium-adjusted (`*Pa`) conventions.
    #[inline]
    pub const fn is_premium_adjusted(self) -> bool {
        matches!(self, DeltaConvention::SpotPa | DeltaConvention::ForwardPa)
    }
}

/// FX option delta under a chosen quoting convention.
///
/// # Arguments
///
/// * `s`, `k`, `t`, `r_d`, `r_f`, `sigma` — as in [`crate::gk_price`];
///   requires `t > 0` and `sigma > 0`.
/// * `option_type` — call or put on the base currency.
/// * `convention` — one of the four [`DeltaConvention`]s.
///   Premium-adjusted deltas assume the premium is paid in the base
///   (foreign) currency.
///
/// Returns the delta in units of foreign notional (per 1 unit of foreign
/// notional of the option).
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs or `t = 0` /
/// `sigma = 0`.
///
/// ```
/// use fx_options_engine::{delta, DeltaConvention, OptionType};
/// let (s, k, t, rd, rf, vol) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);
/// let ds = delta(s, k, t, rd, rf, vol, OptionType::Call, DeltaConvention::Spot).unwrap();
/// let df = delta(s, k, t, rd, rf, vol, OptionType::Call, DeltaConvention::Forward).unwrap();
/// assert!((df - ds * (rf * t).exp()).abs() < 1e-15);
/// ```
pub fn delta(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
    convention: DeltaConvention,
) -> FxResult<f64> {
    let phi = option_type.phi();
    let d1v = d1_fn(s, k, t, r_d, r_f, sigma)?; // validates s, k, t, sigma
    let d2v = d1v - sigma * t.sqrt();
    let f = s * ((r_d - r_f) * t).exp();
    Ok(match convention {
        DeltaConvention::Spot => phi * (-r_f * t).exp() * norm_cdf(phi * d1v),
        DeltaConvention::Forward => phi * norm_cdf(phi * d1v),
        DeltaConvention::SpotPa => phi * (-r_f * t).exp() * (k / f) * norm_cdf(phi * d2v),
        DeltaConvention::ForwardPa => phi * (k / f) * norm_cdf(phi * d2v),
    })
}

/// Convert spot delta to forward delta: `delta_f = delta_s e^{r_f T}`.
///
/// Holds for both plain and premium-adjusted forms.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on non-finite or negative `t`.
pub fn spot_to_forward_delta(delta_spot: f64, t: f64, r_f: f64) -> FxResult<f64> {
    validate_inputs(1.0, 1.0, t, 0.0, r_f, 0.0)?;
    Ok(delta_spot * (r_f * t).exp())
}

/// Convert forward delta to spot delta: `delta_s = delta_f e^{-r_f T}`.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on non-finite or negative `t`.
pub fn forward_to_spot_delta(delta_forward: f64, t: f64, r_f: f64) -> FxResult<f64> {
    validate_inputs(1.0, 1.0, t, 0.0, r_f, 0.0)?;
    Ok(delta_forward * (-r_f * t).exp())
}

/// Premium-adjust a spot delta: `delta_pa = delta_spot - V/S`.
///
/// `price` is the domestic-currency premium; `V/S` is the premium
/// converted to base currency, which is itself a long-base position the
/// hedger already holds and therefore does not need to buy.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] if `s` is not positive and finite.
pub fn premium_adjust_spot_delta(delta_spot: f64, price: f64, s: f64) -> FxResult<f64> {
    if !(s > 0.0) || !s.is_finite() {
        return invalid(format!("Spot S must be positive and finite, got {s}"));
    }
    Ok(delta_spot - price / s)
}

/// ATM-forward strike: `K = F = S e^{(r_d - r_f) T}`.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs.
pub fn atm_forward_strike(s: f64, t: f64, r_d: f64, r_f: f64) -> FxResult<f64> {
    validate_inputs(s, s, t, r_d, r_f, 0.0)?;
    Ok(s * ((r_d - r_f) * t).exp())
}

/// ATM delta-neutral-straddle strike under a delta convention.
///
/// `K = F e^{+sigma^2 T/2}` for unadjusted deltas (call + put spot or
/// forward delta vanishes at d1 = 0), `K = F e^{-sigma^2 T/2}` for
/// premium-adjusted deltas (vanishes at d2 = 0).  This is the strike at
/// which the market quotes "ATM" vol for most pairs.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs.
///
/// ```
/// use fx_options_engine::{atm_dns_strike, delta, DeltaConvention, OptionType};
/// let (s, t, rd, rf, vol) = (1.10, 0.5, 0.0425, 0.0290, 0.0925);
/// let k = atm_dns_strike(s, t, rd, rf, vol, DeltaConvention::SpotPa).unwrap();
/// let dc = delta(s, k, t, rd, rf, vol, OptionType::Call, DeltaConvention::SpotPa).unwrap();
/// let dp = delta(s, k, t, rd, rf, vol, OptionType::Put, DeltaConvention::SpotPa).unwrap();
/// assert!((dc + dp).abs() < 1e-14); // straddle delta vanishes
/// ```
pub fn atm_dns_strike(
    s: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    convention: DeltaConvention,
) -> FxResult<f64> {
    validate_inputs(s, s, t, r_d, r_f, sigma)?;
    let f = s * ((r_d - r_f) * t).exp();
    let sign = if convention.is_premium_adjusted() {
        -1.0
    } else {
        1.0
    };
    Ok(f * (sign * 0.5 * sigma * sigma * t).exp())
}

/// Strike maximising `K*N(d2(K))` — the fold point of the PA call delta.
///
/// Setting `d/dK [K N(d2)] = 0` gives `N(d2) sigma sqrt(T) = n(d2)`, a
/// one-dimensional root in d2 (unique: LHS increasing, RHS log-concave
/// with a single crossing), then
/// `K = F exp(-d2 sigma sqrt(T) - sigma^2 T / 2)`.
fn pa_peak_strike(f: f64, t: f64, sigma: f64) -> FxResult<f64> {
    let v = sigma * t.sqrt();
    let g = |x: f64| norm_cdf(x) * v - norm_pdf(x);
    let root = brentq(g, -20.0, 20.0, 1e-14, 200)?;
    Ok(f * (-root * v - 0.5 * v * v).exp())
}

/// Invert delta -> strike under any of the four conventions.
///
/// Unadjusted conventions are analytic:
/// `K = F exp(-phi z sigma sqrt(T) + sigma^2 T / 2)` with
/// `z = N^{-1}(phi delta e^{r_f T})` (spot) or `N^{-1}(phi delta)`
/// (forward).  Premium-adjusted conventions are solved with Brent.
///
/// For **premium-adjusted calls** the delta-to-strike map is not
/// injective; per market convention the root on the decreasing branch
/// (the larger strike, the one consistent with OTM quoting) is returned.
/// A `target_delta` above the fold's maximum attainable PA delta is an
/// error.
///
/// # Arguments
///
/// * `target_delta` — desired delta, signed (calls in `(0, upper)`, puts
///   in `(lower, 0)`).
/// * `s`, `t`, `r_d`, `r_f`, `sigma` — market inputs; requires `t > 0`,
///   `sigma > 0`.
/// * `option_type` — call or put.
/// * `convention` — one of the four [`DeltaConvention`]s.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] if the delta is out of the attainable
/// range for the convention or the market inputs are invalid.
///
/// ```
/// use fx_options_engine::{delta, strike_from_delta, DeltaConvention, OptionType};
/// let (s, t, rd, rf, vol) = (1.10, 0.5, 0.0425, 0.0290, 0.0925);
/// let k = strike_from_delta(-0.25, s, t, rd, rf, vol,
///                           OptionType::Put, DeltaConvention::Spot).unwrap();
/// let d = delta(s, k, t, rd, rf, vol, OptionType::Put,
///               DeltaConvention::Spot).unwrap();
/// assert!((d + 0.25).abs() < 1e-10);
/// ```
pub fn strike_from_delta(
    target_delta: f64,
    s: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
    convention: DeltaConvention,
) -> FxResult<f64> {
    let phi = option_type.phi();
    validate_inputs(s, s, t, r_d, r_f, sigma)?;
    if t <= 0.0 || sigma <= 0.0 {
        return invalid("strike_from_delta requires T > 0 and sigma > 0");
    }
    require_finite(target_delta, "target_delta")?;
    if phi * target_delta <= 0.0 {
        return invalid(format!(
            "{option_type} delta must have sign {}, got {target_delta}",
            phi as i32
        ));
    }

    let f = s * ((r_d - r_f) * t).exp();
    let v = sigma * t.sqrt();

    if matches!(convention, DeltaConvention::Spot | DeltaConvention::Forward) {
        let fwd_delta = if convention == DeltaConvention::Spot {
            target_delta * (r_f * t).exp()
        } else {
            target_delta
        };
        if !(phi * fwd_delta > 0.0 && phi * fwd_delta < 1.0) {
            return invalid(format!(
                "forward-equivalent delta {fwd_delta} outside (0, 1) range"
            ));
        }
        let z = norm_ppf(phi * fwd_delta); // z = phi * d1
        return Ok(f * (-phi * z * v + 0.5 * v * v).exp());
    }

    // Premium-adjusted: solve phi (K/F) N(phi d2(K)) = fwd-equivalent delta.
    let fwd_delta = if convention == DeltaConvention::SpotPa {
        target_delta * (r_f * t).exp()
    } else {
        target_delta
    };

    let pa_fwd_delta = |k: f64| {
        let d2v = ((f / k).ln() - 0.5 * v * v) / v;
        phi * (k / f) * norm_cdf(phi * d2v)
    };

    if phi > 0.0 {
        // Call: non-monotone; use the decreasing branch [K_peak, inf).
        let k_peak = pa_peak_strike(f, t, sigma)?;
        let max_delta = pa_fwd_delta(k_peak);
        if fwd_delta > max_delta + 1e-14 {
            let scale = if convention == DeltaConvention::SpotPa {
                (-r_f * t).exp()
            } else {
                1.0
            };
            return invalid(format!(
                "premium-adjusted call delta {target_delta} exceeds the maximum \
                 attainable {:.6} for these market inputs",
                max_delta * scale
            ));
        }
        let k_max = f * (30.0 * v).exp();
        let mut k_hi = k_peak;
        while pa_fwd_delta(k_hi) > fwd_delta && k_hi < k_max {
            k_hi *= 2.0;
        }
        return brentq(|k| pa_fwd_delta(k) - fwd_delta, k_peak, k_hi, 1e-14, 200);
    }

    // Put: |PA delta| strictly increasing in K -> unique root.
    if !(fwd_delta > -1.0 && fwd_delta < 0.0) {
        return invalid(format!(
            "premium-adjusted put forward delta {fwd_delta} outside (-1, 0)"
        ));
    }
    let k_lo = f * (-30.0 * v).exp();
    let k_max = f * (30.0 * v).exp();
    let mut k_hi = f;
    while pa_fwd_delta(k_hi) > fwd_delta && k_hi < k_max {
        k_hi *= 2.0;
    }
    brentq(|k| pa_fwd_delta(k) - fwd_delta, k_lo, k_hi, 1e-14, 200)
}
