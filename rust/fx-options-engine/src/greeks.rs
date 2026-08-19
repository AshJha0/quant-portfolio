//! Analytic Garman–Kohlhagen Greeks, including both rhos and vanna/volga.
//!
//! FX specifics:
//!
//! * **Two rhos.**  An FX option has rate sensitivity to *both* legs:
//!   `rho_d = dV/dr_d` (positive for calls — higher domestic rate lifts
//!   the forward) and `rho_f = dV/dr_f` (negative for calls — higher
//!   foreign rate is a larger "dividend" on the base currency).
//! * **Vanna and volga.**  FX desks mark smiles with risk reversals and
//!   butterflies, whose P&L maps directly onto vanna (dDelta/dVol) and
//!   volga (dVega/dVol).  A vanilla book's smile risk is quoted in these
//!   buckets, so they are first-class here.
//!
//! All Greeks are per unit foreign notional, prices in domestic currency.
//! Theta is per year (divide by 365 for a daily theta); vega is per unit
//! of vol (divide by 100 for "per vol point").

use crate::garman_kohlhagen::{d1 as d1_fn, gk_price};
use crate::{invalid, norm_cdf, norm_pdf, validate_inputs, FxResult, OptionType};

/// Full GK Greek set for one option.
///
/// `delta_spot`/`delta_forward` are unadjusted deltas (see
/// [`crate::deltas`] for premium-adjusted variants).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreeksResult {
    /// Premium, domestic ccy per unit foreign notional.
    pub price: f64,
    /// Unadjusted spot delta `phi e^{-r_f T} N(phi d1)`.
    pub delta_spot: f64,
    /// Unadjusted forward delta `phi N(phi d1)`.
    pub delta_forward: f64,
    /// Spot gamma `d2V/dS2`.
    pub gamma: f64,
    /// Vega `dV/dsigma` (per unit of vol; call = put).
    pub vega: f64,
    /// Theta `dV/dt` in calendar time, per year.
    pub theta: f64,
    /// Domestic rho `dV/dr_d`.
    pub rho_domestic: f64,
    /// Foreign rho `dV/dr_f`.
    pub rho_foreign: f64,
    /// Vanna `d2V/(dS dsigma)`.
    pub vanna: f64,
    /// Volga `d2V/dsigma2`.
    pub volga: f64,
}

fn core(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<(f64, f64, f64, f64)> {
    let d1v = d1_fn(s, k, t, r_d, r_f, sigma)?;
    let d2v = d1v - sigma * t.sqrt();
    Ok((d1v, d2v, (-r_f * t).exp(), (-r_d * t).exp()))
}

/// Spot gamma `d2V/dS2 = e^{-r_f T} n(d1) / (S sigma sqrt(T))`.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs (requires `t > 0`,
/// `sigma > 0`).
pub fn gamma(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    let (d1v, _, df_f, _) = core(s, k, t, r_d, r_f, sigma)?;
    Ok(df_f * norm_pdf(d1v) / (s * sigma * t.sqrt()))
}

/// Vega `dV/dsigma = S e^{-r_f T} n(d1) sqrt(T)` (call = put).
///
/// # Errors
///
/// Same as [`gamma`].
pub fn vega(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    let (d1v, _, df_f, _) = core(s, k, t, r_d, r_f, sigma)?;
    Ok(s * df_f * norm_pdf(d1v) * t.sqrt())
}

/// Vanna `d2V/(dS dsigma) = -e^{-r_f T} n(d1) d2 / sigma`.
///
/// The sensitivity a 25-delta risk reversal position monetises.
///
/// # Errors
///
/// Same as [`gamma`].
pub fn vanna(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    let (d1v, d2v, df_f, _) = core(s, k, t, r_d, r_f, sigma)?;
    Ok(-df_f * norm_pdf(d1v) * d2v / sigma)
}

/// Volga `d2V/dsigma2 = vega * d1 * d2 / sigma`.
///
/// The sensitivity a 25-delta butterfly position monetises.
///
/// # Errors
///
/// Same as [`gamma`].
pub fn volga(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> FxResult<f64> {
    let (d1v, d2v, _, _) = core(s, k, t, r_d, r_f, sigma)?;
    Ok(vega(s, k, t, r_d, r_f, sigma)? * d1v * d2v / sigma)
}

/// Closed-form GK Greeks.
///
/// # Arguments
///
/// * `s`, `k`, `t`, `r_d`, `r_f`, `sigma` — as in [`crate::gk_price`];
///   requires `t > 0` and `sigma > 0`.
/// * `option_type` — call or put on the base currency.
///
/// Returns a [`GreeksResult`] with price, delta_spot, delta_forward,
/// gamma, vega, theta (per year), rho_domestic, rho_foreign, vanna,
/// volga.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs or `t = 0` /
/// `sigma = 0`.
///
/// ```
/// use fx_options_engine::{analytic_greeks, OptionType};
/// let g = analytic_greeks(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925,
///                         OptionType::Call).unwrap();
/// assert!(g.rho_domestic > 0.0 && g.rho_foreign < 0.0); // call rho signs
/// assert!(g.gamma > 0.0 && g.vega > 0.0);
/// ```
pub fn analytic_greeks(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
) -> FxResult<GreeksResult> {
    let phi = option_type.phi();
    validate_inputs(s, k, t, r_d, r_f, sigma)?;
    if t <= 0.0 || sigma <= 0.0 {
        return invalid("analytic_greeks requires T > 0 and sigma > 0");
    }
    let (d1v, d2v, df_f, df_d) = core(s, k, t, r_d, r_f, sigma)?;
    let sqrt_t = t.sqrt();
    let n_d1 = norm_pdf(d1v);
    let nc_pd1 = norm_cdf(phi * d1v);
    let nc_pd2 = norm_cdf(phi * d2v);

    let price = phi * (s * df_f * nc_pd1 - k * df_d * nc_pd2);
    let delta_spot = phi * df_f * nc_pd1;
    let theta = -s * df_f * n_d1 * sigma / (2.0 * sqrt_t)
        + phi * (r_f * s * df_f * nc_pd1 - r_d * k * df_d * nc_pd2);
    Ok(GreeksResult {
        price,
        delta_spot,
        delta_forward: phi * nc_pd1,
        gamma: df_f * n_d1 / (s * sigma * sqrt_t),
        vega: s * df_f * n_d1 * sqrt_t,
        theta,
        rho_domestic: phi * k * t * df_d * nc_pd2,
        rho_foreign: -phi * s * t * df_f * nc_pd1,
        vanna: -df_f * n_d1 * d2v / sigma,
        volga: s * df_f * n_d1 * sqrt_t * d1v * d2v / sigma,
    })
}

/// Finite-difference Greeks (see [`finite_difference_greeks`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdGreeks {
    /// Central-difference spot delta.
    pub delta_spot: f64,
    /// Second central difference in spot.
    pub gamma: f64,
    /// Central-difference vega.
    pub vega: f64,
    /// `-dV/dT` (central difference in calendar time).
    pub theta: f64,
    /// Central-difference domestic rho.
    pub rho_domestic: f64,
    /// Central-difference foreign rho.
    pub rho_foreign: f64,
    /// Mixed central stencil `d2V/(dS dsigma)`.
    pub vanna: f64,
    /// Second central difference in vol.
    pub volga: f64,
}

/// Generic first-order central difference `(f(x+h) - f(x-h)) / 2h` for
/// any fallible pricing function — the building block of the FD
/// comparator, usable against *any* model function in this crate.
///
/// # Errors
///
/// Propagates errors from `f`.
///
/// ```
/// use fx_options_engine::greeks::{central_difference, vega};
/// use fx_options_engine::gk_call;
/// let (s, k, t, rd, rf, vol) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);
/// let fd = central_difference(|sig| gk_call(s, k, t, rd, rf, sig), vol, 1e-6)
///     .unwrap();
/// assert!((fd - vega(s, k, t, rd, rf, vol).unwrap()).abs() < 1e-8);
/// ```
pub fn central_difference<F>(f: F, x: f64, h: f64) -> FxResult<f64>
where
    F: Fn(f64) -> FxResult<f64>,
{
    Ok((f(x + h)? - f(x - h)?) / (2.0 * h))
}

/// Generic second-order central difference
/// `(f(x+h) - 2 f(x) + f(x-h)) / h^2`.
///
/// # Errors
///
/// Propagates errors from `f`.
pub fn second_central_difference<F>(f: F, x: f64, h: f64) -> FxResult<f64>
where
    F: Fn(f64) -> FxResult<f64>,
{
    Ok((f(x + h)? - 2.0 * f(x)? + f(x - h)?) / (h * h))
}

/// Central finite-difference Greeks for validating the analytic set.
///
/// Uses relative bumps of size `rel_bump` on `s` and `sigma`, absolute
/// bumps on rates, and a central difference in calendar time for theta
/// (`theta = -dV/dT`).  Second-order Greeks (gamma, vanna, volga) use the
/// standard central stencils.  Mirrors the Python
/// `fx_options.greeks.finite_difference_greeks` exactly.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs, on `t <= 0` or
/// `sigma <= 0` (the Greeks are singular there, and the theta bump
/// `min(1e-6, T/4)` would collapse to zero at `T = 0`, making the
/// difference quotient an unreported `0/0`), or on a `rel_bump` that is
/// not finite and strictly positive (`rel_bump = 0` would divide by a
/// zero bump; `rel_bump = NaN` would return a full set of NaN Greeks,
/// which no downstream limit check can detect).
pub fn finite_difference_greeks(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
    rel_bump: f64,
) -> FxResult<FdGreeks> {
    validate_inputs(s, k, t, r_d, r_f, sigma)?;
    if t <= 0.0 || sigma <= 0.0 {
        return invalid(format!(
            "finite_difference_greeks requires T > 0 and sigma > 0, got T={t}, sigma={sigma}"
        ));
    }
    if !(rel_bump > 0.0) || !rel_bump.is_finite() {
        return invalid(format!(
            "rel_bump must be finite and positive, got {rel_bump}"
        ));
    }
    let h_s = s * rel_bump;
    let h_v = (sigma * rel_bump).max(1e-7);
    let h_r = 1e-6;
    let h_t = (1e-6_f64).min(t / 4.0);

    let p = |s_: f64, sig: f64, rd: f64, rf: f64, t_: f64| {
        gk_price(s_, k, t_, rd, rf, sig, option_type)
    };

    // Larger bump for the sigma second difference: with h ~ sigma*1e-5 the
    // O(eps/h^2) round-off term dominates; sigma*1e-3 balances round-off
    // against the O(h^2) truncation error.
    let h_v2 = (sigma * 1e-3).max(1e-5);
    let base = p(s, sigma, r_d, r_f, t)?;
    let (up_s, dn_s) = (p(s + h_s, sigma, r_d, r_f, t)?, p(s - h_s, sigma, r_d, r_f, t)?);
    let (up_v, dn_v) = (p(s, sigma + h_v, r_d, r_f, t)?, p(s, sigma - h_v, r_d, r_f, t)?);
    let (up_v2, dn_v2) = (
        p(s, sigma + h_v2, r_d, r_f, t)?,
        p(s, sigma - h_v2, r_d, r_f, t)?,
    );
    Ok(FdGreeks {
        delta_spot: (up_s - dn_s) / (2.0 * h_s),
        gamma: (up_s - 2.0 * base + dn_s) / (h_s * h_s),
        vega: (up_v - dn_v) / (2.0 * h_v),
        theta: -(p(s, sigma, r_d, r_f, t + h_t)? - p(s, sigma, r_d, r_f, t - h_t)?) / (2.0 * h_t),
        rho_domestic: (p(s, sigma, r_d + h_r, r_f, t)? - p(s, sigma, r_d - h_r, r_f, t)?)
            / (2.0 * h_r),
        rho_foreign: (p(s, sigma, r_d, r_f + h_r, t)? - p(s, sigma, r_d, r_f - h_r, t)?)
            / (2.0 * h_r),
        vanna: (p(s + h_s, sigma + h_v, r_d, r_f, t)? - p(s + h_s, sigma - h_v, r_d, r_f, t)?
            - p(s - h_s, sigma + h_v, r_d, r_f, t)?
            + p(s - h_s, sigma - h_v, r_d, r_f, t)?)
            / (4.0 * h_s * h_v),
        volga: (up_v2 - 2.0 * base + dn_v2) / (h_v2 * h_v2),
    })
}
