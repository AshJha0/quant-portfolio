//! Analytic Black-Scholes Greeks and finite-difference Greeks for any pricer.
//!
//! # Units and conventions (identical to the Python reference)
//!
//! * `delta` : dV/dS, dimensionless (per unit of spot).
//! * `gamma` : d2V/dS2, per currency unit.
//! * `vega`  : dV/dsigma, currency units per **unit** of annualised vol
//!   (divide by 100 for the market's "per vol point").
//! * `theta` : dV/dt, currency units per **year** (divide by 365 for per-day).
//! * `rho`   : dV/dr, currency units per **unit** of rate.
//! * `vanna` : d2V/(dS dsigma).
//! * `volga` : d2V/dsigma2 (a.k.a. vomma).
//!
//! All rates continuously compounded, annualised; `t` in years (ACT/365F).

use crate::black_scholes::{
    d1_d2, norm_cdf, norm_pdf, validate_inputs, validate_rates, OptionType, PricingError,
};

/// Full Greek set of a European option (units in the module docs).
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_greeks, OptionType};
/// let g = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
/// assert!((g.delta - 0.6368306511756191).abs() < 1e-12);
/// assert!(g.gamma > 0.0 && g.vega > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsGreeks {
    /// Present value, currency units.
    pub price: f64,
    /// dV/dS, dimensionless.
    pub delta: f64,
    /// d2V/dS2, per currency unit.
    pub gamma: f64,
    /// dV/dsigma, per unit of annualised vol.
    pub vega: f64,
    /// dV/dt, per year.
    pub theta: f64,
    /// dV/dr, per unit of rate.
    pub rho: f64,
    /// d2V/(dS dsigma).
    pub vanna: f64,
    /// d2V/dsigma2.
    pub volga: f64,
}

/// Analytic Black-Scholes-Merton Greeks (with continuous dividend yield).
///
/// # Errors
///
/// [`PricingError::InvalidInput`] unless `s`, `k`, `t`, `sigma` are
/// strictly positive (the Greeks are singular at the boundary).
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_greeks, OptionType};
/// let call = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
/// let put = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put).unwrap();
/// // Same gamma and vega for call and put:
/// assert!((call.gamma - put.gamma).abs() < 1e-15);
/// assert!((call.vega - put.vega).abs() < 1e-15);
/// ```
pub fn bs_greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> Result<BsGreeks, PricingError> {
    let (d1, d2) = d1_d2(s, k, t, r, sigma, q)?;
    let sqrt_t = t.sqrt();
    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();
    let pdf_d1 = norm_pdf(d1);

    let gamma = df_q * pdf_d1 / (s * sigma * sqrt_t);
    let vega = s * df_q * pdf_d1 * sqrt_t;
    let vanna = -df_q * pdf_d1 * d2 / sigma;
    let volga = vega * d1 * d2 / sigma;
    let common_theta = -s * df_q * pdf_d1 * sigma / (2.0 * sqrt_t);

    let (price, delta, theta, rho) = match option_type {
        OptionType::Call => {
            let nd1 = norm_cdf(d1);
            let nd2 = norm_cdf(d2);
            (
                s * df_q * nd1 - k * df_r * nd2,
                df_q * nd1,
                common_theta + q * s * df_q * nd1 - r * k * df_r * nd2,
                k * t * df_r * nd2,
            )
        }
        OptionType::Put => {
            let nmd1 = norm_cdf(-d1);
            let nmd2 = norm_cdf(-d2);
            (
                k * df_r * nmd2 - s * df_q * nmd1,
                -df_q * nmd1,
                common_theta - q * s * df_q * nmd1 + r * k * df_r * nmd2,
                -k * t * df_r * nmd2,
            )
        }
    };

    Ok(BsGreeks {
        price,
        delta,
        gamma,
        vega,
        theta,
        rho,
        vanna,
        volga,
    })
}

/// Central finite-difference Greeks for *any* pricer with the BS signature.
///
/// The pricer is a closure over `(s, k, t, r, sigma, q, option_type)`
/// returning a price (`Result`), e.g. [`crate::bs_price`] itself or a
/// closure around [`crate::crr_price`]. Central differences are used
/// everywhere; second derivatives use the three-point stencil; vanna uses
/// the four-point cross stencil. Theta is reported as `dV/dt = -dV/dT`
/// (per year).
///
/// Bump sizes: first derivatives use `rel_bump * max(|x|, 1)` with
/// `rel_bump = 1e-5` (truncation vs round-off balance for analytic
/// pricers); second derivatives use `rel_bump2 = 2e-4` (round-off scales
/// like `eps / h^2`, optimal `h ~ eps^0.25`). These match the Python
/// reference `fd_greeks` defaults.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] on invalid inputs, or if `t`, `sigma`
/// or `s` is too small for the down-leg of the central bump to stay
/// inside the domain (`t - h_t <= 0`, `sigma - h_v2 <= 0`,
/// `s - h_s2 <= 0`) — this is checked up front so that a pricer closure
/// which does *not* validate its own arguments can never be evaluated at
/// a negative volatility or a negative spot. Any error raised by the
/// pricer itself is propagated unchanged.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, bs_greeks, fd_greeks, OptionType};
/// let ot = OptionType::Call;
/// let fd = fd_greeks(bs_price, 100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot).unwrap();
/// let an = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot).unwrap();
/// assert!((fd.delta - an.delta).abs() < 1e-6);
/// assert!((fd.vega - an.vega).abs() < 1e-6 * an.vega.abs().max(1.0));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn fd_greeks<F>(
    pricer: F,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> Result<BsGreeks, PricingError>
where
    F: Fn(f64, f64, f64, f64, f64, f64, OptionType) -> Result<f64, PricingError>,
{
    const REL_BUMP: f64 = 1e-5;
    const REL_BUMP2: f64 = 2e-4;

    validate_inputs(s, k, t, sigma)?;
    validate_rates(r, q)?;

    let f = |sv: f64, sig: f64, tv: f64, rv: f64| pricer(sv, k, tv, rv, sig, q, option_type);

    let h_s = REL_BUMP * s.abs().max(1.0);
    let h_v = REL_BUMP * sigma.abs().max(1.0);
    let h_t = REL_BUMP * t.abs().max(1.0);
    let h_r = REL_BUMP * r.abs().max(1.0);
    let h_s2 = REL_BUMP2 * s.abs().max(1.0);
    let h_v2 = REL_BUMP2 * sigma.abs().max(1.0);
    if t - h_t <= 0.0 {
        return Err(PricingError::InvalidInput(format!(
            "T={t} too small for a central theta bump of {h_t}"
        )));
    }
    // A central bump must stay inside the domain. Because the bumps are
    // floored at `rel_bump * 1`, a spot or vol below the bump would be
    // pushed negative by the down-leg. `bs_price` would reject that, but a
    // user-supplied pricer closure need not: without this guard the
    // returned "vega" could be a difference of prices evaluated at a
    // *negative* volatility, which is silently meaningless. Reject up
    // front instead. `h_*2 >= h_*`, so guarding the second-order bump
    // covers the first-order one too.
    if sigma - h_v2 <= 0.0 {
        return Err(PricingError::InvalidInput(format!(
            "sigma={sigma} too small for a central vega/volga bump of {h_v2}; \
             the down-leg would price at a negative volatility"
        )));
    }
    if s - h_s2 <= 0.0 {
        return Err(PricingError::InvalidInput(format!(
            "S={s} too small for a central delta/gamma bump of {h_s2}; \
             the down-leg would price at a negative spot"
        )));
    }

    let price = f(s, sigma, t, r)?;
    let delta = (f(s + h_s, sigma, t, r)? - f(s - h_s, sigma, t, r)?) / (2.0 * h_s);
    let gamma =
        (f(s + h_s2, sigma, t, r)? - 2.0 * price + f(s - h_s2, sigma, t, r)?) / (h_s2 * h_s2);
    let vega = (f(s, sigma + h_v, t, r)? - f(s, sigma - h_v, t, r)?) / (2.0 * h_v);
    let theta = -(f(s, sigma, t + h_t, r)? - f(s, sigma, t - h_t, r)?) / (2.0 * h_t);
    let rho = (f(s, sigma, t, r + h_r)? - f(s, sigma, t, r - h_r)?) / (2.0 * h_r);
    let vanna = (f(s + h_s2, sigma + h_v2, t, r)? - f(s + h_s2, sigma - h_v2, t, r)?
        - f(s - h_s2, sigma + h_v2, t, r)?
        + f(s - h_s2, sigma - h_v2, t, r)?)
        / (4.0 * h_s2 * h_v2);
    let volga =
        (f(s, sigma + h_v2, t, r)? - 2.0 * price + f(s, sigma - h_v2, t, r)?) / (h_v2 * h_v2);

    Ok(BsGreeks {
        price,
        delta,
        gamma,
        vega,
        theta,
        rho,
        vanna,
        volga,
    })
}
