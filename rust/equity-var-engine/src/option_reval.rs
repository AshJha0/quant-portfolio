//! Option revaluation under VaR scenarios.
//!
//! Black-Scholes-Merton with a continuous carry yield (dividend yield for
//! equity, foreign rate for FX, which gives Garman-Kohlhagen), closed-form
//! Greeks (Hull, *Options, Futures and Other Derivatives*, 11th ed., ch. 19),
//! full revaluation P&L per scenario and the delta-gamma approximation.
//! Mirrors `eq_var.revaluation`; the golden values in
//! `tests/test_option_reval.rs` come from that module.
//!
//! Scenario layout: row-major `n_scenarios x n_factors` spot log returns;
//! vol shocks (absolute, decimal) in the same layout, or empty for none.
//!
//! [`historical_option_var_es`] and [`monte_carlo_option_run`] wire the
//! pricer into historical simulation and Monte Carlo: every scenario reprices
//! every option leg in full (Jorion, *Value at Risk*, 3rd ed., ch. 10). A book
//! with no option legs goes down the unchanged linear path.

use crate::matrix::Matrix;
use crate::monte_carlo::{
    monte_carlo_run, simulate_factor_returns_jitter, var_bootstrap_se, var_kde_se, McRun, MIN_PATHS,
};
use crate::stats::{normal_cdf as ncdf, normal_pdf as npdf};
use crate::{validate_alpha, EqVarError, Result, TailModel};

/// Call or put.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionKind {
    /// European call.
    Call,
    /// European put.
    Put,
}

/// European option on scenario factor `factor`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionLeg {
    /// Signed number of contracts.
    pub quantity: f64,
    /// Call or put.
    pub kind: OptionKind,
    /// Spot today, `> 0`.
    pub spot: f64,
    /// Strike, `> 0`.
    pub strike: f64,
    /// Years, `> 0`.
    pub tau: f64,
    /// Domestic rate, continuously compounded.
    pub rate: f64,
    /// Dividend yield or foreign rate.
    pub carry: f64,
    /// Annualised implied vol, `> 0`.
    pub vol: f64,
    /// Column of the scenario matrix that moves this leg's spot.
    pub factor: usize,
}

/// Greeks per unit quantity; vega, vanna and volga per 1.00 of vol, theta = `-dV/dtau`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsmGreeks {
    /// `dV/dS`.
    pub delta: f64,
    /// `d2V/dS2`.
    pub gamma: f64,
    /// `dV/dsigma`.
    pub vega: f64,
    /// `d2V/dS dsigma`.
    pub vanna: f64,
    /// `d2V/dsigma2`.
    pub volga: f64,
    /// `-dV/dtau` per year.
    pub theta: f64,
}

fn require(ok: bool, what: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(EqVarError::InvalidInput(format!("option_reval: {what}")))
    }
}

fn check_leg(l: &OptionLeg) -> Result<()> {
    require(
        l.quantity.is_finite() && l.rate.is_finite() && l.carry.is_finite(),
        "quantity, rate and carry must be finite",
    )?;
    require(l.spot.is_finite() && l.spot > 0.0, "spot must be > 0")?;
    require(l.strike.is_finite() && l.strike > 0.0, "strike must be > 0")?;
    require(l.tau.is_finite() && l.tau > 0.0, "tau must be > 0")?;
    require(l.vol.is_finite() && l.vol > 0.0, "vol must be > 0")
}

fn check_scenarios(legs: &[OptionLeg], x: &[f64], dv: &[f64], nf: usize) -> Result<usize> {
    require(!legs.is_empty(), "legs must not be empty")?;
    require(
        nf > 0 && x.len().is_multiple_of(nf),
        "scenario size must be a multiple of n_factors",
    )?;
    require(
        dv.is_empty() || dv.len() == x.len(),
        "vol_shocks must match spot_log_returns",
    )?;
    for l in legs {
        check_leg(l)?;
        require(l.factor < nf, "leg factor index out of range")?;
    }
    require(
        x.iter().all(|v| v.is_finite()),
        "spot_log_returns must be finite",
    )?;
    require(
        dv.iter().all(|v| v.is_finite()),
        "vol_shocks must be finite",
    )?;
    Ok(x.len() / nf)
}

fn sign(kind: OptionKind) -> f64 {
    match kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    }
}

fn price_unchecked(s_: f64, k: f64, t: f64, r: f64, b: f64, v: f64, kind: OptionKind) -> f64 {
    let s = sign(kind);
    if t <= 0.0 {
        return (s * (s_ - k)).max(0.0);
    }
    let v = v.max(1e-12);
    let sq = v * t.sqrt();
    let d1 = ((s_ / k).ln() + (r - b + 0.5 * v * v) * t) / sq;
    let d2 = d1 - sq;
    s * (s_ * (-b * t).exp() * ncdf(s * d1) - k * (-r * t).exp() * ncdf(s * d2))
}

/// BSM price; intrinsic value when `tau <= 0`.
pub fn bsm_price(
    spot: f64,
    strike: f64,
    tau: f64,
    rate: f64,
    carry: f64,
    vol: f64,
    kind: OptionKind,
) -> Result<f64> {
    require(
        spot.is_finite() && spot > 0.0 && strike.is_finite() && strike > 0.0,
        "spot and strike must be > 0",
    )?;
    require(
        tau.is_finite() && rate.is_finite() && carry.is_finite(),
        "non-finite input",
    )?;
    require(vol.is_finite() && vol > 0.0, "vol must be > 0")?;
    Ok(price_unchecked(spot, strike, tau, rate, carry, vol, kind))
}

/// Closed-form Greeks per unit quantity.
pub fn bsm_greeks(l: &OptionLeg) -> Result<BsmGreeks> {
    check_leg(l)?;
    let s = sign(l.kind);
    let rt = l.tau.sqrt();
    let sq = l.vol * rt;
    let d1 = ((l.spot / l.strike).ln() + (l.rate - l.carry + 0.5 * l.vol * l.vol) * l.tau) / sq;
    let d2 = d1 - sq;
    let eb = (-l.carry * l.tau).exp();
    let er = (-l.rate * l.tau).exp();
    let pdf = npdf(d1);
    let vega = l.spot * eb * pdf * rt;
    Ok(BsmGreeks {
        delta: s * eb * ncdf(s * d1),
        gamma: eb * pdf / (l.spot * sq),
        vega,
        vanna: -eb * pdf * d2 / l.vol,
        volga: vega * d1 * d2 / l.vol,
        theta: -l.spot * eb * pdf * l.vol / (2.0 * rt) + s * l.carry * l.spot * eb * ncdf(s * d1)
            - s * l.rate * l.strike * er * ncdf(s * d2),
    })
}

/// Exact book P&L per scenario by repricing every leg; `dt` (years) ages
/// the options, floored at expiry. Sticky strike: the same as
/// [`full_revaluation_pnl_vol_mode`] with [`VolMode::StickyStrike`].
pub fn full_revaluation_pnl(
    legs: &[OptionLeg],
    spot_log_returns: &[f64],
    vol_shocks: &[f64],
    n_factors: usize,
    dt: f64,
) -> Result<Vec<f64>> {
    full_revaluation_pnl_vol_mode(
        legs,
        spot_log_returns,
        vol_shocks,
        n_factors,
        dt,
        VolMode::StickyStrike,
        &[],
    )
}

/// How the scenario vol of an option leg is formed; mirrors the Python
/// reference's `vol_mode` (Derman 1999, "Regimes of volatility", Risk
/// 12(4), for the first two). Every mode floors the vol at `1e-12`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolMode {
    /// `sigma_1 = sigma_0 + dv`: the strike's vol does not move with spot.
    StickyStrike,
    /// `sigma_1 = sigma_0 - skew * x + dv`: the smile moves with spot;
    /// `skew = d sigma / d ln(K/S)`, `x` the factor log return.
    StickyDelta,
    /// `sigma_1 = sigma_0 * exp(dv)`: `dv` is the historical log change of
    /// the ATM vol, applied to every strike.
    HistoricalAtm,
}

/// Full revaluation P&L with an explicit vol mode. `skews` is empty (all
/// zero) or has one entry per leg; only [`VolMode::StickyDelta`] reads it.
///
/// # Errors
/// As [`full_revaluation_pnl`], plus a `skews` length mismatch or a
/// non-finite skew.
pub fn full_revaluation_pnl_vol_mode(
    legs: &[OptionLeg],
    spot_log_returns: &[f64],
    vol_shocks: &[f64],
    n_factors: usize,
    dt: f64,
    mode: VolMode,
    skews: &[f64],
) -> Result<Vec<f64>> {
    let n = check_scenarios(legs, spot_log_returns, vol_shocks, n_factors)?;
    require(dt.is_finite() && dt >= 0.0, "dt must be >= 0")?;
    require(
        skews.is_empty() || skews.len() == legs.len(),
        "skews must be empty or one per leg",
    )?;
    require(skews.iter().all(|k| k.is_finite()), "skews must be finite")?;
    let mut pnl = vec![0.0; n];
    for (k, l) in legs.iter().enumerate() {
        let skew = skews.get(k).copied().unwrap_or(0.0);
        let p0 = price_unchecked(l.spot, l.strike, l.tau, l.rate, l.carry, l.vol, l.kind);
        let t1 = (l.tau - dt).max(0.0);
        for (i, p) in pnl.iter_mut().enumerate() {
            let j = i * n_factors + l.factor;
            let x = spot_log_returns[j];
            let d = vol_shocks.get(j).copied().unwrap_or(0.0);
            let s1 = l.spot * x.exp();
            let v1 = match mode {
                VolMode::StickyStrike => l.vol + d,
                VolMode::StickyDelta => l.vol - skew * x + d,
                VolMode::HistoricalAtm => l.vol * d.exp(),
            };
            *p +=
                l.quantity * (price_unchecked(s1, l.strike, t1, l.rate, l.carry, v1, l.kind) - p0);
        }
    }
    Ok(pnl)
}

/// Delta-gamma P&L per scenario: `Delta dS + Gamma dS^2 / 2`, `dS = S (e^x - 1)`.
pub fn delta_gamma_pnl(
    legs: &[OptionLeg],
    spot_log_returns: &[f64],
    n_factors: usize,
) -> Result<Vec<f64>> {
    let n = check_scenarios(legs, spot_log_returns, &[], n_factors)?;
    let mut pnl = vec![0.0; n];
    for l in legs {
        let g = bsm_greeks(l)?;
        for (i, p) in pnl.iter_mut().enumerate() {
            let ds = l.spot * spot_log_returns[i * n_factors + l.factor].exp_m1();
            *p += l.quantity * (g.delta * ds + 0.5 * g.gamma * ds * ds);
        }
    }
    Ok(pnl)
}

/// Scenario P&L of a mixed book: `exposures . r` (the engine's linear
/// convention, unchanged) plus [`full_revaluation_pnl`] of the option legs on
/// the same rows. `exposures` is empty (options only) or has one entry per
/// column; `legs` may be empty. No vol shocks.
///
/// # Errors
/// Empty book or panel, shape mismatch, non-finite input, invalid legs.
pub fn option_book_pnl(
    legs: &[OptionLeg],
    exposures: &[f64],
    factor_returns: &Matrix,
    dt: f64,
) -> Result<Vec<f64>> {
    let (n, nf) = (factor_returns.rows(), factor_returns.cols());
    require(n > 0 && nf > 0, "factor_returns must be non-empty")?;
    require(
        exposures.is_empty() || exposures.len() == nf,
        "exposures must be empty or match columns",
    )?;
    require(
        !legs.is_empty() || !exposures.is_empty(),
        "book has no positions",
    )?;
    require(
        exposures.iter().all(|w| w.is_finite()),
        "exposures must be finite",
    )?;
    let x = factor_returns.as_slice();
    let mut pnl = if legs.is_empty() {
        require(
            x.iter().all(|v| v.is_finite()),
            "spot_log_returns must be finite",
        )?;
        vec![0.0; n]
    } else {
        full_revaluation_pnl(legs, x, &[], nf, dt)?
    };
    if !exposures.is_empty() {
        for (i, p) in pnl.iter_mut().enumerate() {
            let row = &x[i * nf..(i + 1) * nf];
            let mut s = 0.0;
            for (w, r) in exposures.iter().zip(row) {
                s += w * r;
            }
            *p += s;
        }
    }
    Ok(pnl)
}

/// Historical-simulation VaR / ES with full option revaluation: each of the
/// `T >= MIN_OBS` rows of `factor_returns` is a scenario, the book P&L is
/// [`option_book_pnl`], and `(VaR, ES)` come from the S2 order statistic.
/// With no option legs this is exactly the linear path.
///
/// # Errors
/// Fewer than `MIN_OBS` rows, invalid `alpha`, or any [`option_book_pnl`] error.
pub fn historical_option_var_es(
    legs: &[OptionLeg],
    exposures: &[f64],
    factor_returns: &Matrix,
    alpha: f64,
    dt: f64,
) -> Result<(f64, f64)> {
    validate_alpha(alpha)?;
    require(
        factor_returns.rows() >= crate::historical::MIN_OBS,
        "historical option VaR needs at least MIN_OBS rows",
    )?;
    let pnl = option_book_pnl(legs, exposures, factor_returns, dt)?;
    crate::tail::empirical_var_es(&pnl, alpha)
}

/// Monte Carlo VaR / ES with full option revaluation: the same simulation as
/// [`monte_carlo_run`] (model, seed, bootstrap seed offset), then
/// [`option_book_pnl`] on every path. With no option legs it calls
/// [`monte_carlo_run`] unchanged.
///
/// # Errors
/// Invalid `alpha`, fewer than `MIN_PATHS` paths, `n_boot < 2`, shape
/// mismatch, invalid legs or tail model, unfactorisable covariance.
#[allow(clippy::too_many_arguments)]
pub fn monte_carlo_option_run(
    legs: &[OptionLeg],
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
    n_boot: usize,
    dt: f64,
) -> Result<McRun> {
    if legs.is_empty() {
        return monte_carlo_run(exposures, cov, alpha, n_paths, tail, seed, n_boot);
    }
    validate_alpha(alpha)?;
    require(n_boot >= 2, "n_boot must be >= 2")?;
    require(n_paths >= MIN_PATHS, "need at least MIN_PATHS paths")?;
    require(
        cov.rows() == cov.cols() && cov.rows() > 0,
        "cov must be square and non-empty",
    )?;
    require(
        exposures.is_empty() || exposures.len() == cov.rows(),
        "exposures must be empty or match cov",
    )?;
    let (scen, jitter) = simulate_factor_returns_jitter(cov, n_paths, tail, seed)?;
    let pnl = option_book_pnl(legs, exposures, &scen, dt)?;
    let (var, es) = crate::tail::empirical_var_es(&pnl, alpha)?;
    let se = var_bootstrap_se(&pnl, alpha, n_boot, seed ^ 0x9E37_79B9_7F4A_7C15)?;
    let se_kde = var_kde_se(&pnl, alpha)?;
    Ok(McRun {
        pnl,
        var,
        es,
        se,
        se_kde,
        jitter,
    })
}
