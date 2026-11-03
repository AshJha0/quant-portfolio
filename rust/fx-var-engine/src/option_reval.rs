//! Option revaluation under VaR scenarios.
//!
//! Black-Scholes-Merton with a continuous carry yield (dividend yield for
//! equity, foreign rate for FX, which gives Garman-Kohlhagen), closed-form
//! Greeks (Hull, *Options, Futures and Other Derivatives*, 11th ed., ch. 19),
//! full revaluation P&L per scenario and the delta-gamma approximation.
//! Mirrors `fx_var.revaluation`; the golden values in
//! `tests/test_option_reval.rs` come from that module.
//!
//! Scenario layout: row-major `n_scenarios x n_factors` spot log returns;
//! vol shocks (absolute, decimal) in the same layout, or empty for none.

//!
//! [`layout_option_book`] and [`option_book_pnl`] let
//! [`crate::historical::historical_option_var`] and
//! [`crate::monte_carlo::monte_carlo_option_var`] reprice every option leg in
//! full in every scenario (Jorion, *Value at Risk*, 3rd ed., ch. 10). Options
//! are on CCYUSD: spot in USD per unit of CCY, domestic rate `r_USD`, carry
//! `r_CCY`, P&L in USD, so a book holding options must report in USD.

use crate::book::{Book, CompiledBook, Market};
use crate::returns::ReturnsMatrix;
use crate::stats::{norm_cdf as ncdf, norm_pdf as npdf};
use crate::{FxVarError, Result};

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
        Err(FxVarError::Invalid(format!("option_reval: {what}")))
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

/// A linear book plus option legs on one factor axis: `factors` lists the
/// book's own factors first ([`CompiledBook`] order), then any option factor
/// the book does not carry; `legs` have `factor` re-indexed into `factors`.
#[derive(Debug)]
pub struct OptionBookLayout {
    /// Union factor axis.
    pub factors: Vec<String>,
    /// The compiled linear book, `None` when the book is empty.
    pub linear: Option<CompiledBook>,
    /// Option legs, `factor` indexing `factors`.
    pub legs: Vec<OptionLeg>,
}

/// Build the layout. `options[k].factor` indexes `option_factors`, whose
/// names must be `"FX:CCY"` (CCY != USD).
///
/// # Errors
/// No options, a bad factor index or name, invalid legs, or a non-USD base.
pub fn layout_option_book(
    book: &Book,
    market: &Market,
    options: &[OptionLeg],
    option_factors: &[String],
) -> Result<OptionBookLayout> {
    require(!options.is_empty(), "no option legs")?;
    let mut factors = Vec::new();
    let mut linear = None;
    if !book.is_empty() {
        require(
            book.base() == "USD",
            "a book holding options must report in USD",
        )?;
        let c = CompiledBook::new(book, market)?;
        factors = c.factors().to_vec();
        linear = Some(c);
    }
    for name in option_factors {
        require(
            name.len() > 3 && name.starts_with("FX:") && name != "FX:USD",
            "option factors must be named FX:CCY with CCY != USD",
        )?;
    }
    let mut legs = options.to_vec();
    for l in legs.iter_mut() {
        check_leg(l)?;
        require(
            l.factor < option_factors.len(),
            "leg factor index out of range",
        )?;
        let name = &option_factors[l.factor];
        let idx = match factors.iter().position(|f| f == name) {
            Some(i) => i,
            None => {
                factors.push(name.clone());
                factors.len() - 1
            }
        };
        l.factor = idx;
    }
    Ok(OptionBookLayout {
        factors,
        linear,
        legs,
    })
}

/// Scenario P&L (USD): [`CompiledBook::pnl_scenarios`] for the linear legs
/// plus [`full_revaluation_pnl`] for the options. `scenarios` must carry every
/// factor in `layout.factors` (re-selected into that order; extras ignored).
///
/// # Errors
/// Missing factors, non-finite scenarios, negative `dt`.
pub fn option_book_pnl(
    layout: &OptionBookLayout,
    scenarios: &ReturnsMatrix,
    dt: f64,
) -> Result<Vec<f64>> {
    let rets = scenarios.select(&layout.factors)?;
    let nf = layout.factors.len();
    let mut pnl = full_revaluation_pnl(&layout.legs, rets.data.as_slice(), &[], nf, dt)?;
    if let Some(c) = &layout.linear
        && !c.factors().is_empty()
    {
        let lin = c.pnl_scenarios(&rets)?;
        for (p, l) in pnl.iter_mut().zip(lin) {
            *p += l;
        }
    }
    Ok(pnl)
}

/// Vol scenarios for option VaR. The vol shocks sit on the same named factor
/// axis as the spot returns: the vol factor of `"FX:CCY"` is `"VOL:CCY"` (an
/// ATM vol change in vol points for `StickyStrike` / `StickyDelta`, an ATM
/// log change for `HistoricalAtm`). A returns panel or covariance used with
/// these must carry `"VOL:CCY"` for every option factor; the combined axis is
/// `layout.factors` followed by [`option_vol_factors`]. `skews` is empty or
/// one per option leg (in the order passed in), read only by `StickyDelta`.
#[derive(Debug, Clone, PartialEq)]
pub struct VolScenarios {
    /// How the scenario vol is formed.
    pub mode: VolMode,
    /// Per-leg skew `d sigma / d ln(K/S)`; empty means all zero.
    pub skews: Vec<f64>,
}

impl Default for VolScenarios {
    fn default() -> Self {
        Self {
            mode: VolMode::StickyStrike,
            skews: Vec::new(),
        }
    }
}

/// `"VOL:CCY"` for every option factor of `layout`, in `layout.factors` order.
pub fn option_vol_factors(layout: &OptionBookLayout) -> Vec<String> {
    let mut used = vec![false; layout.factors.len()];
    for l in &layout.legs {
        used[l.factor] = true;
    }
    layout
        .factors
        .iter()
        .zip(used)
        .filter(|(_, u)| *u)
        .map(|(f, _)| format!("VOL:{}", &f[3..]))
        .collect()
}

/// `layout.factors` followed by [`option_vol_factors`].
pub fn option_combined_factors(layout: &OptionBookLayout) -> Vec<String> {
    let mut out = layout.factors.clone();
    out.extend(option_vol_factors(layout));
    out
}

/// [`option_book_pnl`] with vol scenarios read from the `"VOL:CCY"` columns
/// of `scenarios`. With `StickyStrike` and all-zero vol columns it equals
/// [`option_book_pnl`] bit for bit.
///
/// # Errors
/// As [`option_book_pnl`], plus missing `"VOL:CCY"` columns or a `skews`
/// length mismatch.
pub fn option_book_pnl_vol(
    layout: &OptionBookLayout,
    scenarios: &ReturnsMatrix,
    dt: f64,
    vol: &VolScenarios,
) -> Result<Vec<f64>> {
    require(
        vol.skews.is_empty() || vol.skews.len() == layout.legs.len(),
        "skews must be empty or one per option leg",
    )?;
    let rets = scenarios.select(&layout.factors)?;
    let vnames = option_vol_factors(layout);
    let vols = scenarios.select(&vnames)?;
    let nf = layout.factors.len();
    let n = rets.n_obs();
    // dv on the spot axis: column j holds the vol shock of factor j (zero for
    // factors no option references; the repricer never reads those).
    let mut dv = vec![0.0; n * nf];
    let mut k = 0;
    for (j, f) in layout.factors.iter().enumerate() {
        if k < vnames.len() && vnames[k] == format!("VOL:{}", &f[3..]) {
            for i in 0..n {
                dv[i * nf + j] = vols.data.get(i, k);
            }
            k += 1;
        }
    }
    let mut pnl = full_revaluation_pnl_vol_mode(
        &layout.legs,
        rets.data.as_slice(),
        &dv,
        nf,
        dt,
        vol.mode,
        &vol.skews,
    )?;
    if let Some(c) = &layout.linear
        && !c.factors().is_empty()
    {
        let lin = c.pnl_scenarios(&rets)?;
        for (p, l) in pnl.iter_mut().zip(lin) {
            *p += l;
        }
    }
    Ok(pnl)
}
