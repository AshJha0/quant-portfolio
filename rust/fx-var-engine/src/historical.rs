//! Historical-simulation VaR: plain HS, age-weighted (BRW), filtered HS.
//!
//! All three variants revalue the *actual book* (full revaluation through
//! [`crate::book::CompiledBook`], including forwards' rate legs) under
//! historical factor-return scenarios (mirrors Python `fx_var.historical_var`
//! / C++ `fxvar::historical_var`):
//!
//! * **Plain**: each of the last T days is an equally weighted scenario.
//! * **Age**: Boudoukh-Richardson-Whitelaw exponential age weights
//!   `w_t ~ lambda^age`: recent days dominate, so the VaR reacts faster
//!   after a regime change.
//! * **Filtered**: Filtered Historical Simulation (Barone-Adesi et al.):
//!   returns are devolatilised by a per-factor EWMA sigma and rescaled to
//!   today's sigma forecast, preserving the empirical cross-sectional
//!   dependence while making the scenario set conditionally
//!   heteroscedastic.
//!
//! Multi-day horizons use sqrt-time scaling of the 1-day figure (documented
//! limitation for carry books with negative skew: `docs/VALIDATION.md`).
//!
//! **Peg blindness**: HS sees only what is in the window. A pegged
//! currency contributes ~zero scenarios, so the engine surfaces a warnings
//! list ([`HistoricalResult::warnings`] / `flagged_peg_factors`) and the
//! desk must add the peg-break stress add-on
//! ([`crate::stress::peg_break_scenario`]).

use crate::book::{Book, CompiledBook, Market};
use crate::expected_shortfall::empirical_var_es;
use crate::option_reval::{OptionLeg, layout_option_book, option_book_pnl};
use crate::returns::{ReturnsMatrix, ewma_volatility, flag_peg_factors, validate_returns};
use crate::stats::{validate_alpha, validate_horizon};
use crate::{FxVarError, Result};

/// Historical-simulation weighting scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsMethod {
    /// Equal-weighted last-T-day scenarios.
    Plain,
    /// Boudoukh-Richardson-Whitelaw exponential age weighting.
    Age,
    /// Filtered Historical Simulation (EWMA devolatilisation/rescaling).
    Filtered,
}

/// Historical VaR quantile rule.
///
/// ES is the exact tail average under either rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistRule {
    /// The `ceil(n alpha)`-th worst loss (BCBS MAR33.4 / MAR99: 99 % on 250
    /// days is the 3rd worst). The default.
    #[default]
    KthWorst,
    /// Type-7 linear quantile of the losses at `h = (n - 1) alpha`, counted
    /// 0-based from the worst (Hyndman and Fan 1996, definition 7). Needs
    /// equal weights, so not with [`HsMethod::Age`].
    Interpolated,
}

/// Type-7 VaR of a non-empty finite P&L sample (see [`HistRule::Interpolated`]).
///
/// # Errors
/// [`FxVarError::Invalid`] for an empty or non-finite sample or an invalid `alpha`.
pub fn interpolated_var(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    if pnl.is_empty() || pnl.iter().any(|x| !x.is_finite()) {
        return Err(FxVarError::Invalid(
            "interpolated_var: need a non-empty finite sample".to_string(),
        ));
    }
    let n = pnl.len();
    let mut loss: Vec<f64> = pnl.iter().map(|x| -x).collect();
    let h = (n - 1) as f64 * alpha;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    // Worst first: the hi+1 largest losses, sorted descending.
    loss.select_nth_unstable_by(hi, |a, b| b.total_cmp(a));
    loss[..=hi].sort_unstable_by(|a, b| b.total_cmp(a));
    Ok(loss[lo] + (h - lo as f64) * (loss[hi] - loss[lo]))
}

/// Equal-weight historical VaR of a P&L sample under `rule`.
///
/// # Errors
/// As [`empirical_var_es`] and [`interpolated_var`].
pub fn historical_var_from_pnl(pnl: &[f64], alpha: f64, rule: HistRule) -> Result<f64> {
    match rule {
        HistRule::KthWorst => Ok(empirical_var_es(pnl, alpha, None)?.0),
        HistRule::Interpolated => interpolated_var(pnl, alpha),
    }
}

fn finish_var(pnl: &[f64], opts: &HistoricalOptions, var1: f64) -> Result<f64> {
    match opts.rule {
        HistRule::KthWorst => Ok(var1),
        HistRule::Interpolated => interpolated_var(pnl, opts.alpha),
    }
}

/// Options controlling [`historical_var`].
#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalOptions {
    /// Tail probability, e.g. `0.01` for 99 % VaR (was a confidence level).
    pub alpha: f64,
    /// Horizon in trading days; `> 1` uses sqrt-time scaling.
    pub horizon_days: f64,
    /// Weighting scheme.
    pub method: HsMethod,
    /// BRW age-weight decay (used when `method == Age`).
    pub decay: f64,
    /// FHS devolatilisation decay (used when `method == Filtered`).
    pub ewma_lambda: f64,
    /// Minimum required history length.
    pub min_obs: usize,
    /// Whether to screen `FX:*` factors for peg blindness.
    pub warn_pegs: bool,
    /// VaR quantile rule (default [`HistRule::KthWorst`]).
    pub rule: HistRule,
}

impl Default for HistoricalOptions {
    fn default() -> Self {
        HistoricalOptions {
            alpha: 0.01,
            horizon_days: 1.0,
            method: HsMethod::Plain,
            decay: 0.995,
            ewma_lambda: 0.94,
            min_obs: 60,
            warn_pegs: true,
            rule: HistRule::KthWorst,
        }
    }
}

/// Result of a historical-simulation VaR run.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalResult {
    /// Positive VaR loss in the book's base currency at the requested
    /// horizon.
    pub var: f64,
    /// Positive ES loss in the book's base currency at the requested
    /// horizon.
    pub es: f64,
    /// Tail probability used.
    pub alpha: f64,
    /// Horizon in trading days used.
    pub horizon_days: f64,
    /// Weighting scheme used.
    pub method: HsMethod,
    /// 1-day scenario P&L vector (profit +, loss -).
    pub pnl: Vec<f64>,
    /// Scenario weights used for the quantile (sums to 1).
    pub weights: Vec<f64>,
    /// `FX:*` factors flagged as peg-like.
    pub flagged_peg_factors: Vec<String>,
    /// Human-readable peg-blindness diagnostics.
    pub warnings: Vec<String>,
}

/// Historical-simulation VaR/ES for `book` at `market`.
///
/// `returns` must contain every factor in `book.factors()` (`"FX:*"` log
/// returns, `"IR:*"` absolute changes); NaNs error. An empty book errors; a
/// non-empty book with no risk factors (pure base-ccy cash) reports exactly
/// zero VaR/ES.
///
/// # Errors
/// [`FxVarError::Invalid`] for an empty book, bad options, or a history
/// that fails validation ([`validate_returns`]).
pub fn historical_var(
    book: &Book,
    market: &Market,
    returns: &ReturnsMatrix,
    opts: &HistoricalOptions,
) -> Result<HistoricalResult> {
    validate_alpha(opts.alpha)?;
    validate_horizon(opts.horizon_days)?;
    let compiled = CompiledBook::new(book, market)?;

    let mut res = HistoricalResult {
        var: 0.0,
        es: 0.0,
        alpha: opts.alpha,
        horizon_days: opts.horizon_days,
        method: opts.method,
        pnl: Vec::new(),
        weights: Vec::new(),
        flagged_peg_factors: Vec::new(),
        warnings: Vec::new(),
    };

    let factors = compiled.factors().to_vec();
    if factors.is_empty() {
        // Pure base-ccy (or USD-in-USD-book) cash: zero risk by construction.
        let n = returns.n_obs();
        res.pnl = vec![0.0; n];
        res.weights = vec![if n > 0 { 1.0 / n as f64 } else { 0.0 }; n];
        return Ok(res);
    }

    validate_returns(returns, &factors, opts.min_obs)?;
    let mut rets = returns.select(&factors)?;

    let weights = prepare_scenarios(&mut rets, opts, &mut res)?;

    res.pnl = compiled.pnl_scenarios(&rets)?;
    res.weights = weights.clone();
    let (var1, es1) = empirical_var_es(&res.pnl, opts.alpha, Some(&weights))?;
    let scale = opts.horizon_days.sqrt();
    res.var = finish_var(&res.pnl, opts, var1)? * scale;
    res.es = es1 * scale;
    Ok(res)
}

/// Peg warnings, scenario weights and (for `Filtered`) the per-factor FHS
/// rescaling of `rets` in place. Shared by the linear and option drivers.
fn prepare_scenarios(
    rets: &mut ReturnsMatrix,
    opts: &HistoricalOptions,
    res: &mut HistoricalResult,
) -> Result<Vec<f64>> {
    if opts.rule == HistRule::Interpolated && opts.method == HsMethod::Age {
        return Err(FxVarError::Invalid(
            "rule Interpolated needs equal weights; use method Plain or Filtered".to_string(),
        ));
    }
    if opts.warn_pegs {
        res.flagged_peg_factors = flag_peg_factors(rets, crate::returns::PEG_VOL_THRESHOLD)?;
        for f in &res.flagged_peg_factors {
            res.warnings.push(format!(
                "peg blindness: factor {f} has daily vol < {} (pegged/managed currency). \
                 Historical and parametric VaR are blind to peg-break risk; add the \
                 peg-break stress add-on (crate::stress::peg_break_scenario).",
                crate::returns::PEG_VOL_THRESHOLD
            ));
        }
    }

    let n = rets.n_obs();
    let weights = match opts.method {
        HsMethod::Plain => vec![1.0 / n as f64; n],
        HsMethod::Age => {
            if !(opts.decay > 0.0 && opts.decay < 1.0) {
                return Err(FxVarError::invalid(format!(
                    "decay must be in (0, 1), got {}",
                    opts.decay
                )));
            }
            let mut w = vec![0.0; n];
            let mut sum = 0.0;
            for i in 0..n {
                let age = (n - 1 - i) as f64; // last row = age 0
                w[i] = opts.decay.powf(age);
                sum += w[i];
            }
            for v in w.iter_mut() {
                *v /= sum;
            }
            w
        }
        HsMethod::Filtered => {
            let ev = ewma_volatility(rets, opts.ewma_lambda)?;
            for i in 0..n {
                for j in 0..rets.n_factors() {
                    let v = rets.data.get(i, j) / ev.sigma.get(i, j) * ev.sigma_next[j];
                    rets.data.set(i, j, v);
                }
            }
            vec![1.0 / n as f64; n]
        }
    };

    Ok(weights)
}

/// Historical-simulation VaR/ES with full option revaluation: the linear
/// book as in [`historical_var`] plus `options` (`leg.factor` indexes
/// `option_factors`, e.g. `"FX:EUR"`), every leg repriced by
/// Garman-Kohlhagen in every scenario and aged by `dt` years. All three
/// methods apply (`Filtered` rescales the spot returns before repricing).
/// `horizon_days` must be 1: sqrt-time scaling is not valid for a
/// non-linear book. The book may be empty; if not, its base must be USD.
/// With no options this is [`historical_var`], unchanged.
///
/// # Errors
/// As [`historical_var`], plus [`crate::option_reval::layout_option_book`]
/// errors and `horizon_days != 1`.
pub fn historical_option_var(
    book: &Book,
    market: &Market,
    options: &[OptionLeg],
    option_factors: &[String],
    returns: &ReturnsMatrix,
    opts: &HistoricalOptions,
    dt: f64,
) -> Result<HistoricalResult> {
    if options.is_empty() {
        return historical_var(book, market, returns, opts);
    }
    validate_alpha(opts.alpha)?;
    validate_horizon(opts.horizon_days)?;
    if opts.horizon_days != 1.0 {
        return Err(FxVarError::invalid(
            "historical_option_var: sqrt-time scaling is not valid for a non-linear book; \
             pass horizon_days = 1 and h-day returns instead",
        ));
    }
    let layout = layout_option_book(book, market, options, option_factors)?;
    let mut res = HistoricalResult {
        var: 0.0,
        es: 0.0,
        alpha: opts.alpha,
        horizon_days: opts.horizon_days,
        method: opts.method,
        pnl: Vec::new(),
        weights: Vec::new(),
        flagged_peg_factors: Vec::new(),
        warnings: Vec::new(),
    };
    validate_returns(returns, &layout.factors, opts.min_obs)?;
    let mut rets = returns.select(&layout.factors)?;
    let weights = prepare_scenarios(&mut rets, opts, &mut res)?;
    res.pnl = option_book_pnl(&layout, &rets, dt)?;
    let (var, es) = empirical_var_es(&res.pnl, opts.alpha, Some(&weights))?;
    res.weights = weights;
    res.var = finish_var(&res.pnl, opts, var)?;
    res.es = es;
    Ok(res)
}

/// [`historical_option_var`] with vol scenarios: `returns` must also carry
/// `"VOL:CCY"` for every option factor (see
/// [`crate::option_reval::VolScenarios`]). The scenarios are the rows of the
/// combined axis [`crate::option_reval::option_combined_factors`];
/// `Filtered` rescales the vol columns by their own EWMA vol, like any other
/// factor.
///
/// # Errors
/// As [`historical_option_var`], plus no options, missing `"VOL:CCY"`
/// columns or a `skews` length mismatch.
#[allow(clippy::too_many_arguments)]
pub fn historical_option_var_vol(
    book: &Book,
    market: &Market,
    options: &[OptionLeg],
    option_factors: &[String],
    returns: &ReturnsMatrix,
    opts: &HistoricalOptions,
    dt: f64,
    vol: &crate::option_reval::VolScenarios,
) -> Result<HistoricalResult> {
    if options.is_empty() {
        return Err(FxVarError::invalid("vol scenarios need option legs"));
    }
    validate_alpha(opts.alpha)?;
    validate_horizon(opts.horizon_days)?;
    if opts.horizon_days != 1.0 {
        return Err(FxVarError::invalid(
            "historical_option_var: sqrt-time scaling is not valid for a non-linear book; \
             pass horizon_days = 1 and h-day returns instead",
        ));
    }
    let layout = layout_option_book(book, market, options, option_factors)?;
    let axis = crate::option_reval::option_combined_factors(&layout);
    let mut res = HistoricalResult {
        var: 0.0,
        es: 0.0,
        alpha: opts.alpha,
        horizon_days: opts.horizon_days,
        method: opts.method,
        pnl: Vec::new(),
        weights: Vec::new(),
        flagged_peg_factors: Vec::new(),
        warnings: Vec::new(),
    };
    validate_returns(returns, &axis, opts.min_obs)?;
    let mut rets = returns.select(&axis)?;
    let weights = prepare_scenarios(&mut rets, opts, &mut res)?;
    res.pnl = crate::option_reval::option_book_pnl_vol(&layout, &rets, dt, vol)?;
    let (var, es) = empirical_var_es(&res.pnl, opts.alpha, Some(&weights))?;
    res.weights = weights;
    res.var = finish_var(&res.pnl, opts, var)?;
    res.es = es;
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::{Position, SpotPosition};
    use crate::matrix::Matrix;

    fn market() -> Market {
        Market::new(
            [("EUR", 1.10), ("JPY", 0.0090)],
            [("USD", 0.05), ("EUR", 0.03), ("JPY", 0.001)],
        )
        .unwrap()
    }

    fn returns(n: usize) -> ReturnsMatrix {
        let mut data = Matrix::zeros(n, 1);
        for t in 0..n {
            data.set(t, 0, 0.01 * (0.1 * t as f64).sin());
        }
        ReturnsMatrix::new(vec!["FX:EUR".to_string()], data).unwrap()
    }

    #[test]
    fn plain_var_positive_and_es_ge_var() {
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))],
            "USD",
        )
        .unwrap();
        let r = historical_var(
            &book,
            &market(),
            &returns(120),
            &HistoricalOptions::default(),
        )
        .unwrap();
        assert!(r.var > 0.0);
        assert!(r.es >= r.var);
    }

    #[test]
    fn zero_risk_book_reports_zero() {
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 0.0, None))],
            "USD",
        )
        .unwrap();
        // still has FX:EUR factor with zero exposure -> zero pnl always.
        let r = historical_var(
            &book,
            &market(),
            &returns(120),
            &HistoricalOptions::default(),
        )
        .unwrap();
        assert_eq!(r.var, 0.0);
        assert_eq!(r.es, 0.0);
    }

    #[test]
    fn age_weighted_reacts_to_recent_regime() {
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))],
            "USD",
        )
        .unwrap();
        let mut opts = HistoricalOptions {
            method: HsMethod::Age,
            ..Default::default()
        };
        opts.decay = 0.9;
        let r = historical_var(&book, &market(), &returns(120), &opts).unwrap();
        assert!(r.var > 0.0);
        assert!((r.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn filtered_hs_runs() {
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))],
            "USD",
        )
        .unwrap();
        let opts = HistoricalOptions {
            method: HsMethod::Filtered,
            ..Default::default()
        };
        let r = historical_var(&book, &market(), &returns(120), &opts).unwrap();
        assert!(r.var > 0.0);
    }

    #[test]
    fn insufficient_history_errors() {
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))],
            "USD",
        )
        .unwrap();
        assert!(
            historical_var(
                &book,
                &market(),
                &returns(10),
                &HistoricalOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn empty_book_errors() {
        assert!(
            historical_var(
                &Book::empty(),
                &market(),
                &returns(120),
                &HistoricalOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn peg_flagging_surfaces_warning() {
        let mut data = Matrix::zeros(120, 1);
        for t in 0..120 {
            data.set(t, 0, 0.00001 * ((t % 3) as f64 - 1.0));
        }
        let rets = ReturnsMatrix::new(vec!["FX:EUR".to_string()], data).unwrap();
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))],
            "USD",
        )
        .unwrap();
        let r = historical_var(&book, &market(), &rets, &HistoricalOptions::default()).unwrap();
        assert_eq!(r.flagged_peg_factors, vec!["FX:EUR".to_string()]);
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn horizon_scaling_is_sqrt_time() {
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))],
            "USD",
        )
        .unwrap();
        let opts1 = HistoricalOptions::default();
        let opts10 = HistoricalOptions {
            horizon_days: 10.0,
            ..Default::default()
        };
        let r1 = historical_var(&book, &market(), &returns(120), &opts1).unwrap();
        let r10 = historical_var(&book, &market(), &returns(120), &opts10).unwrap();
        assert!((r10.var - r1.var * 10f64.sqrt()).abs() < 1e-6 * r10.var);
    }
}
