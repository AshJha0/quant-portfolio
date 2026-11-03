//! VaR backtesting: Kupiec POF, Christoffersen independence, Basel traffic
//! light (mirrors Python `fx_var.backtesting` / C++ `fxvar::backtest`).
//!
//! Exception convention: day `t` is an exception when the realised *loss*
//! exceeds the VaR forecast made ex ante for day `t`: `-pnl_t > var_t`.
//!
//! # Tests
//!
//! * **Kupiec POF** (unconditional coverage): LR test that the exception
//!   frequency equals the tail probability `alpha`; `chi2(1)` via the
//!   regularised incomplete gamma.
//! * **Christoffersen independence**: LR test against first-order Markov
//!   clustering of exceptions; `chi2(1)`. FX desks care because volatility
//!   clustering makes an unconditional method (plain HS, sample-cov
//!   parametric) fail *this* test long before it fails Kupiec.
//! * **Conditional coverage**: `LR_cc = LR_uc + LR_ind`; `chi2(2)`.
//! * **Basel traffic light**: zones from the exact cumulative
//!   `Binomial(n, alpha)` probability `F(x)` of the exception count (pmf
//!   recursion): green `< 95%`, yellow `< 99.99%`, red `>= 99.99%`. The
//!   yellow plus factor is read at the equivalent 250-day count. On the
//!   regulatory 250-day 99% window this reproduces the table: green 0-4,
//!   yellow 5-9 (0.40/0.50/0.65/0.75/0.85), red 10+ (multiplier 4.0).
//! * **Acerbi-Szekely Z2**: ES backtest statistic, paper sign.

use crate::stats::{chi2_sf, validate_alpha};
use crate::{FxVarError, Result};

fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if y <= 0.0 {
        f64::NEG_INFINITY
    } else {
        x * y.ln()
    }
}

/// A likelihood-ratio test outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrTest {
    /// Test statistic (chi-square distributed under H0).
    pub lr: f64,
    /// Survival-function p-value.
    pub p: f64,
}

/// Kupiec result type (the LR statistic and its chi2(1) p-value).
pub type KupiecResult = LrTest;

/// Inputs of the coverage tests ([`kupiec_pof`], [`basel_traffic_light`]).
///
/// Named fields make the old silent swap of `n_obs` and `n_exceptions`
/// impossible: `KupiecInput { n_obs: 250, n_exceptions: 5, alpha: 0.01 }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KupiecInput {
    /// Number of backtest days `T` (`>= 1`).
    pub n_obs: i64,
    /// Number of exceptions `x` (`0 <= x <= T`).
    pub n_exceptions: i64,
    /// Tail probability of the VaR being tested (e.g. `0.01` for 99 %).
    pub alpha: f64,
}

/// Same struct under the name the Basel function reads best with.
pub type BaselInput = KupiecInput;

fn check_coverage_input(inp: &KupiecInput) -> Result<()> {
    crate::tail::validate_alpha(inp.alpha)?;
    if inp.n_obs < 1 {
        return Err(FxVarError::Invalid(format!(
            "n_obs must be >= 1, got {}",
            inp.n_obs
        )));
    }
    if inp.n_exceptions < 0 {
        return Err(FxVarError::Invalid(format!(
            "n_exceptions must be >= 0, got {}",
            inp.n_exceptions
        )));
    }
    if inp.n_exceptions > inp.n_obs {
        return Err(FxVarError::Invalid(format!(
            "n_exceptions ({}) exceeds n_obs ({})",
            inp.n_exceptions, inp.n_obs
        )));
    }
    if inp.n_obs > (1_i64 << 53) {
        return Err(FxVarError::Invalid(format!(
            "n_obs {} exceeds the exactly representable range",
            inp.n_obs
        )));
    }
    Ok(())
}

/// `x ln y` with `0 ln 0 = 0`.
fn xlny(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

/// Kupiec LR statistic for valid inputs:
/// `LR = -2 ln[(1-a)^(n-x) a^x] + 2 ln[(1-p)^(n-x) p^x]`, `p = x/n`.
fn kupiec_lr(inp: &KupiecInput) -> f64 {
    let n = inp.n_obs as f64;
    let x = inp.n_exceptions as f64;
    let a = inp.alpha;
    let p = x / n;
    let l0 = xlny(n - x, 1.0 - a) + xlny(x, a);
    let l1 = xlny(n - x, 1.0 - p) + xlny(x, p);
    (-2.0 * (l0 - l1)).max(0.0)
}

/// `F(x) = P(X <= x)`, `X ~ Binomial(n, alpha)`, by the pmf recursion
/// `pmf(0) = exp(n log1p(-alpha))`,
/// `pmf(j+1) = pmf(j) (n-j)/(j+1) alpha/(1-alpha)`, in `f64`.
fn binomial_cdf_recursion(x: i64, n: i64, alpha: f64) -> Result<f64> {
    let nf = n as f64;
    let mut pmf = (nf * (-alpha).ln_1p()).exp();
    if !(pmf > 0.0) {
        return Err(FxVarError::Numerical(format!(
            "Binomial({n}, {alpha}) pmf(0) underflows; the window is too long \
             for the pmf recursion"
        )));
    }
    let ratio = alpha / (1.0 - alpha);
    let mut cdf = pmf;
    for j in 0..x {
        pmf *= (nf - j as f64) / (j as f64 + 1.0) * ratio;
        cdf += pmf;
    }
    Ok(cdf.min(1.0))
}

/// Basel plus factors for the equivalent 250-day exception counts 5..=9.
const PLUS: [f64; 5] = [0.40, 0.50, 0.65, 0.75, 0.85];

/// Zone, plus factor and `F(x)` for valid inputs (S4 rule).
fn basel_rule(inp: &KupiecInput) -> Result<(u8, f64, f64)> {
    let f = binomial_cdf_recursion(inp.n_exceptions, inp.n_obs, inp.alpha)?;
    if f < 0.95 {
        return Ok((0, 0.0, f));
    }
    if f >= 0.9999 {
        return Ok((2, 1.0, f));
    }
    let mut x_eq = 9usize;
    for xp in 5..=9 {
        if binomial_cdf_recursion(xp, 250, 0.01)? >= f - 1e-12 {
            x_eq = xp as usize;
            break;
        }
    }
    Ok((1, PLUS[x_eq - 5], f))
}

/// Acerbi-Szekely (2014) `Z2` statistic for an ES backtest.
///
/// `Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1`, `I_t = 1{pnl_t < -VaR_t}`,
/// with the paper's sign: `Z2 = 0` in expectation under a correct model and
/// a NEGATIVE value means ES is understated. `var` and `es` are positive
/// losses, one per day or a single value broadcast to every day. Returns
/// `Z2` only; the simulated p-value lives in the Python library.
///
/// # Errors
/// Invalid `alpha`, an empty or non-finite series, a length mismatch, a
/// negative VaR, or an ES that is not strictly positive.
///
/// # Examples
///
/// ```
/// # use fx_var_engine::backtest::acerbi_szekely_z2;
/// // Two exceptions out of four days, each exactly at ES: Z2 = 1 - 2/(4 a).
/// let z2 = acerbi_szekely_z2(&[-2.0, 0.0, -2.0, 1.0], &[1.0], &[2.0], 0.25).unwrap();
/// assert!((z2 - (1.0 - 2.0)).abs() < 1e-12);
/// ```
pub fn acerbi_szekely_z2(pnl: &[f64], var: &[f64], es: &[f64], alpha: f64) -> Result<f64> {
    crate::tail::validate_alpha(alpha)?;
    let t = pnl.len();
    if t == 0 {
        return Err(FxVarError::Invalid(
            "acerbi_szekely_z2: empty pnl".to_string(),
        ));
    }
    for (name, s) in [("var", var), ("es", es)] {
        if s.len() != 1 && s.len() != t {
            return Err(FxVarError::Invalid(format!(
                "acerbi_szekely_z2: {name} must have length 1 or {t}, got {}",
                s.len()
            )));
        }
    }
    if pnl.iter().chain(var).chain(es).any(|v| !v.is_finite()) {
        return Err(FxVarError::Invalid(
            "acerbi_szekely_z2: inputs contain NaN or infinite values".to_string(),
        ));
    }
    if var.iter().any(|v| *v < 0.0) {
        return Err(FxVarError::Invalid(
            "acerbi_szekely_z2: VaR must be >= 0".to_string(),
        ));
    }
    if es.iter().any(|v| !(*v > 0.0)) {
        return Err(FxVarError::Invalid(
            "acerbi_szekely_z2: ES must be > 0".to_string(),
        ));
    }
    let pick = |s: &[f64], i: usize| if s.len() == 1 { s[0] } else { s[i] };
    let denom = t as f64 * alpha;
    let mut acc = 0.0;
    for (i, &p) in pnl.iter().enumerate() {
        if p < -pick(var, i) {
            acc += p / (denom * pick(es, i));
        }
    }
    Ok(acc + 1.0)
}

/// Kupiec proportion-of-failures test (S3): `LR_uc ~ chi2(1)` under H0
/// that the exception probability is the tail probability `alpha`.
///
/// # Errors
/// [`FxVarError::Invalid`] for `alpha` outside `(0, 0.5)`, `n_obs < 1`,
/// `n_exceptions < 0` or `n_exceptions > n_obs`.
///
/// # Examples
///
/// ```
/// use fx_var_engine::backtest::{kupiec_pof, KupiecInput};
/// let r = kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: 8, alpha: 0.01 }).unwrap();
/// assert!(r.p < 0.01);
/// ```
pub fn kupiec_pof(input: &KupiecInput) -> Result<KupiecResult> {
    check_coverage_input(input)?;
    let lr = kupiec_lr(input);
    Ok(LrTest {
        lr,
        p: chi2_sf(lr, 1.0)?,
    })
}

/// Christoffersen (1998) independence test on a 0/1 exception series
/// (first-order Markov alternative); `chi2(1)`. Degenerate series (no
/// transitions of one kind) return `LR = 0`.
///
/// # Errors
/// [`FxVarError::Invalid`] for fewer than 2 observations or non-0/1 values.
pub fn christoffersen_independence(exceedances: &[i32]) -> Result<LrTest> {
    if exceedances.len() < 2 {
        return Err(FxVarError::invalid(
            "need at least 2 observations for the independence test",
        ));
    }
    if exceedances.iter().any(|&e| e != 0 && e != 1) {
        return Err(FxVarError::invalid(
            "exceedances must be a 0/1 (or boolean) series",
        ));
    }
    let (mut n00, mut n01, mut n10, mut n11) = (0.0, 0.0, 0.0, 0.0);
    for t in 1..exceedances.len() {
        match (exceedances[t - 1], exceedances[t]) {
            (0, 0) => n00 += 1.0,
            (0, 1) => n01 += 1.0,
            (1, 0) => n10 += 1.0,
            _ => n11 += 1.0, // (1, 1): values were checked to be 0/1 above
        }
    }
    let pi01 = if n00 + n01 > 0.0 {
        n01 / (n00 + n01)
    } else {
        0.0
    };
    let pi11 = if n10 + n11 > 0.0 {
        n11 / (n10 + n11)
    } else {
        0.0
    };
    let pi = (n01 + n11) / (n00 + n01 + n10 + n11);
    let ll0 = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi);
    let ll1 = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11);
    let raw = -2.0 * (ll0 - ll1);
    let lr = if raw.is_finite() { raw.max(0.0) } else { 0.0 };
    Ok(LrTest {
        lr,
        p: chi2_sf(lr, 1.0)?,
    })
}

/// Christoffersen conditional coverage: `LR_cc = LR_uc + LR_ind`, `chi2(2)`.
///
/// # Errors
/// Same as [`kupiec_pof`] / [`christoffersen_independence`].
pub fn conditional_coverage(exceedances: &[i32], alpha: f64) -> Result<LrTest> {
    let x: i64 = exceedances.iter().map(|&e| e as i64).sum();
    let uc = kupiec_pof(&KupiecInput {
        n_obs: exceedances.len() as i64,
        n_exceptions: x,
        alpha,
    })?;
    let ind = christoffersen_independence(exceedances)?;
    let lr = uc.lr + ind.lr;
    Ok(LrTest {
        lr,
        p: chi2_sf(lr, 2.0)?,
    })
}

/// Basel traffic-light zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    /// Cumulative exception probability `< 95%`.
    Green,
    /// Cumulative exception probability in `[95%, 99.99%)`.
    Yellow,
    /// Cumulative exception probability `>= 99.99%`.
    Red,
}

/// Basel traffic-light outcome for a backtest window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrafficLight {
    /// Zone (green/yellow/red).
    pub zone: Zone,
    /// Observed exception count.
    pub n_exceptions: i64,
    /// Window length.
    pub n_obs: i64,
    /// `F(x) = P(X <= x)` under `Binomial(n, alpha)`.
    pub cumulative_prob: f64,
    /// Plus factor: 0 in green, 0.40 to 0.85 in yellow, 1 in red.
    pub plus_factor: f64,
    /// Capital multiplier `3 + plus_factor`.
    pub multiplier: f64,
}

/// Basel traffic light for any window and any tail probability (S4).
///
/// Zone from `F(x)`: green `< 0.95`, yellow `< 0.9999`, red otherwise. In
/// yellow the plus factor is the 1996 table entry at the equivalent
/// 250-day count `x'`, the smallest `x'` in `5..=9` with
/// `F_250(x') >= F(x) - 1e-12` (`9` if none). The old code took the add-on
/// from the raw count, so a yellow 500-day window with 8 exceptions got
/// the 250-day add-on for 8.
///
/// # Errors
/// [`FxVarError::Invalid`] for invalid counts or `alpha`;
/// [`FxVarError::Numerical`] if `pmf(0)` underflows.
///
/// # Examples
///
/// ```
/// use fx_var_engine::backtest::{basel_traffic_light, BaselInput, Zone};
/// let t = basel_traffic_light(&BaselInput { n_obs: 250, n_exceptions: 5, alpha: 0.01 }).unwrap();
/// assert_eq!(t.zone, Zone::Yellow);
/// assert!((t.multiplier - 3.40).abs() < 1e-12);
/// ```
pub fn basel_traffic_light(input: &BaselInput) -> Result<TrafficLight> {
    check_coverage_input(input)?;
    let (z, plus, f) = basel_rule(input)?;
    let zone = match z {
        0 => Zone::Green,
        1 => Zone::Yellow,
        _ => Zone::Red,
    };
    Ok(TrafficLight {
        zone,
        n_exceptions: input.n_exceptions,
        n_obs: input.n_obs,
        cumulative_prob: f,
        plus_factor: plus,
        multiplier: 3.0 + plus,
    })
}

/// Full VaR backtest summary over a forecast/realisation window.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestResult {
    /// Number of observations.
    pub n_obs: usize,
    /// Number of exceptions.
    pub n_exceptions: i64,
    /// Realised exception rate (`n_exceptions / n_obs`).
    pub exception_rate: f64,
    /// Expected exception rate under H0 (`alpha`).
    pub expected_rate: f64,
    /// Kupiec POF test.
    pub kupiec: LrTest,
    /// Christoffersen independence test.
    pub independence: LrTest,
    /// Christoffersen conditional coverage test.
    pub conditional: LrTest,
    /// Basel traffic-light outcome.
    pub traffic_light: TrafficLight,
    /// Per-day 0/1 exception indicator.
    pub exceedances: Vec<i32>,
}

/// Score a realised P&L series (profit +) against ex-ante positive VaR
/// forecasts for the same days.
///
/// # Errors
/// [`FxVarError::Invalid`] on a length mismatch, fewer than 2 observations,
/// a negative VaR forecast, or non-finite values in either series (NaN
/// policy: refuse; see the note below on why `is_nan` alone is not
/// enough).
pub fn evaluate_var_backtest(
    pnl: &[f64],
    var_forecasts: &[f64],
    alpha: f64,
) -> Result<BacktestResult> {
    validate_alpha(alpha)?;
    if pnl.len() != var_forecasts.len() {
        return Err(FxVarError::invalid(
            "pnl and var_forecasts must have equal length",
        ));
    }
    if pnl.len() < 2 {
        return Err(FxVarError::invalid(
            "need at least 2 observations to backtest",
        ));
    }
    // `is_finite`, not `!is_nan`. The exception test below is `-p > v`:
    // with a NaN on either side it is FALSE, so the day is silently
    // counted as "no exception" - a model whose VaR feed has broken to NaN
    // records ZERO breaches and sails through Kupiec, Christoffersen and
    // the Basel traffic light in the green zone. `+inf` on either side
    // does the same thing. Both must be errors.
    if pnl
        .iter()
        .chain(var_forecasts.iter())
        .any(|v| !v.is_finite())
    {
        return Err(FxVarError::invalid(
            "backtest inputs contain NaN or infinite values; a non-finite day \
             compares false against any threshold and would be silently scored as \
             'no exception' (NaN policy: refuse)",
        ));
    }
    if var_forecasts.iter().any(|v| *v < 0.0) {
        return Err(FxVarError::invalid(
            "var_forecasts must be non-negative (positive-loss convention); a \
             negative forecast scores profitable days as exceptions",
        ));
    }

    let exceedances: Vec<i32> = pnl
        .iter()
        .zip(var_forecasts)
        .map(|(&p, &v)| if -p > v { 1 } else { 0 })
        .collect();
    let x: i64 = exceedances.iter().map(|&e| e as i64).sum();
    let n = pnl.len();

    let input = KupiecInput {
        n_obs: n as i64,
        n_exceptions: x,
        alpha,
    };
    let kupiec = kupiec_pof(&input)?;
    let independence = christoffersen_independence(&exceedances)?;
    let lr_cc = kupiec.lr + independence.lr;
    let conditional = LrTest {
        lr: lr_cc,
        p: chi2_sf(lr_cc, 2.0)?,
    };
    let traffic_light = basel_traffic_light(&input)?;

    Ok(BacktestResult {
        n_obs: n,
        n_exceptions: x,
        exception_rate: x as f64 / n as f64,
        expected_rate: alpha,
        kupiec,
        independence,
        conditional,
        traffic_light,
        exceedances,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kupiec_zero_exceptions_at_nominal_rate() {
        // 2.5 exceptions expected in 250 at 99%; exactly 2 or 3 should not
        // strongly reject.
        let r = kupiec_pof(&KupiecInput {
            n_obs: 250,
            n_exceptions: 3,
            alpha: 0.01,
        })
        .unwrap();
        assert!(r.p > 0.05);
    }

    #[test]
    fn kupiec_excess_exceptions_reject() {
        let r = kupiec_pof(&KupiecInput {
            n_obs: 250,
            n_exceptions: 30,
            alpha: 0.01,
        })
        .unwrap();
        assert!(r.p < 0.01);
    }

    #[test]
    fn christoffersen_clustered_exceptions_reject_independence() {
        let mut e = vec![0; 100];
        // Cluster all 10 exceptions together (t=50..60).
        for i in 50..60 {
            e[i] = 1;
        }
        let r = christoffersen_independence(&e).unwrap();
        assert!(r.lr > 0.0);
    }

    #[test]
    fn basel_zones_at_boundaries() {
        let green = basel_traffic_light(&BaselInput {
            n_obs: 250,
            n_exceptions: 4,
            alpha: 0.01,
        })
        .unwrap();
        assert_eq!(green.zone, Zone::Green);
        let yellow = basel_traffic_light(&BaselInput {
            n_obs: 250,
            n_exceptions: 5,
            alpha: 0.01,
        })
        .unwrap();
        assert_eq!(yellow.zone, Zone::Yellow);
        assert!((yellow.multiplier - 3.40).abs() < 1e-12);
        let red = basel_traffic_light(&BaselInput {
            n_obs: 250,
            n_exceptions: 10,
            alpha: 0.01,
        })
        .unwrap();
        assert_eq!(red.zone, Zone::Red);
        assert!((red.multiplier - 4.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_backtest_end_to_end() {
        let n = 250;
        let mut pnl = vec![0.0; n];
        let var = vec![100.0; n];
        for (t, p) in pnl.iter_mut().enumerate() {
            *p = if t % 37 == 5 { -150.0 } else { -10.0 };
        }
        let r = evaluate_var_backtest(&pnl, &var, 0.01).unwrap();
        assert_eq!(r.n_obs, n);
        assert!(r.n_exceptions > 0);
        assert!(r.traffic_light.n_obs == n as i64);
    }

    #[test]
    fn invalid_inputs_error() {
        assert!(
            kupiec_pof(&KupiecInput {
                n_obs: 250,
                n_exceptions: -1,
                alpha: 0.01
            })
            .is_err()
        );
        assert!(
            kupiec_pof(&KupiecInput {
                n_obs: 250,
                n_exceptions: 251,
                alpha: 0.01
            })
            .is_err()
        );
        assert!(christoffersen_independence(&[0]).is_err());
        assert!(christoffersen_independence(&[0, 2]).is_err());
        assert!(evaluate_var_backtest(&[1.0], &[1.0, 2.0], 0.01).is_err());
        assert!(evaluate_var_backtest(&[f64::NAN, 1.0], &[1.0, 1.0], 0.01).is_err());
    }
}
