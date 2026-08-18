//! VaR backtesting: Kupiec POF, Christoffersen independence, Basel traffic
//! light (mirrors Python `fx_var.backtesting` / C++ `fxvar::backtest`).
//!
//! Exception convention: day `t` is an exception when the realised *loss*
//! exceeds the VaR forecast made ex ante for day `t`: `-pnl_t > var_t`.
//!
//! # Tests
//!
//! * **Kupiec POF** (unconditional coverage) — LR test that the exception
//!   frequency equals `1 - alpha`; `chi2(1)` via the regularised
//!   incomplete gamma.
//! * **Christoffersen independence** — LR test against first-order Markov
//!   clustering of exceptions; `chi2(1)`. FX desks care because volatility
//!   clustering makes an unconditional method (plain HS, sample-cov
//!   parametric) fail *this* test long before it fails Kupiec.
//! * **Conditional coverage** — `LR_cc = LR_uc + LR_ind`; `chi2(2)`.
//! * **Basel traffic light** — zones from the exact cumulative
//!   `Binomial(n, 1-alpha)` probability of the exception count: green
//!   `< 95%`, yellow `< 99.99%`, red `>= 99.99%`, which for the regulatory
//!   250-day 99% window reproduces the table exactly: green 0-4, yellow
//!   5-9 (add-ons 0.40/0.50/0.65/0.75/0.85), red 10+ (multiplier 4.0).

use crate::stats::{binom_cdf, chi2_sf, validate_alpha};
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

fn check_counts(x: i64, n: i64) -> Result<()> {
    if n <= 0 || x < 0 || x > n {
        return Err(FxVarError::invalid(format!("need 0 <= n_exceptions <= n_obs, got x={x}, n={n}")));
    }
    Ok(())
}

/// A likelihood-ratio test outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrTest {
    /// Test statistic (chi-square distributed under H0).
    pub lr: f64,
    /// Survival-function p-value.
    pub p: f64,
}

/// Kupiec proportion-of-failures test: `LR_uc ~ chi2(1)` under H0 that the
/// exception probability is `1 - alpha`.
///
/// # Errors
/// [`FxVarError::Invalid`] for invalid counts or `alpha`.
pub fn kupiec_pof(n_exceptions: i64, n_obs: i64, alpha: f64) -> Result<LrTest> {
    validate_alpha(alpha)?;
    check_counts(n_exceptions, n_obs)?;
    let x = n_exceptions as f64;
    let n = n_obs as f64;
    let p = 1.0 - alpha;
    let pi_hat = x / n;
    let ll0 = xlogy(n - x, 1.0 - p) + xlogy(x, p);
    let ll1 = xlogy(n - x, 1.0 - pi_hat) + xlogy(x, pi_hat);
    let lr = (-2.0 * (ll0 - ll1)).max(0.0);
    Ok(LrTest { lr, p: chi2_sf(lr, 1.0)? })
}

/// Christoffersen (1998) independence test on a 0/1 exception series
/// (first-order Markov alternative); `chi2(1)`. Degenerate series (no
/// transitions of one kind) return `LR = 0`.
///
/// # Errors
/// [`FxVarError::Invalid`] for fewer than 2 observations or non-0/1 values.
pub fn christoffersen_independence(exceedances: &[i32]) -> Result<LrTest> {
    if exceedances.len() < 2 {
        return Err(FxVarError::invalid("need at least 2 observations for the independence test"));
    }
    if exceedances.iter().any(|&e| e != 0 && e != 1) {
        return Err(FxVarError::invalid("exceedances must be a 0/1 (or boolean) series"));
    }
    let (mut n00, mut n01, mut n10, mut n11) = (0.0, 0.0, 0.0, 0.0);
    for t in 1..exceedances.len() {
        match (exceedances[t - 1], exceedances[t]) {
            (0, 0) => n00 += 1.0,
            (0, 1) => n01 += 1.0,
            (1, 0) => n10 += 1.0,
            (1, 1) => n11 += 1.0,
            _ => unreachable!(),
        }
    }
    let pi01 = if n00 + n01 > 0.0 { n01 / (n00 + n01) } else { 0.0 };
    let pi11 = if n10 + n11 > 0.0 { n11 / (n10 + n11) } else { 0.0 };
    let pi = (n01 + n11) / (n00 + n01 + n10 + n11);
    let ll0 = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi);
    let ll1 = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11);
    let raw = -2.0 * (ll0 - ll1);
    let lr = if raw.is_finite() { raw.max(0.0) } else { 0.0 };
    Ok(LrTest { lr, p: chi2_sf(lr, 1.0)? })
}

/// Christoffersen conditional coverage: `LR_cc = LR_uc + LR_ind`, `chi2(2)`.
///
/// # Errors
/// Same as [`kupiec_pof`] / [`christoffersen_independence`].
pub fn conditional_coverage(exceedances: &[i32], alpha: f64) -> Result<LrTest> {
    let x: i64 = exceedances.iter().map(|&e| e as i64).sum();
    let uc = kupiec_pof(x, exceedances.len() as i64, alpha)?;
    let ind = christoffersen_independence(exceedances)?;
    let lr = uc.lr + ind.lr;
    Ok(LrTest { lr, p: chi2_sf(lr, 2.0)? })
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
    /// `P(X <= x)` under `Binomial(n, 1-alpha)`.
    pub cumulative_prob: f64,
    /// Capital multiplier, `3.0` .. `4.0`.
    pub multiplier: f64,
}

/// Basel traffic-light zone and capital multiplier (1996 table add-ons in
/// the yellow zone; multiplier capped at 4.0 in red).
///
/// # Errors
/// [`FxVarError::Invalid`] for invalid counts or `alpha`.
pub fn basel_traffic_light(n_exceptions: i64, n_obs: i64, alpha: f64) -> Result<TrafficLight> {
    validate_alpha(alpha)?;
    check_counts(n_exceptions, n_obs)?;
    let cumulative_prob = binom_cdf(n_exceptions, n_obs, 1.0 - alpha)?;
    let (zone, multiplier) = if cumulative_prob < 0.95 {
        (Zone::Green, 3.0)
    } else if cumulative_prob < 0.9999 {
        // 1996 Basel yellow-zone add-ons by exception count (250d, 99%):
        // 5 -> 0.40, 6 -> 0.50, 7 -> 0.65, 8 -> 0.75, 9 -> 0.85. Yellow at
        // an out-of-table count (non-standard window) takes the nearest
        // boundary add-on, mirroring the Python reference.
        let addon = match n_exceptions {
            5 => 0.40,
            6 => 0.50,
            7 => 0.65,
            8 => 0.75,
            9 => 0.85,
            n if n > 9 => 0.85,
            _ => 0.40,
        };
        (Zone::Yellow, 3.0 + addon)
    } else {
        (Zone::Red, 4.0)
    };
    Ok(TrafficLight { zone, n_exceptions, n_obs, cumulative_prob, multiplier })
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
    /// Expected exception rate under H0 (`1 - alpha`).
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
/// or NaNs (NaN policy: refuse).
pub fn evaluate_var_backtest(pnl: &[f64], var_forecasts: &[f64], alpha: f64) -> Result<BacktestResult> {
    validate_alpha(alpha)?;
    if pnl.len() != var_forecasts.len() {
        return Err(FxVarError::invalid("pnl and var_forecasts must have equal length"));
    }
    if pnl.len() < 2 {
        return Err(FxVarError::invalid("need at least 2 observations to backtest"));
    }
    if pnl.iter().chain(var_forecasts.iter()).any(|v| v.is_nan()) {
        return Err(FxVarError::invalid("backtest inputs contain NaNs (NaN policy: refuse)"));
    }

    let exceedances: Vec<i32> =
        pnl.iter().zip(var_forecasts).map(|(&p, &v)| if -p > v { 1 } else { 0 }).collect();
    let x: i64 = exceedances.iter().map(|&e| e as i64).sum();
    let n = pnl.len();

    let kupiec = kupiec_pof(x, n as i64, alpha)?;
    let independence = christoffersen_independence(&exceedances)?;
    let lr_cc = kupiec.lr + independence.lr;
    let conditional = LrTest { lr: lr_cc, p: chi2_sf(lr_cc, 2.0)? };
    let traffic_light = basel_traffic_light(x, n as i64, alpha)?;

    Ok(BacktestResult {
        n_obs: n,
        n_exceptions: x,
        exception_rate: x as f64 / n as f64,
        expected_rate: 1.0 - alpha,
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
        let r = kupiec_pof(3, 250, 0.99).unwrap();
        assert!(r.p > 0.05);
    }

    #[test]
    fn kupiec_excess_exceptions_reject() {
        let r = kupiec_pof(30, 250, 0.99).unwrap();
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
        let green = basel_traffic_light(4, 250, 0.99).unwrap();
        assert_eq!(green.zone, Zone::Green);
        let yellow = basel_traffic_light(5, 250, 0.99).unwrap();
        assert_eq!(yellow.zone, Zone::Yellow);
        assert!((yellow.multiplier - 3.40).abs() < 1e-12);
        let red = basel_traffic_light(10, 250, 0.99).unwrap();
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
        let r = evaluate_var_backtest(&pnl, &var, 0.99).unwrap();
        assert_eq!(r.n_obs, n);
        assert!(r.n_exceptions > 0);
        assert!(r.traffic_light.n_obs == n as i64);
    }

    #[test]
    fn invalid_inputs_error() {
        assert!(kupiec_pof(-1, 250, 0.99).is_err());
        assert!(kupiec_pof(251, 250, 0.99).is_err());
        assert!(christoffersen_independence(&[0]).is_err());
        assert!(christoffersen_independence(&[0, 2]).is_err());
        assert!(evaluate_var_backtest(&[1.0], &[1.0, 2.0], 0.99).is_err());
        assert!(evaluate_var_backtest(&[f64::NAN, 1.0], &[1.0, 1.0], 0.99).is_err());
    }
}
