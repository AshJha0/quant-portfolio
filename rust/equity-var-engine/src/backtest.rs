//! VaR backtesting: Kupiec POF, Christoffersen independence / conditional
//! coverage, Basel traffic light.
//!
//! A backtest compares ex-ante VaR forecasts against realised P&L: an
//! **exception** is a day with `pnl < -VaR`. LR statistics are
//! asymptotically chi-squared under H0; p-values come from the regularized
//! upper incomplete gamma ([`crate::stats::chi2_sf`]), and the Basel zone
//! probabilities from the exact binomial CDF via the regularized incomplete
//! beta. Mirrors `eq_var.backtesting`.

use crate::stats::{binomial_cdf, chi2_sf};
use crate::{EqVarError, Result};

/// `0 * ln 0 = 0` convention shared by the Kupiec and Christoffersen
/// binomial/Markov log-likelihoods.
fn xlogy(a: f64, b: f64) -> f64 {
    if a > 0.0 && b > 0.0 {
        a * b.ln()
    } else {
        0.0
    }
}

/// Exception indicator per day: `1` iff `pnl_t < -var_t`.
///
/// `var` (positive-loss convention) is scalar-broadcast if it has length 1,
/// otherwise it must have one entry per day. Errors on a negative VaR or a
/// size mismatch.
///
/// # Examples
///
/// ```
/// use eq_var_engine::backtest::exceptions_from_pnl;
/// let pnl = [-120.0, 50.0, -99.9, -100.1, 0.0];
/// let ex = exceptions_from_pnl(&pnl, &[100.0]).unwrap();
/// assert_eq!(ex, vec![1, 0, 0, 1, 0]);
/// ```
pub fn exceptions_from_pnl(pnl: &[f64], var: &[f64]) -> Result<Vec<u8>> {
    if pnl.is_empty() {
        return Err(EqVarError::InvalidInput(
            "exceptions_from_pnl: empty pnl".to_string(),
        ));
    }
    if var.len() != 1 && var.len() != pnl.len() {
        return Err(EqVarError::InvalidInput(
            "exceptions_from_pnl: var must be scalar (length 1) or one entry per day"
                .to_string(),
        ));
    }
    if var.iter().any(|v| *v < 0.0) {
        return Err(EqVarError::InvalidInput(
            "exceptions_from_pnl: VaR must be a positive loss".to_string(),
        ));
    }
    Ok(pnl
        .iter()
        .enumerate()
        .map(|(t, p)| {
            let v = if var.len() == 1 { var[0] } else { var[t] };
            u8::from(*p < -v)
        })
        .collect())
}

/// Kupiec (1995) proportion-of-failures likelihood-ratio test result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KupiecResult {
    /// `LR_uc` statistic, chi2(1) under H0.
    pub lr: f64,
    /// chi2(1) survival probability of `lr`.
    pub pvalue: f64,
    /// `alpha * n_obs` expected exceptions.
    pub expected: f64,
    /// Observed exception rate `x / T`.
    pub rate: f64,
}

/// Kupiec POF (unconditional coverage) test.
///
/// `LR_uc = -2 ln[ (1-p)^(T-x) p^x / ((1-x/T)^(T-x) (x/T)^x) ]` with
/// `p = alpha`, `T = n_obs`, `x = n_exceptions`; `x = 0` and `x = T` use the
/// `0 ln 0 = 0` convention. Errors on bad counts (`n_obs < 1`, `n_exceptions`
/// outside `[0, n_obs]`) or `alpha` outside `(0, 1)`.
pub fn kupiec_pof(n_obs: i64, n_exceptions: i64, alpha: f64) -> Result<KupiecResult> {
    if n_obs < 1 {
        return Err(EqVarError::InvalidInput(format!(
            "kupiec_pof: n_obs must be >= 1, got {n_obs}"
        )));
    }
    if n_exceptions < 0 || n_exceptions > n_obs {
        return Err(EqVarError::InvalidInput(
            "kupiec_pof: n_exceptions must be in [0, n_obs]".to_string(),
        ));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "kupiec_pof: alpha must be in (0, 1), got {alpha}"
        )));
    }
    let t = n_obs as f64;
    let x = n_exceptions as f64;
    let pihat = x / t;

    let ll = |p: f64| xlogy(t - x, 1.0 - p) + xlogy(x, p);
    let ll_alt = if pihat == 0.0 || pihat == 1.0 { 0.0 } else { ll(pihat) };
    let lr = (-2.0 * (ll(alpha) - ll_alt)).max(0.0);

    Ok(KupiecResult {
        lr,
        pvalue: chi2_sf(lr, 1.0)?,
        expected: alpha * t,
        rate: pihat,
    })
}

/// Christoffersen (1998) independence test result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChristoffersenResult {
    /// `LR_ind` statistic, chi2(1) under independence.
    pub lr: f64,
    /// chi2(1) survival probability of `lr`.
    pub pvalue: f64,
    /// Count of (no exception -> no exception) transitions.
    pub n00: f64,
    /// Count of (no exception -> exception) transitions.
    pub n01: f64,
    /// Count of (exception -> no exception) transitions.
    pub n10: f64,
    /// Count of (exception -> exception) transitions.
    pub n11: f64,
    /// `P(exception today | none yesterday)`.
    pub pi01: f64,
    /// `P(exception today | exception yesterday)`.
    pub pi11: f64,
}

/// Independence LR against a first-order Markov alternative.
///
/// Clustered exceptions inflate `n11` and reject independence. Degenerate
/// transition counts use the `0 ln 0 = 0` convention. Requires at least 2
/// observations.
pub fn christoffersen_independence(exceptions: &[u8]) -> Result<ChristoffersenResult> {
    if exceptions.len() < 2 {
        return Err(EqVarError::InvalidInput(
            "christoffersen_independence: need at least 2 observations".to_string(),
        ));
    }
    let (mut n00, mut n01, mut n10, mut n11) = (0.0, 0.0, 0.0, 0.0);
    for w in exceptions.windows(2) {
        let (prev, curr) = (w[0] != 0, w[1] != 0);
        match (prev, curr) {
            (false, false) => n00 += 1.0,
            (false, true) => n01 += 1.0,
            (true, false) => n10 += 1.0,
            (true, true) => n11 += 1.0,
        }
    }
    let n0 = n00 + n01;
    let n1 = n10 + n11;
    let pi01 = if n0 > 0.0 { n01 / n0 } else { 0.0 };
    let pi11 = if n1 > 0.0 { n11 / n1 } else { 0.0 };
    let pi = (n01 + n11) / (n0 + n1);
    let ll_markov =
        xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11);
    let ll_iid = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi);
    let lr = (-2.0 * (ll_iid - ll_markov)).max(0.0);
    Ok(ChristoffersenResult {
        lr,
        pvalue: chi2_sf(lr, 1.0)?,
        n00,
        n01,
        n10,
        n11,
        pi01,
        pi11,
    })
}

/// Conditional-coverage joint test result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalCoverageResult {
    /// `LR_uc + LR_ind`, chi2(2) under H0.
    pub lr: f64,
    /// chi2(2) survival probability of `lr`.
    pub pvalue: f64,
    /// The unconditional-coverage component.
    pub lr_uc: f64,
    /// The independence component.
    pub lr_ind: f64,
}

/// Conditional-coverage joint test: `LR_cc = LR_uc + LR_ind ~ chi2(2)` —
/// correct exception rate **and** independence together.
pub fn christoffersen_cc(exceptions: &[u8], alpha: f64) -> Result<ConditionalCoverageResult> {
    let count: i64 = exceptions.iter().map(|&e| i64::from(e != 0)).sum();
    let uc = kupiec_pof(exceptions.len() as i64, count, alpha)?;
    let ind = christoffersen_independence(exceptions)?;
    let lr = uc.lr + ind.lr;
    Ok(ConditionalCoverageResult {
        lr,
        pvalue: chi2_sf(lr, 2.0)?,
        lr_uc: uc.lr,
        lr_ind: ind.lr,
    })
}

/// Basel traffic-light zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselZone {
    /// 0-4 exceptions in 250 days: multiplier 3.0.
    Green,
    /// 5-9 exceptions: multiplier 3.40-3.85 (regulatory add-on ladder).
    Yellow,
    /// 10+ exceptions: multiplier 4.0, presumption of a flawed model.
    Red,
}

impl BaselZone {
    /// Human-readable zone name (`"green"` / `"yellow"` / `"red"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            BaselZone::Green => "green",
            BaselZone::Yellow => "yellow",
            BaselZone::Red => "red",
        }
    }
}

impl std::fmt::Display for BaselZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of the Basel traffic-light backtest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrafficLightResult {
    /// The zone (green / yellow / red).
    pub zone: BaselZone,
    /// Capital multiplier `k = 3 + add-on`.
    pub multiplier: f64,
    /// `P(X <= n_exceptions)`, `X ~ Binomial(n_obs, 0.01)`.
    pub cumulative_prob: f64,
}

/// Basel (1996 supervisory framework) traffic light for 99 % VaR on the
/// standard 250-day window.
///
/// 0-4 exceptions = green (multiplier 3.0), 5-9 = yellow (add-ons
/// 0.40 / 0.50 / 0.65 / 0.75 / 0.85), 10+ = red (add-on 1.0). Zone
/// boundaries are the regulatory **exact** counts; the cumulative binomial
/// probability of the observed count under a correct model is reported
/// alongside.
///
/// # Examples
///
/// ```
/// use eq_var_engine::backtest::{basel_traffic_light, BaselZone};
/// let r = basel_traffic_light(5, 250).unwrap();
/// assert_eq!(r.zone, BaselZone::Yellow);
/// assert!((r.multiplier - 3.40).abs() < 1e-12);
/// ```
pub fn basel_traffic_light(n_exceptions: i64, n_obs: i64) -> Result<TrafficLightResult> {
    if n_exceptions < 0 {
        return Err(EqVarError::InvalidInput(
            "basel_traffic_light: n_exceptions must be >= 0".to_string(),
        ));
    }
    if n_obs < 1 {
        return Err(EqVarError::InvalidInput(
            "basel_traffic_light: n_obs must be >= 1".to_string(),
        ));
    }
    const YELLOW_ADDON: [f64; 5] = [0.40, 0.50, 0.65, 0.75, 0.85];
    let (zone, addon) = if n_exceptions <= 4 {
        (BaselZone::Green, 0.0)
    } else if n_exceptions <= 9 {
        (BaselZone::Yellow, YELLOW_ADDON[(n_exceptions - 5) as usize])
    } else {
        (BaselZone::Red, 1.0)
    };
    let k = n_exceptions.min(n_obs) as u64;
    let cumulative_prob = binomial_cdf(k, n_obs as u64, 0.01)?;
    Ok(TrafficLightResult {
        zone,
        multiplier: 3.0 + addon,
        cumulative_prob,
    })
}
