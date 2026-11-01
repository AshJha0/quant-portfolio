//! VaR backtesting: Kupiec POF, Christoffersen independence / conditional
//! coverage, Basel traffic light.
//!
//! A backtest compares ex-ante VaR forecasts against realised P&L: an
//! **exception** is a day with `pnl < -VaR`. LR statistics are
//! asymptotically chi-squared under H0; p-values come from the regularized
//! upper incomplete gamma ([`crate::stats::chi2_sf`]), and the Basel zone
//! probabilities from the exact binomial CDF by pmf recursion. The ES
//! backtest is the Acerbi-Szekely `Z2` statistic ([`acerbi_szekely_z2`]).
//! Mirrors `eq_var.backtesting`.

use crate::stats::chi2_sf;
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
/// otherwise it must have one entry per day. Errors on a negative VaR, a
/// non-finite P&L or VaR, or a size mismatch.
///
/// # Why non-finite input is rejected rather than skipped
///
/// The exception test is `pnl_t < -var_t`. If either side is NaN the
/// comparison is **false**, so the day is silently counted as "no
/// exception". A model whose VaR feed has broken to NaN therefore records
/// **zero breaches** and sails through Kupiec, Christoffersen and the
/// Basel traffic light in the green zone: the exact opposite of what a
/// backtest is for. A NaN in either series is an error here, so the
/// breakage surfaces on the day it happens.
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
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "exceptions_from_pnl: pnl contains NaN or infinite values; a NaN day              compares false against any VaR and would be silently counted as              'no exception'"
                .to_string(),
        ));
    }
    if var.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "exceptions_from_pnl: VaR contains NaN or infinite values; a NaN VaR              is never breached and would let a broken model pass Kupiec and the              Basel traffic light with zero exceptions"
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

/// Kupiec POF (unconditional coverage) test (S3).
///
/// `LR = -2 ln[(1-a)^(n-x) a^x] + 2 ln[(1-p)^(n-x) p^x]`, `p = x/n`,
/// `0 ln 0 = 0`; the p-value is the chi2(1) upper tail.
///
/// # Errors
/// `alpha` outside `(0, 0.5)`, `n_obs < 1`, `n_exceptions < 0`, or
/// `n_exceptions > n_obs` (no clamping).
///
/// # Examples
///
/// ```
/// use eq_var_engine::backtest::{kupiec_pof, KupiecInput};
/// let r = kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: 5, alpha: 0.01 }).unwrap();
/// assert!(r.lr > 0.0 && r.pvalue > 0.05);
/// ```
pub fn kupiec_pof(input: &KupiecInput) -> Result<KupiecResult> {
    check_coverage_input(input)?;
    let lr = kupiec_lr(input);
    let t = input.n_obs as f64;
    Ok(KupiecResult {
        lr,
        pvalue: chi2_sf(lr, 1.0)?,
        expected: input.alpha * t,
        rate: input.n_exceptions as f64 / t,
    })
}

/// Old positional Kupiec call `(n_obs, n_exceptions, alpha)`, kept for one
/// release. It forwards to [`kupiec_pof`], so it now rejects `alpha >= 0.5`.
#[deprecated(since = "0.2.0", note = "use kupiec_pof(&KupiecInput { .. })")]
pub fn kupiec_pof_positional(n_obs: i64, n_exceptions: i64, alpha: f64) -> Result<KupiecResult> {
    kupiec_pof(&KupiecInput { n_obs, n_exceptions, alpha })
}

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
        return Err(EqVarError::InvalidInput(format!("n_obs must be >= 1, got {}", inp.n_obs)));
    }
    if inp.n_exceptions < 0 {
        return Err(EqVarError::InvalidInput(format!(
            "n_exceptions must be >= 0, got {}",
            inp.n_exceptions
        )));
    }
    if inp.n_exceptions > inp.n_obs {
        return Err(EqVarError::InvalidInput(format!(
            "n_exceptions ({}) exceeds n_obs ({})",
            inp.n_exceptions, inp.n_obs
        )));
    }
    if inp.n_obs > (1_i64 << 53) {
        return Err(EqVarError::InvalidInput(format!(
            "n_obs {} exceeds the exactly representable range",
            inp.n_obs
        )));
    }
    Ok(())
}

/// `x ln y` with `0 ln 0 = 0`.
fn xlny(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
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
        return Err(EqVarError::Numerical(format!(
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
/// # use eq_var_engine::backtest::acerbi_szekely_z2;
/// // Two exceptions out of four days, each exactly at ES: Z2 = 1 - 2/(4 a).
/// let z2 = acerbi_szekely_z2(&[-2.0, 0.0, -2.0, 1.0], &[1.0], &[2.0], 0.25).unwrap();
/// assert!((z2 - (1.0 - 2.0)).abs() < 1e-12);
/// ```
pub fn acerbi_szekely_z2(pnl: &[f64], var: &[f64], es: &[f64], alpha: f64) -> Result<f64> {
    crate::tail::validate_alpha(alpha)?;
    let t = pnl.len();
    if t == 0 {
        return Err(EqVarError::InvalidInput("acerbi_szekely_z2: empty pnl".to_string()));
    }
    for (name, s) in [("var", var), ("es", es)] {
        if s.len() != 1 && s.len() != t {
            return Err(EqVarError::InvalidInput(format!(
                "acerbi_szekely_z2: {name} must have length 1 or {t}, got {}",
                s.len()
            )));
        }
    }
    if pnl.iter().chain(var).chain(es).any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "acerbi_szekely_z2: inputs contain NaN or infinite values".to_string(),
        ));
    }
    if var.iter().any(|v| *v < 0.0) {
        return Err(EqVarError::InvalidInput("acerbi_szekely_z2: VaR must be >= 0".to_string()));
    }
    if es.iter().any(|v| !(*v > 0.0)) {
        return Err(EqVarError::InvalidInput("acerbi_szekely_z2: ES must be > 0".to_string()));
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

/// Conditional-coverage joint test: `LR_cc = LR_uc + LR_ind ~ chi2(2)`:
/// correct exception rate **and** independence together.
pub fn christoffersen_cc(exceptions: &[u8], alpha: f64) -> Result<ConditionalCoverageResult> {
    let count: i64 = exceptions.iter().map(|&e| i64::from(e != 0)).sum();
    let uc = kupiec_pof(&KupiecInput {
        n_obs: exceptions.len() as i64,
        n_exceptions: count,
        alpha,
    })?;
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
    /// `F(x) < 0.95`: multiplier 3.0.
    Green,
    /// `0.95 <= F(x) < 0.9999`: multiplier 3 + plus factor (0.40 to 0.85).
    Yellow,
    /// `F(x) >= 0.9999`: multiplier 4.0, presumption of a flawed model.
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
    /// Capital multiplier `k = 3 + plus_factor`.
    pub multiplier: f64,
    /// Plus factor: 0 in green, 0.40 to 0.85 in yellow, 1 in red.
    pub plus_factor: f64,
    /// `F(x) = P(X <= n_exceptions)`, `X ~ Binomial(n_obs, alpha)`.
    pub cumulative_prob: f64,
}

/// Basel traffic light for any window and any tail probability (S4).
///
/// `F(x) = P(X <= x)`, `X ~ Binomial(n_obs, alpha)` by the pmf recursion.
/// Green if `F(x) < 0.95`, yellow if `F(x) < 0.9999`, red otherwise. In
/// yellow the plus factor is read from the 1996 table at the equivalent
/// 250-day count: the smallest `x'` in `5..=9` with
/// `F_250(x') >= F(x) - 1e-12`, `F_250` the `Binomial(250, 0.01)` CDF
/// (`x' = 9` if none). On the standard window `(250, 0.01)` this gives the
/// table: 0-4 green, 5-9 yellow with 0.40/0.50/0.65/0.75/0.85, 10+ red.
///
/// # Errors
/// Same input rules as [`kupiec_pof`]; a numerical error if `pmf(0)`
/// underflows (windows with `n_obs * alpha` beyond about 700).
///
/// # Examples
///
/// ```
/// use eq_var_engine::backtest::{basel_traffic_light, BaselInput, BaselZone};
/// let r = basel_traffic_light(&BaselInput { n_obs: 250, n_exceptions: 5, alpha: 0.01 }).unwrap();
/// assert_eq!(r.zone, BaselZone::Yellow);
/// assert!((r.multiplier - 3.40).abs() < 1e-12);
/// // 5 exceptions in 500 days is green: the zone follows the window.
/// let r = basel_traffic_light(&BaselInput { n_obs: 500, n_exceptions: 5, alpha: 0.01 }).unwrap();
/// assert_eq!(r.zone, BaselZone::Green);
/// ```
pub fn basel_traffic_light(input: &BaselInput) -> Result<TrafficLightResult> {
    check_coverage_input(input)?;
    let (z, plus, f) = basel_rule(input)?;
    let zone = match z {
        0 => BaselZone::Green,
        1 => BaselZone::Yellow,
        _ => BaselZone::Red,
    };
    Ok(TrafficLightResult {
        zone,
        multiplier: 3.0 + plus,
        plus_factor: plus,
        cumulative_prob: f,
    })
}

/// Old 250-day call `(n_exceptions, n_obs)` at alpha 0.01, kept for one
/// release. It forwards to [`basel_traffic_light`]: the zone now follows
/// the binomial rule for the given window, and `n_exceptions > n_obs` is
/// an error instead of being clamped.
#[deprecated(since = "0.2.0", note = "use basel_traffic_light(&BaselInput { .. })")]
pub fn basel_traffic_light_250(n_exceptions: i64, n_obs: i64) -> Result<TrafficLightResult> {
    basel_traffic_light(&BaselInput { n_obs, n_exceptions, alpha: 0.01 })
}
