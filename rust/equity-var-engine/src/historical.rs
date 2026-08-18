//! Historical-simulation VaR: plain, age-weighted (BRW) and filtered (FHS).
//!
//! Conventions (identical to the Python reference `eq_var.historical_var`):
//!
//! * `pnl` slices are P&L in currency units, **loss < 0**;
//! * `alpha` is the tail probability (`0.01` → 99 % VaR);
//! * VaR is reported as a **positive** number: `VaR_alpha = -Q_alpha(pnl)`;
//! * plain historical VaR uses NumPy's default `"linear"` (Hyndman–Fan
//!   type-7) interpolation between order statistics — the most common desk
//!   choice; alternatives (lower/higher order statistic) differ by `O(1/n)`
//!   and are discussed in the Python project's docs/METHODOLOGY.md;
//! * weighted quantiles (BRW) use the step-function inversion of the
//!   weighted empirical CDF.

use crate::{validate_alpha, EqVarError, Result};

/// Minimum P&L observations for a historical tail estimate: fewer than this
/// cannot resolve a 1–5 % tail sensibly.
pub const MIN_OBS: usize = 50;

fn validate_pnl(pnl: &[f64], min_obs: usize) -> Result<()> {
    if pnl.len() < min_obs {
        return Err(EqVarError::InvalidInput(format!(
            "need at least {min_obs} P&L observations for historical VaR, got {}; \
             empirical tail quantiles are meaningless on shorter samples",
            pnl.len()
        )));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "pnl contains NaN or infinite values".to_string(),
        ));
    }
    Ok(())
}

fn sorted_copy(pnl: &[f64]) -> Vec<f64> {
    let mut v = pnl.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite values compare totally"));
    v
}

/// Empirical quantile with NumPy-default linear (Hyndman–Fan type-7)
/// interpolation: `h = (n - 1) q`, result
/// `x_(floor(h)) + (h - floor(h)) * (x_(floor(h)+1) - x_(floor(h)))` on the
/// ascending order statistics.
///
/// `q` must be in `[0, 1]`; input needs at least one finite observation.
pub fn linear_quantile(pnl: &[f64], q: f64) -> Result<f64> {
    if pnl.is_empty() {
        return Err(EqVarError::InvalidInput(
            "quantile of an empty sample is undefined".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&q) {
        return Err(EqVarError::InvalidInput(format!(
            "quantile level must be in [0, 1], got {q}"
        )));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "pnl contains NaN or infinite values".to_string(),
        ));
    }
    let sorted = sorted_copy(pnl);
    Ok(linear_quantile_sorted(&sorted, q))
}

/// Type-7 quantile on an already ascending-sorted, non-empty slice.
pub(crate) fn linear_quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * q;
    let lo = h.floor() as usize;
    if lo + 1 >= n {
        return sorted[n - 1];
    }
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
}

/// Plain historical-simulation VaR (equal weights).
///
/// Returns `-quantile_alpha(pnl)` (positive for a loss) with linear
/// (type-7) interpolation between order statistics. Requires at least
/// [`MIN_OBS`] observations and `alpha` in `(0, 0.5)`.
///
/// # Examples
///
/// ```
/// use eq_var_engine::historical::historical_var;
/// let pnl: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
/// let var = historical_var(&pnl, 0.05).unwrap();
/// assert!((var - 95.05).abs() < 1e-12); // hand-computed type-7 quantile
/// ```
pub fn historical_var(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl(pnl, MIN_OBS)?;
    Ok(-linear_quantile(pnl, alpha)?)
}

/// Boudoukh–Richardson–Whitelaw exponential age weights.
///
/// Observation `i` (0 = oldest, `n - 1` = most recent) gets weight
/// proportional to `lam^{n-1-i}`; weights sum to 1 exactly:
/// `w_i = (1 - lam) lam^{n-1-i} / (1 - lam^n)`.
pub fn brw_weights(n: usize, lam: f64) -> Result<Vec<f64>> {
    if !(lam > 0.0 && lam < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "decay lam must be in (0, 1), got {lam}"
        )));
    }
    if n < 1 {
        return Err(EqVarError::InvalidInput("n must be >= 1".to_string()));
    }
    let norm = (1.0 - lam) / (1.0 - lam.powi(n as i32));
    Ok((0..n)
        .map(|i| norm * lam.powi((n - 1 - i) as i32))
        .collect())
}

/// Age-weighted (BRW) historical VaR.
///
/// Recent observations receive exponentially larger weight
/// (`w_t ~ lam^age`). The weighted empirical CDF is inverted at `alpha`:
/// VaR is minus the smallest P&L whose cumulative weight (ascending P&L
/// order, stable sort) reaches `alpha`.
///
/// `lam -> 1` recovers plain historical simulation up to the
/// interpolation-scheme difference (BRW uses the step-CDF inversion).
pub fn age_weighted_var(pnl: &[f64], alpha: f64, lam: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl(pnl, MIN_OBS)?;
    let w = brw_weights(pnl.len(), lam)?;
    // Stable argsort by P&L ascending (ties keep chronological order).
    let mut order: Vec<usize> = (0..pnl.len()).collect();
    order.sort_by(|&a, &b| pnl[a].partial_cmp(&pnl[b]).expect("finite values"));
    let mut cum = 0.0;
    let mut idx = pnl.len() - 1;
    for (rank, &i) in order.iter().enumerate() {
        cum += w[i];
        if cum >= alpha {
            idx = rank;
            break;
        }
    }
    Ok(-pnl[order[idx]])
}

/// One-step-ahead EWMA (RiskMetrics) volatility forecasts.
///
/// `sigma2[t] = lam * sigma2[t-1] + (1 - lam) * x[t-1]^2` — `sigma[t]` is
/// the forecast for day `t` made with information up to `t - 1`, so
/// standardising `x[t] / sigma[t]` uses no look-ahead. Seeded with the
/// full-sample population variance (`ddof = 0`, matching the Python
/// reference default `init="sample"`); a zero-variance series is floored at
/// 1e-16 to avoid downstream division by zero. Returns one `sigma[t] > 0`
/// per input observation.
pub fn ewma_volatility(x: &[f64], lam: f64) -> Result<Vec<f64>> {
    if !(lam > 0.0 && lam < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "decay lam must be in (0, 1), got {lam}"
        )));
    }
    if x.len() < 2 {
        return Err(EqVarError::InvalidInput(
            "need at least 2 observations for EWMA volatility".to_string(),
        ));
    }
    let mut seed = crate::stats::population_variance(x)?;
    if seed <= 0.0 {
        seed = 1e-16;
    }
    let mut sig2 = vec![0.0; x.len()];
    sig2[0] = seed;
    for t in 1..x.len() {
        sig2[t] = lam * sig2[t - 1] + (1.0 - lam) * x[t - 1] * x[t - 1];
    }
    Ok(sig2.iter().map(|v| v.max(1e-32).sqrt()).collect())
}

/// Filtered historical simulation (FHS) VaR — the industry workhorse.
///
/// Barone-Adesi et al. / Hull–White devolatilisation:
///
/// 1. one-step-ahead EWMA vol forecasts `sigma_t` for each day;
/// 2. standardise `z_t = pnl_t / sigma_t` (i.i.d.-ish innovations);
/// 3. rescale every innovation to tomorrow's forecast
///    `sigma_{T+1}^2 = lam * sigma_T^2 + (1 - lam) * pnl_T^2`;
/// 4. empirical `alpha` quantile (type-7) of the rescaled scenarios.
///
/// VaR responds to the current vol regime while keeping the empirical
/// (fat-tailed, skewed) shape of the standardised innovations.
pub fn filtered_historical_var(pnl: &[f64], alpha: f64, lam: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl(pnl, MIN_OBS)?;
    let sigma = ewma_volatility(pnl, lam)?;
    let n = pnl.len();
    let sigma_next =
        (lam * sigma[n - 1] * sigma[n - 1] + (1.0 - lam) * pnl[n - 1] * pnl[n - 1]).sqrt();
    let scenarios: Vec<f64> = pnl
        .iter()
        .zip(sigma.iter())
        .map(|(p, s)| p / s * sigma_next)
        .collect();
    Ok(-linear_quantile(&scenarios, alpha)?)
}

/// Square-root-of-time scaling: `VaR_h = VaR_1 * sqrt(h)`.
///
/// Valid only for i.i.d. returns with zero drift; understates multi-day
/// risk under volatility clustering / autocorrelation (docs/VALIDATION.md).
pub fn scale_var_sqrt_time(var_1d: f64, horizon_days: u32) -> Result<f64> {
    if horizon_days < 1 {
        return Err(EqVarError::InvalidInput(format!(
            "horizon_days must be >= 1, got {horizon_days}"
        )));
    }
    Ok(var_1d * (horizon_days as f64).sqrt())
}

/// Overlapping h-day P&L sums for direct multi-day historical VaR.
///
/// Returns the rolling sums `sum(pnl[t..t+h])` for `t = 0..=n-h`. Caveat:
/// overlapping sums are serially dependent, so the effective sample size is
/// ~`n/h` and quantile standard errors are much larger than the nominal
/// count suggests.
pub fn overlapping_horizon_pnl(pnl: &[f64], horizon_days: usize) -> Result<Vec<f64>> {
    if horizon_days < 1 {
        return Err(EqVarError::InvalidInput(format!(
            "horizon_days must be >= 1, got {horizon_days}"
        )));
    }
    if pnl.len() < horizon_days + MIN_OBS {
        return Err(EqVarError::InvalidInput(format!(
            "need at least {} observations for {horizon_days}-day overlapping windows, got {}",
            horizon_days + MIN_OBS,
            pnl.len()
        )));
    }
    let mut csum = vec![0.0; pnl.len() + 1];
    for (i, v) in pnl.iter().enumerate() {
        csum[i + 1] = csum[i] + v;
    }
    Ok((0..=pnl.len() - horizon_days)
        .map(|t| csum[t + horizon_days] - csum[t])
        .collect())
}
