//! Historical-simulation VaR: plain, age-weighted (BRW) and filtered (FHS).
//!
//! Conventions (identical to the Python reference `eq_var.historical_var`):
//!
//! * `pnl` slices are P&L in currency units, **loss < 0**;
//! * `alpha` is the tail probability (`0.01` → 99 % VaR);
//! * VaR is reported as a **positive** number: `VaR_alpha = -Q_alpha(pnl)`;
//! * plain historical VaR is the order statistic `L(m)`,
//!   `m = ceil(n alpha - 1e-9)`, of the losses sorted descending (see
//!   [`crate::tail`]); there is no interpolation;
//! * weighted quantiles (BRW) use the same rule on the weighted step CDF.

use crate::matrix::Matrix;
use crate::{validate_alpha, EqVarError, Result};

/// EWMA seed window: the variance seed is the mean of `r^2` over the first
/// `min(SEED_WINDOW, n)` observations.
pub const SEED_WINDOW: usize = 30;

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
    // `total_cmp` is a total order on all f64 bit patterns, so this sort
    // cannot panic even if a caller reaches it with a NaN (the public
    // entry points reject NaN first; this removes the panic path entirely
    // rather than relying on that).
    v.sort_by(f64::total_cmp);
    v
}

/// Empirical quantile with NumPy-default linear (Hyndman–Fan type-7)
/// interpolation, kept for one release. No VaR path uses it any more.
///
/// `h = (n - 1) q`, result
/// `x_(floor(h)) + (h - floor(h)) * (x_(floor(h)+1) - x_(floor(h)))` on the
/// ascending order statistics.
///
/// `q` must be in `[0, 1]`; input needs at least one finite observation.
#[deprecated(
    since = "0.2.0",
    note = "VaR is the order statistic now; use tail::empirical_var"
)]
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
fn linear_quantile_sorted(sorted: &[f64], q: f64) -> f64 {
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
/// Returns the loss order statistic `L(m)`, `m = ceil(n alpha - 1e-9)`
/// (positive for a loss). Requires at least [`MIN_OBS`] observations and
/// `alpha` in `(0, 0.5)`.
///
/// # Examples
///
/// ```
/// use eq_var_engine::historical::historical_var;
/// let pnl: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
/// // k = 100 * 0.05 = 5: the 5th largest loss.
/// assert_eq!(historical_var(&pnl, 0.05).unwrap(), 96.0);
/// ```
pub fn historical_var(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl(pnl, MIN_OBS)?;
    crate::tail::empirical_var(pnl, alpha)
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
    if n > i32::MAX as usize {
        return Err(EqVarError::InvalidInput(format!("n = {n} is too large")));
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
/// (`w_t ~ lam^age`). The weighted empirical CDF is inverted at `alpha`
/// with the shared rule of [`crate::tail::weighted_var_es`]: VaR is the loss
/// at which the cumulative weight (losses descending, stable sort) first
/// reaches `alpha - 1e-12`.
///
/// `lam -> 1` recovers plain historical simulation.
pub fn age_weighted_var(pnl: &[f64], alpha: f64, lam: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl(pnl, MIN_OBS)?;
    let w = brw_weights(pnl.len(), lam)?;
    Ok(crate::tail::weighted_var_es(pnl, &w, alpha)?.0)
}

/// Seeded RiskMetrics variance recursion for one series.
///
/// `s2[0] = mean(r[0..w]^2)`, `w = min(SEED_WINDOW, n)`;
/// `s2[t] = lam s2[t-1] + (1 - lam) r[t-1]^2`. Returns `(s2[0..n], s2[n])`,
/// the last entry being the one-step-ahead forecast. A zero seed is floored
/// at `1e-300` so standardising never divides by zero.
fn ewma_var_path(x: impl Iterator<Item = f64> + Clone, n: usize, lam: f64) -> (Vec<f64>, f64) {
    let w = SEED_WINDOW.min(n);
    let seed = x.clone().take(w).map(|v| v * v).sum::<f64>() / w as f64;
    let mut s2 = Vec::with_capacity(n);
    let mut cur = seed.max(1e-300);
    for v in x {
        s2.push(cur);
        cur = lam * cur + (1.0 - lam) * v * v;
    }
    (s2, cur)
}

/// One-step-ahead EWMA (RiskMetrics) volatility forecasts.
///
/// `sigma2[t] = lam * sigma2[t-1] + (1 - lam) * x[t-1]^2`, zero mean.
/// `sigma[t]` is the forecast for day `t` made with information up to
/// `t - 1`. The seed `sigma2[0]` is the mean of `x^2` over the first
/// `min(30, n)` observations ([`SEED_WINDOW`]). That window is the only
/// look-ahead: days inside it are standardised with a seed that saw them.
/// There is no full-sample seed. Returns one `sigma[t] > 0` per input
/// observation; [`ewma_volatility_forecast`] also returns `sigma[n]`.
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
    if x.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "ewma_volatility: input contains NaN or infinite values".to_string(),
        ));
    }
    Ok(ewma_volatility_forecast(x, lam)?.0)
}

/// [`ewma_volatility`] plus the one-step-ahead forecast `sigma[n]` for the
/// day after the sample.
///
/// # Errors
/// Same as [`ewma_volatility`].
pub fn ewma_volatility_forecast(x: &[f64], lam: f64) -> Result<(Vec<f64>, f64)> {
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
    if x.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "ewma_volatility: input contains NaN or infinite values".to_string(),
        ));
    }
    let (s2, next) = ewma_var_path(x.iter().copied(), x.len(), lam);
    Ok((s2.iter().map(|v| v.sqrt()).collect(), next.sqrt()))
}

/// Per-factor filtered historical scenarios (S9).
///
/// For each factor `j`: EWMA vol `sigma_{t,j}` as in [`ewma_volatility`]
/// (`lam`, seed window 30), standardised `z_{t,j} = r_{t,j} / sigma_{t,j}`,
/// rescaled `r*_{t,j} = z_{t,j} sigma_{T+1,j}` with the one-step-ahead vol.
/// Returns the `(T, n)` panel of rescaled factor returns.
///
/// # Errors
/// [`EqVarError::InvalidInput`] for a panel with fewer than 2 rows, no
/// columns, non-finite entries, or `lam` outside `(0, 1)`.
pub fn fhs_factor_scenarios(returns: &Matrix, lam: f64) -> Result<Matrix> {
    let (t, n) = (returns.rows(), returns.cols());
    if n == 0 {
        return Err(EqVarError::InvalidInput(
            "return panel has no columns".to_string(),
        ));
    }
    if !(lam > 0.0 && lam < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "decay lam must be in (0, 1), got {lam}"
        )));
    }
    if t < 2 {
        return Err(EqVarError::InvalidInput(
            "need at least 2 observations for EWMA volatility".to_string(),
        ));
    }
    if !returns.all_finite() {
        return Err(EqVarError::InvalidInput(
            "return panel contains NaN or infinite values".to_string(),
        ));
    }
    let mut out = Matrix::zeros(t, n);
    for j in 0..n {
        let col = (0..t).map(|i| returns.get(i, j));
        let (s2, next) = ewma_var_path(col, t, lam);
        let s_next = next.sqrt();
        for (i, v) in s2.iter().enumerate() {
            out.set(i, j, returns.get(i, j) / v.sqrt() * s_next);
        }
    }
    Ok(out)
}

/// Filtered historical simulation VaR for a linear book, per factor (S9).
///
/// Rescales every factor with [`fhs_factor_scenarios`], revalues the book
/// linearly (`pnl_t = exposures . r*_t`) and returns the S2 order-statistic
/// VaR. This is the linear-book port of the Python full-revaluation FHS.
///
/// # Errors
/// [`EqVarError::InvalidInput`] for an empty book, a shape mismatch, fewer
/// than [`MIN_OBS`] rows, non-finite input, or invalid `alpha` / `lam`.
pub fn filtered_historical_var_factors(
    returns: &Matrix,
    exposures: &[f64],
    alpha: f64,
    lam: f64,
) -> Result<f64> {
    validate_alpha(alpha)?;
    if exposures.is_empty() {
        return Err(EqVarError::InvalidInput(
            "book has no positions".to_string(),
        ));
    }
    if exposures.iter().any(|w| !w.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "exposures must all be finite".to_string(),
        ));
    }
    if returns.cols() != exposures.len() {
        return Err(EqVarError::InvalidInput(format!(
            "panel has {} factors, book has {} exposures",
            returns.cols(),
            exposures.len()
        )));
    }
    if returns.rows() < MIN_OBS {
        return Err(EqVarError::InvalidInput(format!(
            "need at least {MIN_OBS} observations for FHS, got {}",
            returns.rows()
        )));
    }
    let scen = fhs_factor_scenarios(returns, lam)?;
    let pnl = scen.matvec(exposures)?;
    crate::tail::empirical_var(&pnl, alpha)
}

/// Filtered historical simulation (FHS) VaR: the industry workhorse.
///
/// Barone-Adesi et al. / Hull–White devolatilisation:
///
/// 1. one-step-ahead EWMA vol forecasts `sigma_t` for each day;
/// 2. standardise `z_t = pnl_t / sigma_t` (i.i.d.-ish innovations);
/// 3. rescale every innovation to tomorrow's forecast
///    `sigma_{T+1}^2 = lam * sigma_T^2 + (1 - lam) * pnl_T^2`;
/// 4. type-7 `alpha` quantile of the rescaled scenarios (old arithmetic).
///
/// Deprecated: filtering the scalar P&L ignores that the book's factor mix
/// changes the P&L vol. Use [`filtered_historical_var_factors`].
///
/// Kept for one release with its old arithmetic unchanged, as in the Python
/// reference, so existing numbers do not move: full-sample population
/// variance seed and type-7 interpolated quantile.
#[deprecated(
    since = "0.2.0",
    note = "P&L-level FHS; use filtered_historical_var_factors (per-factor S9)"
)]
pub fn filtered_historical_var(pnl: &[f64], alpha: f64, lam: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl(pnl, MIN_OBS)?;
    if !(lam > 0.0 && lam < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "decay lam must be in (0, 1), got {lam}"
        )));
    }
    let mut seed = crate::stats::population_variance(pnl)?;
    if seed <= 0.0 {
        seed = 1e-16;
    }
    let n = pnl.len();
    let mut sig2 = vec![0.0; n];
    sig2[0] = seed;
    for t in 1..n {
        sig2[t] = lam * sig2[t - 1] + (1.0 - lam) * pnl[t - 1] * pnl[t - 1];
    }
    let sigma: Vec<f64> = sig2.iter().map(|v| v.max(1e-32).sqrt()).collect();
    let sigma_next =
        (lam * sigma[n - 1] * sigma[n - 1] + (1.0 - lam) * pnl[n - 1] * pnl[n - 1]).sqrt();
    let mut scenarios: Vec<f64> = pnl
        .iter()
        .zip(sigma.iter())
        .map(|(p, s)| p / s * sigma_next)
        .collect();
    scenarios.sort_by(f64::total_cmp);
    Ok(-linear_quantile_sorted(&scenarios, alpha))
}

/// Square-root-of-time scaling: `VaR_h = VaR_1 * sqrt(h)`.
///
/// Valid only for i.i.d. returns with zero drift; understates multi-day
/// risk under volatility clustering / autocorrelation (docs/VALIDATION.md).
pub fn scale_var_sqrt_time(var_1d: f64, horizon_days: u32) -> Result<f64> {
    if !var_1d.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "var_1d must be finite, got {var_1d}"
        )));
    }
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
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "pnl contains NaN or infinite values".to_string(),
        ));
    }
    if pnl.len() < horizon_days.saturating_add(MIN_OBS) {
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
