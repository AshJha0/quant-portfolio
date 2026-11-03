//! Monte Carlo VaR / ES for a linear (delta) portfolio.
//!
//! Simulate factor returns from a multivariate normal or multivariate
//! Student-t (common-mixing-variable construction `Z / sqrt(W/df)`, scale
//! matrix `cov * (df-2)/df` so the **covariance** matches `cov` exactly while
//! the tails fatten), revalue the linear portfolio `P&L = w . r`, and read
//! VaR / ES off the scenario distribution with the shared S2 estimator
//! ([`crate::tail`]). [`monte_carlo_run`] simulates once and returns VaR,
//! ES, the bootstrap standard error ([`var_bootstrap_se`], primary) and the
//! KDE standard error ([`var_kde_se`], diagnostic) from the same scenarios.
//!
//! For a linear book MC adds nothing over the closed form except a check of
//! it; option books go through
//! [`crate::option_reval::monte_carlo_option_run`], which reprices every path.
//!
//! Determinism: every draw comes from a seeded [`crate::rng::Rng`]
//! (xoshiro256++, Box–Muller normals, Marsaglia–Tsang gamma). There is no
//! platform-dependent RNG, so results repeat exactly for a given seed on
//! the same build and platform. Mirrors `eq_var.monte_carlo_var` semantically (same
//! factor model, same quantile convention); the RNG stream differs from
//! NumPy's and from the C++ engine's `std::mt19937_64` stream, so
//! cross-language / cross-engine agreement is statistical (within
//! order-statistic error bars), not bitwise.

use crate::matrix::Matrix;
use crate::psd::PsdRepair;
use crate::rng::Rng;
use crate::{EqVarError, Result, TailModel, validate_alpha};

/// Minimum scenario count for [`var_order_statistic_se`] / the tail read-off.
/// It mirrors the C++ engine's `mc_tail_metrics` floor.
pub const MIN_PATHS: usize = 100;

/// Minimum observations for [`var_bootstrap_se`] and [`var_kde_se`]. Lower
/// than [`MIN_PATHS`]: resampling a standard deviation is meaningful well
/// below the sample size needed for a raw tail read-off.
pub const MIN_BOOTSTRAP_OBS: usize = 10;

/// Simulate `n_paths` factor-return scenarios (an `(n_paths, n)` panel) with
/// target covariance `cov`.
///
/// `TailModel::StudentT { df }` requires `df > 2`; the Cholesky factor is
/// taken of `cov * (df-2)/df` (variance-matched) rather than `cov` so the
/// **simulated** covariance still equals `cov` while the joint tails fatten.
/// Uses [`Matrix::cholesky_repair`] without repair, so a PSD-but-singular
/// `cov` (e.g. a riskless leg) still simulates with a reported jitter; see
/// [`simulate_factor_returns_jitter`]. Repeats exactly for a fixed `seed`.
pub fn simulate_factor_returns(
    cov: &Matrix,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
) -> Result<Matrix> {
    Ok(simulate_factor_returns_jitter(cov, n_paths, tail, seed)?.0)
}

/// [`simulate_factor_returns`] plus the Cholesky jitter that was needed
/// (`0.0` when the covariance factorised as given).
///
/// # Errors
/// Same as [`simulate_factor_returns`].
pub fn simulate_factor_returns_jitter(
    cov: &Matrix,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
) -> Result<(Matrix, f64)> {
    if n_paths < 1 {
        return Err(EqVarError::InvalidInput(
            "simulate_factor_returns: n_paths must be >= 1".to_string(),
        ));
    }
    if cov.rows() != cov.cols() {
        return Err(EqVarError::InvalidInput(format!(
            "simulate_factor_returns: covariance must be square, got {} x {}",
            cov.rows(),
            cov.cols()
        )));
    }
    if !cov.all_finite() {
        return Err(EqVarError::InvalidInput(
            "simulate_factor_returns: covariance contains NaN or infinite entries".to_string(),
        ));
    }
    let n = cov.rows();
    let df = match tail {
        TailModel::Normal => None,
        TailModel::StudentT { df } => {
            if !(df > 2.0) || !df.is_finite() {
                return Err(EqVarError::InvalidInput(format!(
                    "simulate_factor_returns: Student-t df must be finite and > 2 for finite variance, got {df}"
                )));
            }
            Some(df)
        }
    };
    // Scale matrix cov * (df-2)/df so the simulated covariance equals cov.
    let scale = match df {
        None => cov.clone(),
        Some(df) => {
            let factor = (df - 2.0) / df;
            let mut m = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    m.set(i, j, cov.get(i, j) * factor);
                }
            }
            m
        }
    };
    let cf = scale.cholesky_repair(PsdRepair::None)?;
    let chol = cf.factor;
    let mut rng = Rng::new(seed);
    let mut out = Matrix::zeros(n_paths, n);
    let mut z = vec![0.0; n];
    for p in 0..n_paths {
        for zi in z.iter_mut() {
            *zi = rng.standard_normal();
        }
        // r = L z (lower-triangular product).
        for i in 0..n {
            let mut s = 0.0;
            for (j, zj) in z.iter().enumerate().take(i + 1) {
                s += chol.get(i, j) * zj;
            }
            out.set(p, i, s);
        }
        if let Some(df) = df {
            let w = rng.chi_square(df) / df;
            let m = 1.0 / w.sqrt();
            for i in 0..n {
                let v = out.get(p, i) * m;
                out.set(p, i, v);
            }
        }
    }
    Ok((out, cf.jitter))
}

/// Linear-portfolio scenario P&L: `pnl[t] = sum_j exposures[j] * returns(t, j)`.
///
/// Errors on an empty portfolio or a column-count mismatch with the panel.
pub fn portfolio_pnl(returns: &Matrix, exposures: &[f64]) -> Result<Vec<f64>> {
    if exposures.is_empty() {
        return Err(EqVarError::InvalidInput(
            "portfolio_pnl: book has no positions".to_string(),
        ));
    }
    if exposures.iter().any(|w| !w.is_finite()) || !returns.all_finite() {
        return Err(EqVarError::InvalidInput(
            "portfolio_pnl: exposures and returns must be finite".to_string(),
        ));
    }
    if returns.cols() != exposures.len() {
        return Err(EqVarError::InvalidInput(format!(
            "portfolio_pnl: panel has {} factors, portfolio has {} exposures",
            returns.cols(),
            exposures.len()
        )));
    }
    returns.matvec(exposures)
}

/// Simulate factor returns and revalue the linear portfolio in one step:
/// `simulate_factor_returns` followed by `portfolio_pnl`.
pub fn monte_carlo_pnl(
    exposures: &[f64],
    cov: &Matrix,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
) -> Result<Vec<f64>> {
    if exposures.is_empty() {
        return Err(EqVarError::InvalidInput(
            "monte_carlo_pnl: book has no positions".to_string(),
        ));
    }
    if cov.rows() != cov.cols() || cov.rows() != exposures.len() {
        return Err(EqVarError::InvalidInput(format!(
            "monte_carlo_pnl: covariance shape {} x {} does not match {} exposures",
            cov.rows(),
            cov.cols(),
            exposures.len()
        )));
    }
    let scen = simulate_factor_returns(cov, n_paths, tail, seed)?;
    portfolio_pnl(&scen, exposures)
}

fn sorted_copy(pnl: &[f64]) -> Vec<f64> {
    let mut v = pnl.to_vec();
    // `total_cmp` orders every f64 bit pattern, so this sort has no panic
    // path (the callers validate finiteness first regardless).
    v.sort_by(f64::total_cmp);
    v
}

fn validate_pnl_mc(pnl: &[f64]) -> Result<()> {
    if pnl.len() < MIN_PATHS {
        return Err(EqVarError::InvalidInput(format!(
            "need at least {MIN_PATHS} scenarios, got {}",
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

/// Monte Carlo VaR: the S2 order-statistic VaR of the scenario P&L
/// (`>= `[`MIN_PATHS`]` `scenarios). To get VaR, ES and their standard
/// errors from one simulation use [`monte_carlo_run`].
pub fn monte_carlo_var(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
) -> Result<f64> {
    validate_alpha(alpha)?;
    let pnl = monte_carlo_pnl(exposures, cov, n_paths, tail, seed)?;
    validate_pnl_mc(&pnl)?;
    crate::tail::empirical_var(&pnl, alpha)
}

/// Monte Carlo Expected Shortfall: the exact tail integral of the scenario
/// step CDF (same estimator as [`crate::expected_shortfall::expected_shortfall`]).
pub fn monte_carlo_es(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
) -> Result<f64> {
    validate_alpha(alpha)?;
    let pnl = monte_carlo_pnl(exposures, cov, n_paths, tail, seed)?;
    validate_pnl_mc(&pnl)?;
    crate::expected_shortfall::expected_shortfall(&pnl, alpha)
}

/// Old symmetric order-statistic finite-difference SE, kept for one
/// release. Use [`var_bootstrap_se`] (primary) or [`var_kde_se`].
///
/// Order-statistic standard error of the `alpha`-quantile VaR estimate on a
/// scenario P&L sample.
///
/// Asymptotic quantile variance `alpha(1-alpha) / (n f(q)^2)`, with the
/// density `f` estimated by a symmetric order-statistic finite difference of
/// bandwidth `ceil(sqrt(alpha * n))` around the quantile rank. Returns `0`
/// in the degenerate case where the local window has zero spread (e.g. a
/// riskless / zero-variance portfolio).
#[deprecated(since = "0.2.0", note = "use var_bootstrap_se (primary) or var_kde_se")]
pub fn var_order_statistic_se(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl_mc(pnl)?;
    let sorted = sorted_copy(pnl);
    let n = sorted.len();
    let h = alpha * (n - 1) as f64;
    let rank = (h.round() as usize).min(n - 1);
    let m = ((alpha * n as f64).sqrt().ceil() as usize).max(1);
    let ilo = rank.saturating_sub(m);
    let ihi = (rank + m).min(n - 1);
    let dx = sorted[ihi] - sorted[ilo];
    if dx > 0.0 {
        let f_hat = ((ihi - ilo) as f64 / n as f64) / dx;
        Ok((alpha * (1.0 - alpha) / n as f64).sqrt() / f_hat)
    } else {
        Ok(0.0)
    }
}

/// Everything one Monte Carlo simulation gives: see [`monte_carlo_run`].
#[derive(Debug, Clone, PartialEq)]
pub struct McRun {
    /// Scenario P&L (profit +).
    pub pnl: Vec<f64>,
    /// S2 order-statistic VaR (positive loss).
    pub var: f64,
    /// S2 Acerbi-Tasche ES (positive loss).
    pub es: f64,
    /// Bootstrap standard error of the VaR (primary).
    pub se: f64,
    /// KDE (Silverman) standard error of the VaR (diagnostic).
    pub se_kde: f64,
    /// Cholesky jitter that was needed (0 if none).
    pub jitter: f64,
}

/// Simulate once and derive VaR, ES and both standard errors from the
/// same scenarios.
///
/// The bootstrap uses `n_boot` resamples seeded from `seed` (offset so its
/// stream differs from the simulation stream).
///
/// # Errors
/// Invalid `alpha`, an empty book, fewer than [`MIN_PATHS`] paths,
/// `n_boot < 2`, an invalid tail model, or a covariance that is not
/// factorisable within the jitter cap.
pub fn monte_carlo_run(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
    n_boot: usize,
) -> Result<McRun> {
    validate_alpha(alpha)?;
    if n_boot < 2 {
        return Err(EqVarError::InvalidInput(format!(
            "n_boot must be >= 2 to estimate a standard deviation, got {n_boot}"
        )));
    }
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
    if cov.rows() != cov.cols() || cov.rows() != exposures.len() {
        return Err(EqVarError::InvalidInput(format!(
            "covariance shape {} x {} does not match {} exposures",
            cov.rows(),
            cov.cols(),
            exposures.len()
        )));
    }
    let (scen, jitter) = simulate_factor_returns_jitter(cov, n_paths, tail, seed)?;
    let pnl = portfolio_pnl(&scen, exposures)?;
    validate_pnl_mc(&pnl)?;
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

/// Bootstrap standard error of the S2 order-statistic VaR (primary SE).
///
/// Resamples `pnl` with replacement `n_boot` times from one seeded
/// [`Rng`], recomputes the S2 VaR `L(m)` on each resample (selection, not a
/// sort; `m` depends only on `n` and `alpha`), and returns the sample
/// standard deviation (`ddof = 1`) of the `n_boot` estimates. Memory is one
/// resample buffer of length `n`, never an `n_boot x n` index matrix.
///
/// # Errors
/// Invalid `alpha`, fewer than 10 or non-finite observations, or
/// `n_boot < 2` (a standard deviation needs two draws).
pub fn var_bootstrap_se(pnl: &[f64], alpha: f64, n_boot: usize, seed: u64) -> Result<f64> {
    validate_alpha(alpha)?;
    if pnl.len() < 10 {
        return Err(EqVarError::InvalidInput(format!(
            "need at least 10 observations to bootstrap, got {}",
            pnl.len()
        )));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "pnl contains NaN or infinite values".to_string(),
        ));
    }
    if n_boot < 2 {
        return Err(EqVarError::InvalidInput(format!(
            "n_boot must be >= 2 to estimate a standard deviation, got {n_boot}"
        )));
    }
    let n = pnl.len();
    let m = crate::tail::tail_rank(n, alpha)?;
    let mut rng = Rng::new(seed);
    let mut buf = vec![0.0; n];
    let mut sum = 0.0;
    let mut est = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        for slot in buf.iter_mut() {
            let idx = ((rng.uniform() * n as f64) as usize).min(n - 1);
            *slot = -pnl[idx];
        }
        let (_, lm, _) = buf.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
        let v = *lm;
        est.push(v);
        sum += v;
    }
    let mean = sum / n_boot as f64;
    let ss: f64 = est.iter().map(|v| (v - mean) * (v - mean)).sum();
    Ok((ss / (n_boot - 1) as f64).sqrt())
}

/// Kernel-density standard error of the VaR (diagnostic only).
///
/// `SE = sqrt(alpha (1 - alpha) / n) / f_hat(q)` with `q = -VaR` the
/// reported S2 quantile and `f_hat` a Gaussian KDE with Silverman's
/// bandwidth `h = 0.9 min(sd, IQR/1.34) n^(-1/5)`. A fixed bandwidth is
/// tuned to the bulk of the distribution, so deep in the tail it tends to
/// understate the SE; [`var_bootstrap_se`] is the primary estimate.
///
/// # Errors
/// Invalid `alpha`, fewer than 10 or non-finite observations.
pub fn var_kde_se(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    let n = pnl.len();
    if n < 10 {
        return Err(EqVarError::InvalidInput(format!(
            "need at least 10 scenarios for a VaR standard error, got {n}"
        )));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "pnl contains NaN or infinite values".to_string(),
        ));
    }
    let q = -crate::tail::empirical_var(pnl, alpha)?;
    let nf = n as f64;
    let mean = pnl.iter().sum::<f64>() / nf;
    let sd = (pnl.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0)).sqrt();
    let mut sorted = pnl.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let type7 = |p: f64| {
        let pos = p * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = pos - lo as f64;
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    };
    let iqr = type7(0.75) - type7(0.25);
    let spread = if iqr > 0.0 { sd.min(iqr / 1.34) } else { sd };
    if !(spread > 0.0) {
        // Degenerate (constant) sample: the quantile has no sampling error.
        return Ok(0.0);
    }
    let h = 0.9 * spread * nf.powf(-0.2);
    let inv_sqrt_2pi = 0.398_942_280_401_432_7;
    let dens = pnl
        .iter()
        .map(|x| {
            let u = (q - x) / h;
            inv_sqrt_2pi * (-0.5 * u * u).exp()
        })
        .sum::<f64>()
        / (nf * h);
    if !(dens > 0.0) {
        return Err(EqVarError::Numerical(
            "KDE density at the VaR quantile underflowed to zero".to_string(),
        ));
    }
    Ok((alpha * (1.0 - alpha) / nf).sqrt() / dens)
}
