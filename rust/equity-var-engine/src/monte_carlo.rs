//! Monte Carlo VaR / ES for a linear (delta) portfolio.
//!
//! Simulate factor returns from a multivariate normal or multivariate
//! Student-t (common-mixing-variable construction `Z / sqrt(W/df)`, scale
//! matrix `cov * (df-2)/df` so the **covariance** matches `cov` exactly while
//! the tails fatten), revalue the linear portfolio `P&L = w . r`, and read
//! VaR / ES off the scenario distribution with an order-statistic standard
//! error ([`var_order_statistic_se`]) cross-checked by a distribution-free
//! bootstrap standard error ([`var_bootstrap_se`]).
//!
//! Determinism: every draw comes from a seeded [`crate::rng::Rng`]
//! (xoshiro256++, Box–Muller normals, Marsaglia–Tsang gamma) — no
//! platform-dependent RNG — so results are bit-reproducible for a given seed
//! on a given build. Mirrors `eq_var.monte_carlo_var` semantically (same
//! factor model, same quantile convention); the RNG stream differs from
//! NumPy's and from the C++ engine's `std::mt19937_64` stream, so
//! cross-language / cross-engine agreement is statistical (within
//! order-statistic error bars), not bitwise.

use crate::matrix::Matrix;
use crate::rng::Rng;
use crate::{validate_alpha, EqVarError, Result, TailModel};

/// Minimum scenario count for [`var_order_statistic_se`] / the tail read-off
/// — mirrors the C++ engine's `mc_tail_metrics` floor.
pub const MIN_PATHS: usize = 100;

/// Minimum observations for [`var_bootstrap_se`] — mirrors the Python
/// reference's `var_standard_error_bootstrap` floor. Deliberately lower than
/// [`MIN_PATHS`]: resampling a standard deviation is meaningful well below
/// the sample size needed for a raw tail read-off.
pub const MIN_BOOTSTRAP_OBS: usize = 10;

/// Simulate `n_paths` factor-return scenarios (an `(n_paths, n)` panel) with
/// target covariance `cov`.
///
/// `TailModel::StudentT { df }` requires `df > 2`; the Cholesky factor is
/// taken of `cov * (df-2)/df` (variance-matched) rather than `cov` so the
/// **simulated** covariance still equals `cov` while the joint tails fatten.
/// Uses [`Matrix::cholesky_jitter`] so a PSD-but-singular `cov` (e.g. a
/// riskless leg) still simulates. Bit-reproducible in `seed`.
pub fn simulate_factor_returns(
    cov: &Matrix,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
) -> Result<Matrix> {
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
            "simulate_factor_returns: covariance contains NaN or infinite entries"
                .to_string(),
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
    let chol = scale.cholesky_jitter(1e-10, 12)?;
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
    Ok(out)
}

/// Linear-portfolio scenario P&L: `pnl[t] = sum_j exposures[j] * returns(t, j)`.
///
/// Errors on an empty portfolio or a column-count mismatch with the panel.
pub fn portfolio_pnl(returns: &Matrix, exposures: &[f64]) -> Result<Vec<f64>> {
    if exposures.is_empty() {
        return Err(EqVarError::InvalidInput(
            "portfolio_pnl: empty portfolio (no exposures); nothing to revalue".to_string(),
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
            "monte_carlo_pnl: empty portfolio (no exposures)".to_string(),
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

/// Monte Carlo VaR: minus the type-7 linear-interpolated `alpha`-quantile of
/// a scenario P&L sample (`>= `[`MIN_PATHS`]` `scenarios).
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
    Ok(-crate::historical::linear_quantile(&pnl, alpha)?)
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

/// Order-statistic standard error of the `alpha`-quantile VaR estimate on a
/// scenario P&L sample.
///
/// Asymptotic quantile variance `alpha(1-alpha) / (n f(q)^2)`, with the
/// density `f` estimated by a symmetric order-statistic finite difference of
/// bandwidth `ceil(sqrt(alpha * n))` around the quantile rank. Returns `0`
/// in the degenerate case where the local window has zero spread (e.g. a
/// riskless / zero-variance portfolio).
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

fn validate_pnl_bootstrap(pnl: &[f64]) -> Result<()> {
    if pnl.len() < MIN_BOOTSTRAP_OBS {
        return Err(EqVarError::InvalidInput(format!(
            "need at least {MIN_BOOTSTRAP_OBS} observations to bootstrap, got {}",
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

/// Bootstrap standard error of the empirical `alpha`-quantile VaR estimate.
///
/// Distribution-free cross-check for [`var_order_statistic_se`]: resamples
/// `pnl` with replacement `n_boot` times using the engine's own
/// [`crate::rng::Rng`] (seeded once from `seed`), takes the same type-7
/// linear-interpolated quantile ([`crate::historical::linear_quantile`]) on
/// each resample to get a re-estimated VaR, and returns the sample standard
/// deviation (`ddof = 1`) across the `n_boot` re-estimates. Mirrors
/// `eq_var.monte_carlo_var.var_standard_error_bootstrap` bit for bit in
/// algorithm (same resampling scheme, same quantile convention) though not
/// in RNG stream.
///
/// [`var_order_statistic_se`]'s local finite-difference bandwidth
/// (`ceil(sqrt(alpha * n))`) is tuned to resolve the bulk of the
/// distribution; deep in the tail or at modest scenario counts that
/// bandwidth undersmooths and the estimate can understate the true sampling
/// SE by roughly 10-15 %. This estimator carries no such bandwidth
/// assumption — the price is `n_boot` extra quantile evaluations. Running
/// both and comparing is the desk-standard cross-check (see
/// docs/VALIDATION.md).
///
/// Requires `pnl.len() >= `[`MIN_BOOTSTRAP_OBS`]`, `alpha` in `(0, 0.5)`, and
/// `n_boot >= 2` (a standard deviation needs at least two draws).
pub fn var_bootstrap_se(pnl: &[f64], alpha: f64, n_boot: usize, seed: u64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_pnl_bootstrap(pnl)?;
    if n_boot < 2 {
        return Err(EqVarError::InvalidInput(format!(
            "n_boot must be >= 2 to estimate a standard deviation, got {n_boot}"
        )));
    }
    let n = pnl.len();
    let mut rng = Rng::new(seed);
    let mut resample = vec![0.0; n];
    let mut boot_vars = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        for slot in resample.iter_mut() {
            let idx = ((rng.uniform() * n as f64) as usize).min(n - 1);
            *slot = pnl[idx];
        }
        let sorted = sorted_copy(&resample);
        boot_vars.push(-crate::historical::linear_quantile_sorted(&sorted, alpha));
    }
    let mean = boot_vars.iter().sum::<f64>() / n_boot as f64;
    let ss: f64 = boot_vars.iter().map(|v| (v - mean) * (v - mean)).sum();
    Ok((ss / (n_boot - 1) as f64).sqrt())
}
