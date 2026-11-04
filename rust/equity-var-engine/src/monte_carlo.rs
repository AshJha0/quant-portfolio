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
//! Determinism (W4(b), v3.0.0): path `p` of a run with seed `s` draws from
//! its own Philox4x32-10 stream `(s, p)` ([`crate::rng::PathRng`]): one
//! uniform per 64-bit word, a normal by Wichura's AS 241 inverse CDF, the
//! `n` factor normals first and then (Student-t) the Marsaglia-Tsang
//! chi-square. A path depends only on `(s, p)`, so [`monte_carlo_pnl`],
//! [`monte_carlo_var`] and [`monte_carlo_run`] run on all cores and give the
//! same bits on any thread count, and the C++ engine uses the same stream
//! definition (agreement to rounding, not by luck of seeds). Bootstrap
//! resample `b` likewise draws its indices from Philox stream `(s', b)`, so
//! [`var_bootstrap_se`] is threaded too. NumPy's stream differs, so Python
//! agreement is statistical.

use crate::matrix::Matrix;
use crate::psd::PsdRepair;
use crate::rng::PathRng;
use crate::{EqVarError, Result, TailModel, validate_alpha};

/// Minimum scenario count for [`var_kde_se`] / the tail read-off.
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
    let (chol, df, jitter) = scaled_factor(cov, tail)?;
    let mut out = Matrix::zeros(n_paths, n);
    let mut z = vec![0.0; n];
    for p in 0..n_paths {
        let mut rng = PathRng::new(seed, p as u64);
        rng.fill_normals(&mut z);
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
    Ok((out, jitter))
}

/// Factor `cov` (scaled by `(df - 2) / df` for Student-t) as the
/// simulation uses it: returns the lower factor, the Student-t df (if any)
/// and the Cholesky jitter.
pub(crate) fn scaled_factor(cov: &Matrix, tail: TailModel) -> Result<(Matrix, Option<f64>, f64)> {
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
    Ok((cf.factor, df, cf.jitter))
}

/// Scenario P&L of a linear book straight from the lower factor `chol`
/// (`n x n`, row-major), without forming the factor panel (W4(a)).
///
/// For a linear book `P&L_p = w . (m_p L z_p) = m_p (L^T w) . z_p`, so
/// `v = L^T w` is computed once (O(n^2)) and each path costs one O(n) dot
/// product instead of an O(n^2) triangular matvec. The draw order is that
/// of [`simulate_factor_returns`] (n normals, then the Student-t mixing
/// chi-square), so the result equals [`simulate_factor_returns`] followed
/// by [`portfolio_pnl`] to rounding only (summation order differs).
/// `v` and `z` are scratch of length `n`.
pub(crate) fn linear_pnl_from_factor(
    chol: &[f64],
    exposures: &[f64],
    df: Option<f64>,
    seed: u64,
    out: &mut [f64],
    v: &mut [f64],
    z: &mut [f64],
) {
    exposure_loadings(chol, exposures, v);
    for (p, o) in out.iter_mut().enumerate() {
        *o = linear_path(v, df, seed, p as u64, z);
    }
}

/// `v = L^T w` for the lower factor `chol` (`n x n`, row-major).
fn exposure_loadings(chol: &[f64], exposures: &[f64], v: &mut [f64]) {
    let n = exposures.len();
    v.iter_mut().for_each(|x| *x = 0.0);
    for (i, (row, w)) in chol.chunks_exact(n).zip(exposures).enumerate() {
        for (vj, l) in v.iter_mut().zip(&row[..=i]) {
            *vj += l * w;
        }
    }
}

/// P&L of path `p`: `m_p v . z_p` with `z_p` from [`PathRng::new`]`(seed, p)`.
#[inline]
fn linear_path(v: &[f64], df: Option<f64>, seed: u64, p: u64, z: &mut [f64]) -> f64 {
    let mut rng = PathRng::new(seed, p);
    rng.fill_normals(z);
    let mut acc = 0.0;
    for (vj, zj) in v.iter().zip(z.iter()) {
        acc += vj * zj;
    }
    if let Some(df) = df {
        // PathRng::chi_square never panics: a bad df gives NaN, which the
        // callers reject as a non-finite P&L.
        let w = rng.chi_square(df) / df;
        acc *= 1.0 / w.sqrt();
    }
    acc
}

/// Paths per thread below which the run stays on the caller's thread.
const MIN_PATHS_PER_THREAD: usize = 2_048;

/// [`linear_pnl_from_factor`] on `threads` threads. Every path has its own
/// stream, so the result is bit-identical for any `threads`.
fn linear_pnl_threaded(
    chol: &[f64],
    exposures: &[f64],
    df: Option<f64>,
    seed: u64,
    out: &mut [f64],
    threads: usize,
) {
    let n = exposures.len();
    let mut v = vec![0.0; n];
    exposure_loadings(chol, exposures, &mut v);
    let v = &v;
    qp_core::par::for_row_chunks(out, 1, MIN_PATHS_PER_THREAD, threads, |first, chunk| {
        let mut z = vec![0.0; n];
        for (k, o) in chunk.iter_mut().enumerate() {
            *o = linear_path(v, df, seed, (first + k) as u64, &mut z);
        }
    });
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
/// `simulate_factor_returns` followed by `portfolio_pnl`, to rounding.
/// Runs on [`qp_core::par::default_threads`] threads; the result does not
/// depend on the thread count (see [`monte_carlo_pnl_threads`]).
pub fn monte_carlo_pnl(
    exposures: &[f64],
    cov: &Matrix,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
) -> Result<Vec<f64>> {
    monte_carlo_pnl_threads(
        exposures,
        cov,
        n_paths,
        tail,
        seed,
        qp_core::par::default_threads(),
    )
}

/// [`monte_carlo_pnl`] on at most `threads` threads (`0` is taken as 1).
/// Path `p` draws from its own Philox stream ([`PathRng`]), so the output
/// is bit-identical for every `threads`.
///
/// # Errors
/// As [`monte_carlo_pnl`].
pub fn monte_carlo_pnl_threads(
    exposures: &[f64],
    cov: &Matrix,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
    threads: usize,
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
    Ok(linear_pnl_jitter(exposures, cov, n_paths, tail, seed, threads)?.0)
}

/// O(n) per path linear P&L plus the Cholesky jitter (shared by
/// [`monte_carlo_pnl`] and [`monte_carlo_run`]). Validates like
/// [`simulate_factor_returns`] and [`portfolio_pnl`] together.
fn linear_pnl_jitter(
    exposures: &[f64],
    cov: &Matrix,
    n_paths: usize,
    tail: TailModel,
    seed: u64,
    threads: usize,
) -> Result<(Vec<f64>, f64)> {
    if n_paths < 1 {
        return Err(EqVarError::InvalidInput(
            "simulate_factor_returns: n_paths must be >= 1".to_string(),
        ));
    }
    if exposures.iter().any(|w| !w.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "portfolio_pnl: exposures and returns must be finite".to_string(),
        ));
    }
    if !cov.all_finite() {
        return Err(EqVarError::InvalidInput(
            "simulate_factor_returns: covariance contains NaN or infinite entries".to_string(),
        ));
    }
    let (chol, df, jitter) = scaled_factor(cov, tail)?;
    let mut pnl = vec![0.0; n_paths];
    linear_pnl_threaded(chol.as_slice(), exposures, df, seed, &mut pnl, threads);
    if pnl.iter().any(|x| !x.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "portfolio_pnl: exposures and returns must be finite".to_string(),
        ));
    }
    Ok((pnl, jitter))
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
    let (pnl, jitter) = linear_pnl_jitter(
        exposures,
        cov,
        n_paths,
        tail,
        seed,
        qp_core::par::default_threads(),
    )?;
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
    let mut est = vec![0.0; n_boot];
    // Resample b has its own Philox stream (seed, b), so the resamples run
    // on all cores and the estimates do not depend on the thread count.
    let min_rows = (1usize << 16).div_ceil(n).max(1);
    qp_core::par::for_row_chunks(
        &mut est,
        1,
        min_rows,
        qp_core::par::default_threads(),
        |first, chunk| {
            let mut buf = vec![0.0; n];
            bootstrap_var_estimates(pnl, m, seed, first as u64, chunk, &mut buf);
        },
    );
    let sum: f64 = est.iter().sum();
    let mean = sum / n_boot as f64;
    let ss: f64 = est.iter().map(|v| (v - mean) * (v - mean)).sum();
    Ok((ss / (n_boot - 1) as f64).sqrt())
}

/// VaR (`m`-th largest loss) of bootstrap resamples `first_b ..
/// first_b + est.len()` into `est`: resample `b` draws `n` indices
/// `min(n - 1, floor(u n))` from Philox stream `(seed, b)` (W4(b)).
/// `buf` is scratch of length `pnl.len()`.
pub(crate) fn bootstrap_var_estimates(
    pnl: &[f64],
    m: usize,
    seed: u64,
    first_b: u64,
    est: &mut [f64],
    buf: &mut [f64],
) {
    if m > TOP_M_MAX {
        for (k, e) in est.iter_mut().enumerate() {
            *e = bootstrap_resample_var(pnl, m, seed, first_b + k as u64, buf);
        }
        return;
    }
    // Small rank (the usual 99 % VaR on a few hundred days): keep the m
    // largest losses as they are drawn, instead of filling a buffer and
    // selecting. Losses are compared by their `total_cmp` key, so the m-th
    // largest is the same value, bit for bit, as the selection in
    // `bootstrap_resample_var` returns (equal keys are equal bits). The keys
    // of the n losses are computed once into `buf` (as raw bits) and shared
    // by every resample.
    let n = pnl.len();
    for (slot, &p) in buf.iter_mut().zip(pnl.iter()) {
        *slot = f64::from_bits(total_key(-p) as u64);
    }
    let keys = &buf[..n];
    for (k, e) in est.iter_mut().enumerate() {
        *e = top_m_resample(keys, m, seed, first_b + k as u64);
    }
}

/// [`bootstrap_resample_var`] for `m <= TOP_M_MAX`, from the loss keys.
fn top_m_resample(keys: &[f64], m: usize, seed: u64, b: u64) -> f64 {
    let n = keys.len();
    let mut top = [i64::MIN; TOP_M_MAX];
    let top = &mut top[..m];
    let mut thr = i64::MIN;
    let mut block = 0u64;
    let mut done = 0usize;
    let mut idx = [0usize; 2 * crate::rng::LANES];
    while done < n {
        let len = (n - done).min(2 * crate::rng::LANES);
        // Philox words for LANES blocks side by side (the same words as
        // `PhiloxStream::new(seed, b)`; a test checks it), turned into
        // indices without a 64-bit float-to-integer cast.
        qp_core::path_rng::philox_indices(seed, b, block, n, &mut idx);
        block += crate::rng::LANES as u64;
        for &i in &idx[..len] {
            // `i <= n - 1`; `get` keeps the path panic-free.
            let k = keys.get(i).map_or(i64::MIN, |v| v.to_bits() as i64);
            if k > thr {
                let mut j = m - 1;
                while j > 0 && top[j - 1] < k {
                    top[j] = top[j - 1];
                    j -= 1;
                }
                top[j] = k;
                thr = top[m - 1];
            }
        }
        done += len;
    }
    from_total_key(thr)
}

/// VaR (`m`-th largest loss) of bootstrap resample `b`: `n` indices
/// `min(n - 1, floor(u n))` from Philox stream `(seed, b)` (W4(b)), then a
/// selection. `buf` is scratch of length `pnl.len()`.
pub(crate) fn bootstrap_resample_var(
    pnl: &[f64],
    m: usize,
    seed: u64,
    b: u64,
    buf: &mut [f64],
) -> f64 {
    let n = pnl.len();
    let nf = n as f64;
    // Uniforms come from the batched Philox generator (LANES blocks side by
    // side, the same words as `PhiloxStream::new(seed, b)`; a test checks
    // it). The one-word-at-a-time stream cost more than the rest of the
    // resample put together.
    let mut u = [0.0f64; 2 * crate::rng::LANES];
    let mut block = 0u64;
    for chunk in buf.chunks_mut(2 * crate::rng::LANES) {
        let uc = &mut u[..chunk.len()];
        crate::rng::philox_uniforms(seed, b, block, uc);
        block += crate::rng::LANES as u64;
        for (slot, &ui) in chunk.iter_mut().zip(uc.iter()) {
            let idx = ((ui * nf) as usize).min(n - 1);
            // `idx <= n - 1` by the `min`; `get` keeps the path panic-free.
            *slot = -pnl.get(idx).copied().unwrap_or(0.0);
        }
    }
    let (_, lm, _) = buf.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
    *lm
}

/// Largest rank `m` that [`bootstrap_var_estimates`] tracks with a running
/// top-m list instead of a full buffer and a selection.
const TOP_M_MAX: usize = 16;

use qp_core::resample::{from_total_key, total_key};

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-Round-8 resample: fill a buffer from the same Philox words,
    /// then select. The running top-m path must return the same bits.
    fn resample_by_selection(pnl: &[f64], m: usize, seed: u64, b: u64) -> f64 {
        let n = pnl.len();
        let mut rng = crate::rng::PathRng::new(seed, b);
        let mut buf: Vec<f64> = (0..n)
            .map(|_| -pnl[((rng.uniform() * n as f64) as usize).min(n - 1)])
            .collect();
        let (_, lm, _) = buf.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
        *lm
    }

    #[test]
    fn top_m_resample_matches_selection_bit_for_bit() {
        let mut cases = 0;
        for n in [10usize, 11, 31, 32, 33, 64, 250, 257, 1000] {
            // Heavy ties and both signed zeros, so equal keys are exercised.
            let pnl: Vec<f64> = (0..n)
                .map(|i| match i % 7 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 5.0,
                    _ => ((i * 7919) % 1013) as f64 * 0.37 - 180.0,
                })
                .collect();
            let mut buf = vec![0.0; n];
            for m in 1..=(TOP_M_MAX + 2).min(n) {
                let seed = 0x5EED ^ (m as u64) << 7 ^ n as u64;
                let mut est = vec![0.0; 12];
                bootstrap_var_estimates(&pnl, m, seed, 3, &mut est, &mut buf);
                for (k, got) in est.iter().enumerate() {
                    let b = 3 + k as u64;
                    let want = resample_by_selection(&pnl, m, seed, b);
                    assert_eq!(got.to_bits(), want.to_bits(), "n={n} m={m} b={b}");
                    let single = bootstrap_resample_var(&pnl, m, seed, b, &mut buf.clone());
                    assert_eq!(single.to_bits(), want.to_bits(), "n={n} m={m} b={b}");
                    cases += 1;
                }
            }
        }
        assert!(cases > 1000);
    }
}
