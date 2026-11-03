//! Allocation-free hot paths for latency-sensitive callers.
//!
//! Every function here returns the same `f64` bits as its counterpart in the
//! main API on the same input. The difference is the cost model:
//!
//! * no heap allocation on the success path (the caller passes a scratch
//!   slice or a reusable [`McWorkspace`]);
//! * errors are a small `Copy` enum, [`VarFault`], with no `String`;
//! * no panics on any input: short workspaces and bad shapes are errors.
//!
//! The tests `tests/test_fast_alloc.rs` (counting allocator) and
//! `tests/test_fast_equiv.rs` (bit equality with the main API) pin both
//! properties.
//!
//! The Monte Carlo workspace factors the covariance once in
//! [`McWorkspace::prepare`] (that step allocates) and then revalues the book
//! path by path. It never builds the `n_paths x n` scenario panel, so a run
//! touches `O(n_paths)` memory instead of `O(n_paths * n)`.
//!
//! The MC draw stays on one thread. The RNG is one serial xoshiro256++
//! stream and splitting it across threads would change the numbers.

use crate::TailModel;
use crate::matrix::Matrix;
use crate::psd::PsdRepair;
use crate::stats::{normal_pdf, normal_ppf, student_t_pdf, student_t_ppf};

/// Error of the allocation-free API. `Copy`, no payload, no allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarFault {
    /// `alpha` outside `(0, 0.5)`.
    BadAlpha,
    /// Fewer observations or paths than the estimator needs.
    TooFewObservations,
    /// A NaN or infinite input value.
    NonFinite,
    /// Slice or matrix shapes do not agree.
    ShapeMismatch,
    /// The caller's scratch buffer is shorter than the input.
    WorkspaceTooSmall,
    /// A model parameter is out of range (`df <= 2`, `n_boot < 2`, ...).
    BadParameter,
    /// The covariance is not positive semi-definite.
    NotPsd,
    /// The workspace has no Cholesky factor yet; call [`McWorkspace::prepare`].
    NotPrepared,
    /// A numerical step failed on well-formed input.
    Numerical,
}

impl std::fmt::Display for VarFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for VarFault {}

/// Result alias for the allocation-free API.
pub type FastResult<T> = std::result::Result<T, VarFault>;

#[inline]
fn check_alpha(alpha: f64) -> FastResult<()> {
    if alpha > 0.0 && alpha < 0.5 {
        Ok(())
    } else {
        Err(VarFault::BadAlpha)
    }
}

#[inline]
fn check_finite(x: &[f64]) -> FastResult<()> {
    // Sum of `is_finite` flags avoids an early exit so the loop vectorises.
    if x.iter().fold(true, |ok, v| ok & v.is_finite()) {
        Ok(())
    } else {
        Err(VarFault::NonFinite)
    }
}

/// S2 tail rank `m = ceil(n alpha - 1e-9)` clamped to `[1, n]`. Same
/// arithmetic as [`crate::tail::tail_rank`]; `n >= 1`, valid `alpha`.
#[inline]
fn rank(n: usize, alpha: f64) -> usize {
    let k = n as f64 * alpha;
    let m = (k - 1e-9).ceil();
    (m.max(1.0) as usize).min(n)
}

/// Write losses into `work[..n]` and return `(var, es)` with the same
/// arithmetic as [`crate::tail::empirical_var_es`]. `n >= 1`.
fn var_es_into(pnl: &[f64], alpha: f64, work: &mut [f64], want_es: bool) -> FastResult<(f64, f64)> {
    let n = pnl.len();
    let losses = work.get_mut(..n).ok_or(VarFault::WorkspaceTooSmall)?;
    for (l, p) in losses.iter_mut().zip(pnl) {
        *l = -p;
    }
    let m = rank(n, alpha);
    let (top, lm, _) = losses.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
    let var = *lm;
    if !want_es {
        return Ok((var, f64::NAN));
    }
    let k = n as f64 * alpha;
    top.sort_unstable_by(|a, b| b.total_cmp(a));
    let head: f64 = top.iter().sum();
    let es = (head + (k - (m - 1) as f64) * var) / k;
    Ok((var, es))
}

/// Allocation-free [`crate::historical::historical_var`]. `work` must hold
/// at least `pnl.len()` values; its contents are overwritten.
///
/// # Errors
/// [`VarFault::BadAlpha`], [`VarFault::TooFewObservations`] (fewer than
/// [`crate::historical::MIN_OBS`]), [`VarFault::NonFinite`],
/// [`VarFault::WorkspaceTooSmall`].
///
/// # Examples
/// ```
/// use eq_var_engine::fast::historical_var_ws;
/// let pnl: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
/// let mut work = vec![0.0; pnl.len()];
/// assert_eq!(historical_var_ws(&pnl, 0.05, &mut work), Ok(96.0));
/// ```
pub fn historical_var_ws(pnl: &[f64], alpha: f64, work: &mut [f64]) -> FastResult<f64> {
    check_alpha(alpha)?;
    if pnl.len() < crate::historical::MIN_OBS {
        return Err(VarFault::TooFewObservations);
    }
    check_finite(pnl)?;
    Ok(var_es_into(pnl, alpha, work, false)?.0)
}

/// Allocation-free [`crate::tail::empirical_var_es`] (`n >= 1`). Returns
/// `(VaR, ES)`, both positive for a loss.
///
/// # Errors
/// As [`historical_var_ws`], with a floor of one observation.
pub fn empirical_var_es_ws(pnl: &[f64], alpha: f64, work: &mut [f64]) -> FastResult<(f64, f64)> {
    check_alpha(alpha)?;
    if pnl.is_empty() {
        return Err(VarFault::TooFewObservations);
    }
    check_finite(pnl)?;
    var_es_into(pnl, alpha, work, true)
}

/// Allocation-free [`crate::expected_shortfall::expected_shortfall`]
/// (at least [`crate::expected_shortfall::MIN_OBS`] observations).
///
/// # Errors
/// As [`historical_var_ws`].
pub fn expected_shortfall_ws(pnl: &[f64], alpha: f64, work: &mut [f64]) -> FastResult<f64> {
    check_alpha(alpha)?;
    if pnl.len() < crate::expected_shortfall::MIN_OBS {
        return Err(VarFault::TooFewObservations);
    }
    check_finite(pnl)?;
    Ok(var_es_into(pnl, alpha, work, true)?.1)
}

/// Allocation-free [`crate::parametric::portfolio_sigma`]: `sqrt(w' S w)`.
///
/// Fuses the mat-vec and the dot product. The summation order matches the
/// main API term by term, so the result has the same bits.
///
/// # Errors
/// [`VarFault::ShapeMismatch`] (empty book or wrong shape),
/// [`VarFault::NonFinite`], [`VarFault::NotPsd`].
pub fn portfolio_sigma_fast(exposures: &[f64], cov: &Matrix) -> FastResult<f64> {
    let n = exposures.len();
    if n == 0 || cov.rows() != n || cov.cols() != n {
        return Err(VarFault::ShapeMismatch);
    }
    check_finite(exposures)?;
    let data = cov.as_slice();
    check_finite(data)?;
    // `Iterator::sum` for f64 starts from -0.0; keep that seed.
    let mut var = -0.0f64;
    // Four rows at a time: each row keeps its own serial sum (same order,
    // same bits as `Matrix::matvec`), but the four chains run side by side,
    // so the add latency overlaps.
    let quad = n * 4;
    let mut blocks = data.chunks_exact(quad);
    let (wq, wrem) = exposures.as_chunks::<4>();
    for (blk, ws) in (&mut blocks).zip(wq) {
        let (r0, rest) = blk.split_at(n);
        let (r1, rest) = rest.split_at(n);
        let (r2, r3) = rest.split_at(n);
        let (mut a0, mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0, 0.0);
        for ((((x, a), b), c), d) in exposures.iter().zip(r0).zip(r1).zip(r2).zip(r3) {
            a0 += a * x;
            a1 += b * x;
            a2 += c * x;
            a3 += d * x;
        }
        let [w0, w1, w2, w3] = *ws;
        var += w0 * a0;
        var += w1 * a1;
        var += w2 * a2;
        var += w3 * a3;
    }
    for (row, w) in blocks.remainder().chunks_exact(n).zip(wrem) {
        let mut acc = 0.0;
        for (a, b) in row.iter().zip(exposures) {
            acc += a * b;
        }
        var += w * acc;
    }
    let scale = exposures.iter().fold(1.0f64, |m, w| m.max(w.abs()));
    if var < -1e-10 * scale * scale {
        return Err(VarFault::NotPsd);
    }
    Ok(var.max(0.0).sqrt())
}

#[inline]
fn tail_z(alpha: f64, tail: TailModel) -> FastResult<f64> {
    match tail {
        TailModel::Normal => normal_ppf(alpha).map_err(|_| VarFault::Numerical),
        TailModel::StudentT { df } => {
            if !(df > 2.0) || !df.is_finite() {
                return Err(VarFault::BadParameter);
            }
            Ok(student_t_ppf(alpha, df).map_err(|_| VarFault::Numerical)?
                * ((df - 2.0) / df).sqrt())
        }
    }
}

/// Allocation-free [`crate::parametric::parametric_var`] (zero mean,
/// one day).
///
/// # Errors
/// [`VarFault::BadAlpha`], [`VarFault::BadParameter`] (`df <= 2`), and the
/// errors of [`portfolio_sigma_fast`].
pub fn parametric_var_fast(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    tail: TailModel,
) -> FastResult<f64> {
    check_alpha(alpha)?;
    let sigma = portfolio_sigma_fast(exposures, cov)? * 1.0f64.sqrt();
    let mu = 0.0 * 1.0;
    let z = tail_z(alpha, tail)?;
    Ok(-(mu + z * sigma))
}

/// Allocation-free [`crate::expected_shortfall::parametric_es`] (zero mean,
/// one day).
///
/// # Errors
/// As [`parametric_var_fast`].
pub fn parametric_es_fast(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    tail: TailModel,
) -> FastResult<f64> {
    check_alpha(alpha)?;
    let sigma = portfolio_sigma_fast(exposures, cov)? * 1.0f64.sqrt();
    let mean = 0.0 * 1.0;
    match tail {
        TailModel::Normal => {
            let z = normal_ppf(alpha).map_err(|_| VarFault::Numerical)?;
            Ok(sigma * normal_pdf(z) / alpha - mean)
        }
        TailModel::StudentT { df } => {
            if !(df > 2.0) || !df.is_finite() {
                return Err(VarFault::BadParameter);
            }
            let q = student_t_ppf(alpha, df).map_err(|_| VarFault::Numerical)?;
            let pdf = student_t_pdf(q, df).map_err(|_| VarFault::Numerical)?;
            let es_std = pdf * (df + q * q) / ((df - 1.0) * alpha);
            Ok(sigma * es_std * ((df - 2.0) / df).sqrt() - mean)
        }
    }
}

/// Bootstrap SE core shared by [`var_bootstrap_se_ws`] and the MC
/// workspace: the same per-resample Philox streams and sums as
/// [`crate::monte_carlo::var_bootstrap_se`].
fn bootstrap_core(
    pnl: &[f64],
    alpha: f64,
    n_boot: usize,
    seed: u64,
    buf: &mut [f64],
    est: &mut [f64],
) -> FastResult<f64> {
    let n = pnl.len();
    let buf = buf.get_mut(..n).ok_or(VarFault::WorkspaceTooSmall)?;
    let est = est.get_mut(..n_boot).ok_or(VarFault::WorkspaceTooSmall)?;
    let m = rank(n, alpha);
    for (b, e) in est.iter_mut().enumerate() {
        *e = crate::monte_carlo::bootstrap_resample_var(pnl, m, seed, b as u64, buf);
    }
    let sum: f64 = est.iter().sum();
    let mean = sum / n_boot as f64;
    let ss: f64 = est.iter().map(|v| (v - mean) * (v - mean)).sum();
    Ok((ss / (n_boot - 1) as f64).sqrt())
}

/// Allocation-free [`crate::monte_carlo::var_bootstrap_se`]. `buf` holds at
/// least `pnl.len()` values and `est` at least `n_boot`.
///
/// # Errors
/// [`VarFault::BadAlpha`], [`VarFault::TooFewObservations`] (fewer than
/// 10), [`VarFault::NonFinite`], [`VarFault::BadParameter`] (`n_boot < 2`),
/// [`VarFault::WorkspaceTooSmall`].
pub fn var_bootstrap_se_ws(
    pnl: &[f64],
    alpha: f64,
    n_boot: usize,
    seed: u64,
    buf: &mut [f64],
    est: &mut [f64],
) -> FastResult<f64> {
    check_alpha(alpha)?;
    if pnl.len() < 10 {
        return Err(VarFault::TooFewObservations);
    }
    check_finite(pnl)?;
    if n_boot < 2 {
        return Err(VarFault::BadParameter);
    }
    bootstrap_core(pnl, alpha, n_boot, seed, buf, est)
}

/// KDE SE core: same arithmetic as [`crate::monte_carlo::var_kde_se`];
/// `q` is minus the S2 VaR, `sorted` receives an ascending copy.
fn kde_core(pnl: &[f64], alpha: f64, q: f64, sorted: &mut [f64]) -> FastResult<f64> {
    let n = pnl.len();
    let sorted = sorted.get_mut(..n).ok_or(VarFault::WorkspaceTooSmall)?;
    let nf = n as f64;
    let mean = pnl.iter().sum::<f64>() / nf;
    let sd = (pnl.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0)).sqrt();
    sorted.copy_from_slice(pnl);
    sorted.sort_unstable_by(f64::total_cmp);
    let sorted = &*sorted;
    let type7 = |p: f64| {
        let pos = p * (n - 1) as f64;
        let lo = (pos.floor() as usize).min(n - 1);
        let hi = (lo + 1).min(n - 1);
        let frac = pos - lo as f64;
        let (a, b) = (
            sorted.get(lo).copied().unwrap_or(0.0),
            sorted.get(hi).copied().unwrap_or(0.0),
        );
        a + frac * (b - a)
    };
    let iqr = type7(0.75) - type7(0.25);
    let spread = if iqr > 0.0 { sd.min(iqr / 1.34) } else { sd };
    if !(spread > 0.0) {
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
        return Err(VarFault::Numerical);
    }
    Ok((alpha * (1.0 - alpha) / nf).sqrt() / dens)
}

/// Statistics of one Monte Carlo run, without the scenario vector (read it
/// with [`McWorkspace::pnl`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McStats {
    /// S2 order-statistic VaR (positive loss).
    pub var: f64,
    /// S2 Acerbi-Tasche ES (positive loss).
    pub es: f64,
    /// Bootstrap standard error of the VaR.
    pub se: f64,
    /// KDE standard error of the VaR.
    pub se_kde: f64,
    /// Cholesky jitter added in [`McWorkspace::prepare`].
    pub jitter: f64,
}

/// Reusable Monte Carlo workspace.
///
/// [`McWorkspace::new`] sizes the buffers and [`McWorkspace::prepare`]
/// factors a covariance. Both allocate. After that, [`McWorkspace::run`]
/// and [`McWorkspace::var`] allocate nothing as long as `n_paths` and
/// `n_boot` stay within the sizes given to `new`.
///
/// `run` returns the same bits as [`crate::monte_carlo::monte_carlo_run`]
/// for the same covariance, tail model, seed and `n_boot`.
///
/// # Examples
/// ```
/// use eq_var_engine::prelude::*;
/// use eq_var_engine::fast::McWorkspace;
/// let corr = Matrix::from_vec(2, 2, vec![1.0, 0.3, 0.3, 1.0]).unwrap();
/// let cov = covariance_from_vols(&[0.01, 0.02], &corr).unwrap();
/// let w = [1.0e6, -5.0e5];
/// let mut ws = McWorkspace::new(2, 1_000, 50);
/// ws.prepare(&cov, TailModel::Normal).unwrap();
/// let s = ws.run(&w, 0.01, 1_000, 7, 50).unwrap();
/// let old = monte_carlo_run(&w, &cov, 0.01, 1_000, TailModel::Normal, 7, 50).unwrap();
/// assert_eq!(s.var.to_bits(), old.var.to_bits());
/// ```
#[derive(Debug, Clone)]
pub struct McWorkspace {
    n: usize,
    chol: Vec<f64>,
    df: Option<f64>,
    jitter: f64,
    prepared: bool,
    z: Vec<f64>,
    v: Vec<f64>,
    pnl: Vec<f64>,
    buf: Vec<f64>,
    est: Vec<f64>,
}

impl McWorkspace {
    /// Buffers for `n` factors, up to `max_paths` paths and `max_boot`
    /// bootstrap resamples.
    pub fn new(n: usize, max_paths: usize, max_boot: usize) -> Self {
        McWorkspace {
            n,
            chol: vec![0.0; n * n],
            df: None,
            jitter: 0.0,
            prepared: false,
            z: vec![0.0; n],
            v: vec![0.0; n],
            pnl: vec![0.0; max_paths],
            buf: vec![0.0; max_paths],
            est: vec![0.0; max_boot],
        }
    }

    /// Factor `cov` (scaled by `(df - 2) / df` for Student-t) exactly as
    /// [`crate::monte_carlo::simulate_factor_returns_jitter`] does. This
    /// step allocates; call it when the covariance changes.
    ///
    /// # Errors
    /// [`VarFault::ShapeMismatch`], [`VarFault::NonFinite`],
    /// [`VarFault::BadParameter`] (`df <= 2`), [`VarFault::NotPsd`] when the
    /// factorisation fails within the jitter cap.
    pub fn prepare(&mut self, cov: &Matrix, tail: TailModel) -> FastResult<()> {
        self.prepared = false;
        let n = self.n;
        if cov.rows() != n || cov.cols() != n {
            return Err(VarFault::ShapeMismatch);
        }
        if !cov.all_finite() {
            return Err(VarFault::NonFinite);
        }
        let df = match tail {
            TailModel::Normal => None,
            TailModel::StudentT { df } => {
                if !(df > 2.0) || !df.is_finite() {
                    return Err(VarFault::BadParameter);
                }
                Some(df)
            }
        };
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
        let cf = scale
            .cholesky_repair(PsdRepair::None)
            .map_err(|_| VarFault::NotPsd)?;
        self.chol.copy_from_slice(cf.factor.as_slice());
        self.df = df;
        self.jitter = cf.jitter;
        self.prepared = true;
        Ok(())
    }

    /// Scenario P&L of the last [`McWorkspace::run`] or
    /// [`McWorkspace::var`] call (`n_paths` values).
    pub fn pnl(&self, n_paths: usize) -> &[f64] {
        &self.pnl[..n_paths.min(self.pnl.len())]
    }

    fn simulate(&mut self, exposures: &[f64], n_paths: usize, seed: u64) -> FastResult<()> {
        if !self.prepared {
            return Err(VarFault::NotPrepared);
        }
        let n = self.n;
        if exposures.len() != n || n == 0 {
            return Err(VarFault::ShapeMismatch);
        }
        check_finite(exposures)?;
        if n_paths < crate::monte_carlo::MIN_PATHS {
            return Err(VarFault::TooFewObservations);
        }
        let pnl = self
            .pnl
            .get_mut(..n_paths)
            .ok_or(VarFault::WorkspaceTooSmall)?;
        // O(n) per path via v = L^T w (W4(a)); same code as the main API.
        crate::monte_carlo::linear_pnl_from_factor(
            &self.chol,
            exposures,
            self.df,
            seed,
            pnl,
            &mut self.v,
            &mut self.z,
        );
        check_finite(pnl)
    }

    /// Simulate once and return VaR, ES and both standard errors with the
    /// same bits as [`crate::monte_carlo::monte_carlo_run`].
    ///
    /// # Errors
    /// [`VarFault::BadAlpha`], [`VarFault::BadParameter`] (`n_boot < 2`),
    /// [`VarFault::NotPrepared`], [`VarFault::ShapeMismatch`],
    /// [`VarFault::TooFewObservations`] (fewer than
    /// [`crate::monte_carlo::MIN_PATHS`] paths),
    /// [`VarFault::WorkspaceTooSmall`], [`VarFault::NonFinite`],
    /// [`VarFault::Numerical`].
    pub fn run(
        &mut self,
        exposures: &[f64],
        alpha: f64,
        n_paths: usize,
        seed: u64,
        n_boot: usize,
    ) -> FastResult<McStats> {
        check_alpha(alpha)?;
        if n_boot < 2 {
            return Err(VarFault::BadParameter);
        }
        if self.est.len() < n_boot {
            return Err(VarFault::WorkspaceTooSmall);
        }
        self.simulate(exposures, n_paths, seed)?;
        let pnl = &self.pnl[..n_paths];
        let (var, es) = var_es_into(pnl, alpha, &mut self.buf, true)?;
        let se = bootstrap_core(
            pnl,
            alpha,
            n_boot,
            seed ^ 0x9E37_79B9_7F4A_7C15,
            &mut self.buf,
            &mut self.est,
        )?;
        let se_kde = kde_core(pnl, alpha, -var, &mut self.buf)?;
        Ok(McStats {
            var,
            es,
            se,
            se_kde,
            jitter: self.jitter,
        })
    }

    /// Simulate once and return the S2 VaR only, with the same bits as
    /// [`crate::monte_carlo::monte_carlo_var`].
    ///
    /// # Errors
    /// As [`McWorkspace::run`] without the `n_boot` check.
    pub fn var(
        &mut self,
        exposures: &[f64],
        alpha: f64,
        n_paths: usize,
        seed: u64,
    ) -> FastResult<f64> {
        check_alpha(alpha)?;
        self.simulate(exposures, n_paths, seed)?;
        Ok(var_es_into(&self.pnl[..n_paths], alpha, &mut self.buf, false)?.0)
    }
}
