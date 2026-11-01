//! Monte Carlo VaR: multivariate normal / Student-t / jump-mixture factors.
//!
//! Factor returns are simulated from a daily covariance (Cholesky with
//! jitter escalation — two perfectly correlated pegged currencies are a
//! legitimate FX input) and the book is fully revalued scenario by
//! scenario through [`crate::book::CompiledBook`] (mirrors Python
//! `fx_var.monte_carlo_var` / C++ `fxvar::monte_carlo_var`).
//!
//! # Distributions
//!
//! * [`crate::McDist::Normal`] — `MVN(0, Sigma h)`.
//! * [`crate::McDist::StudentT`] — multivariate Student-t scaled to match
//!   `Sigma` exactly (`X = Z sqrt((df-2)/df) / sqrt(W/df)`), so the
//!   comparison with normal MC is at equal covariance: any 99% VaR
//!   difference is pure tail shape. EM currency returns are the textbook
//!   case (df 4-6).
//! * [`crate::McDist::Jump`] — normal diffusion plus a common Poisson jump
//!   count: the per-day probability `p` gives the intensity
//!   `lam = -ln(1 - p)`, the count over the horizon is
//!   `N ~ Poisson(lam h)`, and each listed factor moves by the sum of `N`
//!   iid jump sizes `~ N(mu_J, sigma_J)`, i.e. `N mu_J + sigma_J sqrt(N) Z`.
//!   The old overlay drew one Bernoulli(p) jump whatever the horizon, so a
//!   10-day VaR saw the 1-day jump risk. The jump is uncompensated: it adds
//!   drift `lam h mu_J` as well as variance, on top of `Sigma`, by design.
//!
//! # Determinism
//!
//! All randomness flows through [`crate::rng::Rng`] (xoshiro256++ driving
//! inverse-CDF transforms implemented in this crate). A fixed seed repeats
//! the scenario set and VaR exactly on the same build and platform; no
//! claim is made across platforms (libm `ln`/`exp` may differ by an ulp).
//!
//! # Standard error
//!
//! One simulation gives VaR, ES and both standard errors.
//! [`var_standard_error_bootstrap`] is the primary SE: resample the scenario
//! P&L, recompute the S2 order-statistic VaR, take the standard deviation.
//! [`var_standard_error`] is a diagnostic: `sqrt(a(1-a)/n) / f(q)` with a
//! Silverman-bandwidth Gaussian KDE evaluated at the reported quantile.
//! A fixed bandwidth is tuned to the bulk of the distribution, so deep in
//! the tail the KDE figure tends to understate the SE.

use std::collections::HashMap;

use crate::book::{Book, CompiledBook, Market};
use crate::expected_shortfall::empirical_var_es;
use crate::matrix::Matrix;
use crate::returns::{FactorCov, ReturnsMatrix};
use crate::rng::Rng;
use crate::stats::{validate_alpha, validate_horizon};
use crate::{FxVarError, McDist, Result};

/// Common-jump overlay for [`McDist::Jump`]: `prob` is the per-day jump
/// probability, turned into a Poisson count over the horizon (see the
/// module docs). Each listed factor moves by `mean[f] + std[f] * Z` per
/// jump (log-return units for FX factors; factors not listed do not jump).
/// E.g. `{"FX:TRY": -0.15}` is a 15% (log) devaluation of the lira vs USD.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JumpSpec {
    /// Per-day jump probability, in `[0, 1)`.
    pub prob: f64,
    /// Jump mean per factor.
    pub mean: HashMap<String, f64>,
    /// Jump standard deviation per factor (`>= 0`).
    pub stdev: HashMap<String, f64>,
}

/// Options controlling [`monte_carlo_var`].
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarloOptions {
    /// Tail probability, e.g. `0.01` for 99 % VaR (was a confidence level).
    pub alpha: f64,
    /// Horizon in trading days.
    pub horizon_days: f64,
    /// Number of scenarios to simulate.
    pub n_scenarios: usize,
    /// Simulation distribution.
    pub dist: McDist,
    /// Student-t degrees of freedom (used when `dist == StudentT`; must be
    /// `> 2`).
    pub df: f64,
    /// Jump overlay (used when `dist == Jump`).
    pub jumps: JumpSpec,
    /// RNG seed.
    pub seed: u64,
    /// Bootstrap resamples for the primary VaR standard error (`>= 2`).
    pub n_boot: usize,
}

impl Default for MonteCarloOptions {
    fn default() -> Self {
        MonteCarloOptions {
            alpha: 0.01,
            horizon_days: 1.0,
            n_scenarios: 50_000,
            dist: McDist::Normal,
            df: 6.0,
            jumps: JumpSpec::default(),
            seed: 0,
            n_boot: 200,
        }
    }
}

/// Monte Carlo VaR result (positive base-ccy losses).
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarloResult {
    /// Positive VaR loss.
    pub var: f64,
    /// Positive ES loss.
    pub es: f64,
    /// Tail probability used.
    pub alpha: f64,
    /// Horizon in trading days used.
    pub horizon_days: f64,
    /// Simulation distribution used.
    pub dist: McDist,
    /// Number of scenarios simulated.
    pub n_scenarios: usize,
    /// Bootstrap standard error of the VaR estimate (primary).
    pub se_var: f64,
    /// KDE (Silverman) standard error at the reported quantile (diagnostic).
    pub se_var_kde: f64,
    /// Cholesky diagonal jitter that was needed (0 if none).
    pub jitter: f64,
    /// Simulated scenario P&L.
    pub pnl: Vec<f64>,
    /// Non-empty when the covariance needed diagonal jitter to factorise.
    pub cholesky_warning: String,
}

/// Simulate factor-return scenarios from a daily covariance. `cov.cov` is
/// scaled by `horizon_days` (i.i.d. aggregation).
///
/// # Errors
/// [`FxVarError::Invalid`] for `n_scenarios < 1`, a Student-t `df` that is
/// not finite and `> 2`, a jump probability outside `[0, 1)`, a non-finite
/// jump mean or a negative / non-finite jump stdev, a Poisson mean
/// `lam h > 700`, or a shape mismatch between `cov.factors` and `cov.cov`;
/// [`FxVarError::Numerical`] if the (possibly jittered) covariance still
/// cannot be factorised.
///
/// Returns `(scenarios, jitter_warning)`: `jitter_warning` is non-empty
/// when diagonal jitter was needed.
pub fn simulate_factor_returns(
    cov: &FactorCov,
    n_scenarios: usize,
    dist: McDist,
    df: f64,
    jumps: &JumpSpec,
    seed: u64,
    horizon_days: f64,
) -> Result<(ReturnsMatrix, String)> {
    let (r, w, _) = simulate_factor_returns_jitter(cov, n_scenarios, dist, df, jumps, seed, horizon_days)?;
    Ok((r, w))
}

/// [`simulate_factor_returns`] plus the Cholesky jitter actually added
/// (`0.0` if the horizon covariance factorised as given).
///
/// # Errors
/// Same as [`simulate_factor_returns`].
pub fn simulate_factor_returns_jitter(
    cov: &FactorCov,
    n_scenarios: usize,
    dist: McDist,
    df: f64,
    jumps: &JumpSpec,
    seed: u64,
    horizon_days: f64,
) -> Result<(ReturnsMatrix, String, f64)> {
    validate_horizon(horizon_days)?;
    if n_scenarios < 1 {
        return Err(FxVarError::invalid("n_scenarios must be >= 1"));
    }
    let k = cov.factors.len();
    if cov.cov.rows() != k || cov.cov.cols() != k {
        return Err(FxVarError::invalid(
            "simulate_factor_returns: cov labels do not match matrix shape",
        ));
    }
    // `!(df > 2.0)`, not `df <= 2.0`: the latter is false for NaN, which
    // used to reach the gamma sampler.
    if dist == McDist::StudentT && (!(df > 2.0) || !df.is_finite()) {
        return Err(FxVarError::invalid(format!(
            "Student-t df must be finite and > 2 for finite variance, got {df}"
        )));
    }
    let mut jump_lam_h = 0.0;
    if dist == McDist::Jump {
        if !(jumps.prob >= 0.0 && jumps.prob < 1.0) {
            return Err(FxVarError::invalid(format!(
                "jump prob is a per-day probability in [0, 1), got {}",
                jumps.prob
            )));
        }
        for (f, &s) in &jumps.stdev {
            if !(s >= 0.0) || !s.is_finite() {
                return Err(FxVarError::invalid(format!("jump stdev for {f} must be finite and >= 0")));
            }
        }
        for (f, &m) in &jumps.mean {
            if !m.is_finite() {
                return Err(FxVarError::invalid(format!("jump mean for {f} must be finite")));
            }
        }
        jump_lam_h = -(-jumps.prob).ln_1p() * horizon_days;
        if jump_lam_h > 700.0 {
            return Err(FxVarError::invalid(format!(
                "Poisson jump mean lam * h = {jump_lam_h} is too large"
            )));
        }
    }

    let mut a = cov.cov.clone();
    for i in 0..k {
        for j in 0..k {
            a.set(i, j, a.get(i, j) * horizon_days);
        }
    }
    let cf = a.cholesky_repair(crate::psd::PsdRepair::None)?;
    let (lower, jitter) = (cf.factor, cf.jitter);
    let warning = if jitter > 0.0 {
        format!(
            "covariance was not positive definite; added diagonal jitter {jitter:e} to \
             factorise (pegged/managed-currency block is the expected trigger)"
        )
    } else {
        String::new()
    };

    let mut rng = Rng::new(seed);
    let mut data = Matrix::zeros(n_scenarios, k);

    // Pre-resolve jump columns so the per-scenario loop is branch-light.
    let mut jump_col = Vec::new();
    let mut jump_mean = Vec::new();
    let mut jump_std = Vec::new();
    if dist == McDist::Jump {
        for (f, &m) in &jumps.mean {
            if let Some(col) = cov.factors.iter().position(|x| x == f) {
                jump_col.push(col);
                jump_mean.push(m);
                jump_std.push(jumps.stdev.get(f).copied().unwrap_or(0.0));
            }
            // scenario library convention: ignore unknown factors
        }
    }

    let t_scale = if dist == McDist::StudentT { ((df - 2.0) / df).sqrt() } else { 1.0 };
    let mut z = vec![0.0; k];
    for s in 0..n_scenarios {
        for zi in z.iter_mut() {
            *zi = rng.normal();
        }
        for i in 0..k {
            let li = lower.row(i);
            let mut acc = 0.0;
            for j in 0..=i {
                acc += li[j] * z[j];
            }
            data.set(s, i, acc);
        }
        if dist == McDist::StudentT {
            let w = rng.chi_square(df) / df;
            let m = t_scale / w.sqrt();
            for i in 0..k {
                let v = data.get(s, i) * m;
                data.set(s, i, v);
            }
        } else if dist == McDist::Jump {
            let n_jumps = poisson(&mut rng, jump_lam_h) as f64;
            // Draw the jump normals unconditionally so the scenario count,
            // not the jump pattern, fixes the random stream.
            for idx in 0..jump_col.len() {
                let zj = if jump_std[idx] > 0.0 { rng.normal() } else { 0.0 };
                if n_jumps > 0.0 {
                    let col = jump_col[idx];
                    let v = data.get(s, col) + n_jumps * jump_mean[idx] + jump_std[idx] * n_jumps.sqrt() * zj;
                    data.set(s, col, v);
                }
            }
        }
    }
    Ok((ReturnsMatrix { factors: cov.factors.clone(), data }, warning, jitter))
}

/// Poisson(`mu`) draw by sequential inversion (one uniform per draw);
/// `mu <= 700` so `exp(-mu)` does not underflow.
fn poisson(rng: &mut Rng, mu: f64) -> u64 {
    if mu <= 0.0 {
        return 0;
    }
    let u = rng.uniform();
    let mut p = (-mu).exp();
    let mut cdf = p;
    let mut k = 0u64;
    while u > cdf && k < 100_000 {
        k += 1;
        p *= mu / k as f64;
        cdf += p;
        if p == 0.0 && cdf < u {
            break;
        }
    }
    k
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
pub fn var_standard_error_bootstrap(pnl: &[f64], alpha: f64, n_boot: usize, seed: u64) -> Result<f64> {
    validate_alpha(alpha)?;
    if pnl.len() < 10 {
        return Err(FxVarError::Invalid(format!(
            "need at least 10 observations to bootstrap, got {}",
            pnl.len()
        )));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(FxVarError::Invalid("pnl contains NaN or infinite values".to_string()));
    }
    if n_boot < 2 {
        return Err(FxVarError::Invalid(format!(
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
/// understate the SE; [`var_standard_error_bootstrap`] is the primary estimate.
///
/// # Errors
/// Invalid `alpha`, fewer than 10 or non-finite observations.
pub fn var_standard_error(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    let n = pnl.len();
    if n < 10 {
        return Err(FxVarError::Invalid(format!(
            "need at least 10 scenarios for a VaR standard error, got {n}"
        )));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(FxVarError::Invalid("pnl contains NaN or infinite values".to_string()));
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
        return Err(FxVarError::Numerical(
            "KDE density at the VaR quantile underflowed to zero".to_string(),
        ));
    }
    Ok((alpha * (1.0 - alpha) / nf).sqrt() / dens)
}

/// Monte Carlo VaR/ES with full revaluation of the book, simulated once:
/// VaR, ES, the bootstrap SE and the KDE SE all come from the same
/// scenarios. `cov` must cover
/// every factor in `book.factors()` (extra factors are ignored).
///
/// # Errors
/// [`FxVarError::Invalid`] for an empty book, bad options, or missing
/// covariance factors; a factorless (pure base-ccy cash) book reports
/// exactly zero.
pub fn monte_carlo_var(
    book: &Book,
    market: &Market,
    cov: &FactorCov,
    opts: &MonteCarloOptions,
) -> Result<MonteCarloResult> {
    validate_alpha(opts.alpha)?;
    validate_horizon(opts.horizon_days)?;
    if opts.n_boot < 2 {
        return Err(FxVarError::invalid(format!("n_boot must be >= 2, got {}", opts.n_boot)));
    }
    let compiled = CompiledBook::new(book, market)?;

    let mut res = MonteCarloResult {
        var: 0.0,
        es: 0.0,
        alpha: opts.alpha,
        horizon_days: opts.horizon_days,
        dist: opts.dist,
        n_scenarios: opts.n_scenarios,
        se_var: 0.0,
        se_var_kde: 0.0,
        jitter: 0.0,
        pnl: Vec::new(),
        cholesky_warning: String::new(),
    };
    if compiled.factors().is_empty() {
        res.pnl = vec![0.0; opts.n_scenarios];
        return Ok(res); // pure base-ccy cash: zero risk
    }

    let sub = cov.select(compiled.factors())?;
    let (scen, warning, jitter) = simulate_factor_returns_jitter(
        &sub,
        opts.n_scenarios,
        opts.dist,
        opts.df,
        &opts.jumps,
        opts.seed,
        opts.horizon_days,
    )?;
    res.jitter = jitter;
    res.cholesky_warning = warning;
    res.pnl = compiled.pnl_scenarios(&scen)?;
    let (var, es) = empirical_var_es(&res.pnl, opts.alpha, None)?;
    res.var = var;
    res.es = es;
    res.se_var = var_standard_error_bootstrap(&res.pnl, opts.alpha, opts.n_boot, opts.seed ^ 0x9E37_79B9_7F4A_7C15)?;
    res.se_var_kde = var_standard_error(&res.pnl, opts.alpha)?;
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::{Position, SpotPosition};
    use crate::TailDist;

    fn cov1() -> FactorCov {
        FactorCov { factors: vec!["FX:EUR".to_string()], cov: Matrix::from_rows(&[vec![0.0001]]).unwrap() }
    }

    #[test]
    fn simulate_is_deterministic_for_fixed_seed() {
        let (a, _) = simulate_factor_returns(&cov1(), 100, McDist::Normal, 6.0, &JumpSpec::default(), 7, 1.0).unwrap();
        let (b, _) = simulate_factor_returns(&cov1(), 100, McDist::Normal, 6.0, &JumpSpec::default(), 7, 1.0).unwrap();
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn different_seeds_differ() {
        let (a, _) = simulate_factor_returns(&cov1(), 100, McDist::Normal, 6.0, &JumpSpec::default(), 1, 1.0).unwrap();
        let (b, _) = simulate_factor_returns(&cov1(), 100, McDist::Normal, 6.0, &JumpSpec::default(), 2, 1.0).unwrap();
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn student_t_dist_has_fatter_realised_tails() {
        let big_cov = FactorCov {
            factors: vec!["FX:EUR".to_string()],
            cov: Matrix::from_rows(&[vec![0.0001]]).unwrap(),
        };
        let (n, _) =
            simulate_factor_returns(&big_cov, 20_000, McDist::Normal, 6.0, &JumpSpec::default(), 11, 1.0).unwrap();
        let (t, _) =
            simulate_factor_returns(&big_cov, 20_000, McDist::StudentT, 5.0, &JumpSpec::default(), 11, 1.0).unwrap();
        let max_abs = |m: &Matrix| m.as_slice().iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(max_abs(&t.data) > max_abs(&n.data));
    }

    #[test]
    fn jump_overlay_adds_tail_mass() {
        // p = 1 - e^{-5}: lam = 5 jumps a day, P(N = 0) = e^{-5} ~ 0.7 %.
        let mut jumps = JumpSpec { prob: 1.0 - (-5.0_f64).exp(), ..Default::default() };
        jumps.mean.insert("FX:EUR".to_string(), -0.10);
        let (scen, _) =
            simulate_factor_returns(&cov1(), 50, McDist::Jump, 6.0, &jumps, 3, 1.0).unwrap();
        let hits = (0..50).filter(|&i| scen.data.get(i, 0) < -0.05).count();
        assert!(hits >= 45, "hits = {hits}");
        // p = 1 is no longer a valid per-day probability (infinite intensity).
        jumps.prob = 1.0;
        assert!(simulate_factor_returns(&cov1(), 50, McDist::Jump, 6.0, &jumps, 3, 1.0).is_err());
    }

    #[test]
    fn jump_count_scales_with_horizon() {
        // Per-day p = 0.02 -> lam = -ln(0.98). Mean jump P&L over h days is
        // lam * h * mu_J (uncompensated). The old Bernoulli overlay gave
        // p * mu_J whatever h, so its 10-day mean was 10x too small.
        let mut jumps = JumpSpec { prob: 0.02, ..Default::default() };
        jumps.mean.insert("FX:EUR".to_string(), -0.10);
        let tiny = FactorCov { factors: vec!["FX:EUR".to_string()], cov: Matrix::from_rows(&[vec![1e-12]]).unwrap() };
        let n = 200_000;
        let (scen, _) = simulate_factor_returns(&tiny, n, McDist::Jump, 6.0, &jumps, 9, 10.0).unwrap();
        let mean = scen.data.as_slice().iter().sum::<f64>() / n as f64;
        let lam_h = -(0.98_f64).ln() * 10.0;
        let want = lam_h * -0.10;
        // SE of the mean: 0.1 * sqrt(lam_h / n) ~ 1e-4; allow 5 SE.
        let se = 0.10 * (lam_h / n as f64).sqrt();
        assert!((mean - want).abs() < 5.0 * se, "mean {mean} want {want} se {se}");
    }

    #[test]
    fn monte_carlo_var_normal_agrees_with_parametric() {
        let market = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
        let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))], "USD").unwrap();
        let cov = cov1();
        let opts = MonteCarloOptions { n_scenarios: 200_000, seed: 42, ..Default::default() };
        let mc = monte_carlo_var(&book, &market, &cov, &opts).unwrap();
        let exposures = CompiledBook::new(&book, &market).unwrap().linear_exposures(1e-6).unwrap();
        let closed = crate::parametric::var_covar(&exposures, &cov.cov, 0.01, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        // Full revaluation: P&L = 11m (e^r - 1), r ~ N(0, 0.01^2), so the
        // exact 99 % VaR is 11m (1 - e^{z sigma}), z = Phi^{-1}(0.01); the
        // delta closed form is ~1 % higher (convexity).
        let z = crate::stats::inv_norm_cdf(0.01).unwrap();
        let exact = 11.0e6 * (1.0 - (z * 0.01).exp());
        // 4 bootstrap SEs: a two-sided 6e-5 miss probability for a correct engine.
        assert!((mc.var - exact).abs() < 4.0 * mc.se_var, "{} vs {} (se {})", mc.var, exact, mc.se_var);
        assert!(closed.var > exact);
        assert!(mc.se_var > 0.0 && mc.se_var_kde > 0.0);
    }

    #[test]
    fn empty_book_errors() {
        let market = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
        assert!(monte_carlo_var(&Book::empty(), &market, &cov1(), &MonteCarloOptions::default()).is_err());
    }

    #[test]
    fn n_scenarios_zero_errors() {
        let opts = MonteCarloOptions { n_scenarios: 0, ..Default::default() };
        assert!(
            simulate_factor_returns(&cov1(), opts.n_scenarios, opts.dist, opts.df, &opts.jumps, opts.seed, 1.0)
                .is_err()
        );
    }

    #[test]
    fn pegged_currencies_trigger_cholesky_jitter() {
        // Two perfectly correlated pegs: an exactly-representable rank-1
        // matrix (v v' for v = [0.25, 0.125], both binary-exact), so the
        // second Cholesky pivot is exactly zero and the jitter path is
        // guaranteed to engage (a general rank-1 double matrix can round to
        // a tiny positive pivot instead).
        let cov = FactorCov {
            factors: vec!["FX:HKD".to_string(), "FX:AED".to_string()],
            cov: Matrix::from_rows(&[vec![6.25e-2, 3.125e-2], vec![3.125e-2, 1.5625e-2]]).unwrap(),
        };
        let (_, warning) =
            simulate_factor_returns(&cov, 100, McDist::Normal, 6.0, &JumpSpec::default(), 1, 1.0).unwrap();
        assert!(!warning.is_empty());
    }

    fn synthetic_pnl(n: usize) -> Vec<f64> {
        // Deterministic synthetic P&L via the crate's own RNG (normal-ish,
        // heavier in the left tail so the KDE/bootstrap comparison is
        // meaningful rather than trivially close).
        let mut rng = Rng::new(123);
        (0..n).map(|_| 1_000.0 * rng.normal()).collect()
    }

    #[test]
    fn bootstrap_se_is_deterministic_for_fixed_seed() {
        let pnl = synthetic_pnl(500);
        let a = var_standard_error_bootstrap(&pnl, 0.01, 200, 7).unwrap();
        let b = var_standard_error_bootstrap(&pnl, 0.01, 200, 7).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bootstrap_se_errors_on_too_few_scenarios() {
        let pnl = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0];
        assert!(var_standard_error_bootstrap(&pnl, 0.01, 100, 0).is_err());
        // n_boot = 1 used to divide by zero in the sample std.
        assert!(var_standard_error_bootstrap(&synthetic_pnl(50), 0.01, 1, 0).is_err());
    }

    #[test]
    fn bootstrap_se_errors_on_invalid_alpha() {
        let pnl = synthetic_pnl(50);
        assert!(var_standard_error_bootstrap(&pnl, 0.0, 100, 0).is_err());
        assert!(var_standard_error_bootstrap(&pnl, 0.99, 100, 0).is_err());
        assert!(var_standard_error_bootstrap(&pnl, 1.0, 100, 0).is_err());
        assert!(var_standard_error_bootstrap(&pnl, 1.5, 100, 0).is_err());
    }

    #[test]
    fn bootstrap_se_agrees_with_kde_within_a_sane_multiple() {
        let pnl = synthetic_pnl(20_000);
        let kde_se = var_standard_error(&pnl, 0.001).unwrap();
        let boot_se = var_standard_error_bootstrap(&pnl, 0.001, 500, 1).unwrap();
        assert!(boot_se > 0.0 && kde_se > 0.0);
        assert!(boot_se < 3.0 * kde_se);
        assert!(kde_se < 3.0 * boot_se);
    }

    #[test]
    fn bootstrap_se_handles_degenerate_constant_pnl_without_nan_or_panic() {
        let pnl = vec![-5.0; 50];
        let se = var_standard_error_bootstrap(&pnl, 0.01, 100, 0).unwrap();
        assert!(se.is_finite());
        assert_eq!(se, 0.0);
    }
}
