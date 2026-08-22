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
//! * [`crate::McDist::Jump`] — normal diffusion plus a `Bernoulli(p)`
//!   common jump event with per-factor jump sizes `~ N(mu_J, sigma_J)`:
//!   the devaluation / peg-break overlay. The jump *adds* variance on top
//!   of `Sigma` by design — it models exactly the risk the covariance
//!   matrix cannot see.
//!
//! # Determinism
//!
//! All randomness flows through [`crate::rng::Rng`] (xoshiro256++ driving
//! inverse-CDF transforms implemented in this crate). A fixed seed gives
//! bit-reproducible scenario sets and VaR figures across runs on this
//! platform.
//!
//! # Standard error
//!
//! [`var_standard_error`] uses the asymptotic order-statistic formula
//! `SE = sqrt(a(1-a)/n) / f(q)` with a Gaussian-KDE (Silverman bandwidth)
//! density estimate at the quantile. Convergence tests accept MC vs closed
//! form within 3 SE.
//!
//! **Known limitation** (see `docs/VALIDATION.md` F5): a fixed-bandwidth
//! (Silverman) KDE is tuned to the bulk of the P&L distribution, not the
//! tail it is evaluated at, so at deep confidence levels (`alpha >= 0.995`)
//! or with modest scenario counts it *systematically underestimates* the
//! true sampling SE — directionally overconfident, not just noisy. This is
//! a property of fixed-bandwidth density estimation at extreme quantiles
//! generally, not a bug specific to this implementation; the Python
//! reference (`fx_var.monte_carlo_var`) shares it and documents the same
//! benchmark. [`var_standard_error_bootstrap`] sidesteps it entirely
//! (distribution-free, no bandwidth to choose - unbiased to ~1-2% in the
//! same benchmark, at the cost of higher trial-to-trial variance in the SE
//! estimate itself unless `n_boot` is generous); prefer it to cross-check
//! the KDE estimate whenever `alpha >= 0.995` or scenario counts are
//! modest.

use std::collections::HashMap;

use crate::book::{Book, CompiledBook, Market};
use crate::expected_shortfall::{empirical_var, empirical_var_es};
use crate::matrix::Matrix;
use crate::returns::{FactorCov, ReturnsMatrix};
use crate::rng::Rng;
use crate::stats::{norm_pdf, sample_std, validate_alpha, validate_horizon};
use crate::{FxVarError, McDist, Result};

/// Common-jump overlay for [`McDist::Jump`]: with per-scenario probability
/// `prob`, every factor listed in `mean` jumps by `mean[f] + std[f] * Z`
/// (log-return units for FX factors; factors not listed do not jump). E.g.
/// `{"FX:TRY": -0.15}` is a 15% (log) devaluation of the lira vs USD.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JumpSpec {
    /// Per-scenario jump probability, in `[0, 1]`.
    pub prob: f64,
    /// Jump mean per factor.
    pub mean: HashMap<String, f64>,
    /// Jump standard deviation per factor (`>= 0`).
    pub stdev: HashMap<String, f64>,
}

/// Options controlling [`monte_carlo_var`].
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarloOptions {
    /// Confidence level, e.g. `0.99` for 99 % VaR.
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
}

impl Default for MonteCarloOptions {
    fn default() -> Self {
        MonteCarloOptions {
            alpha: 0.99,
            horizon_days: 1.0,
            n_scenarios: 50_000,
            dist: McDist::Normal,
            df: 6.0,
            jumps: JumpSpec::default(),
            seed: 0,
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
    /// Confidence level used.
    pub alpha: f64,
    /// Horizon in trading days used.
    pub horizon_days: f64,
    /// Simulation distribution used.
    pub dist: McDist,
    /// Number of scenarios simulated.
    pub n_scenarios: usize,
    /// Asymptotic standard error of the VaR estimate.
    pub se_var: f64,
    /// Simulated scenario P&L.
    pub pnl: Vec<f64>,
    /// Non-empty when the covariance needed diagonal jitter to factorise.
    pub cholesky_warning: String,
}

/// Simulate factor-return scenarios from a daily covariance. `cov.cov` is
/// scaled by `horizon_days` (i.i.d. aggregation).
///
/// # Errors
/// [`FxVarError::Invalid`] for `n_scenarios < 1`, Student-t `df <= 2`, a
/// jump probability outside `[0, 1]`, negative jump stdev, or a shape
/// mismatch between `cov.factors` and `cov.cov`;
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
    if dist == McDist::StudentT && df <= 2.0 {
        return Err(FxVarError::invalid("Student-t df must be > 2 for finite variance"));
    }
    if dist == McDist::Jump {
        if !(jumps.prob >= 0.0 && jumps.prob <= 1.0) {
            return Err(FxVarError::invalid(format!("jump prob must be in [0, 1], got {}", jumps.prob)));
        }
        for (f, &s) in &jumps.stdev {
            if s < 0.0 {
                return Err(FxVarError::invalid(format!("jump stdev for {f} must be >= 0")));
            }
        }
    }

    let mut a = cov.cov.clone();
    for i in 0..k {
        for j in 0..k {
            a.set(i, j, a.get(i, j) * horizon_days);
        }
    }
    let (lower, jitter) = a.cholesky_with_jitter(8)?;
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
            let hit = rng.uniform() < jumps.prob;
            // Draw the jump normals unconditionally so the scenario count,
            // not the hit pattern, fixes the random stream.
            for idx in 0..jump_col.len() {
                let zj = if jump_std[idx] > 0.0 { rng.normal() } else { 0.0 };
                if hit {
                    let col = jump_col[idx];
                    let v = data.get(s, col) + jump_mean[idx] + jump_std[idx] * zj;
                    data.set(s, col, v);
                }
            }
        }
    }
    Ok((ReturnsMatrix { factors: cov.factors.clone(), data }, warning))
}

/// Asymptotic standard error of the empirical VaR estimate:
/// `SE = sqrt(alpha (1-alpha) / n) / f_hat(q)`, `f_hat` a Gaussian KDE with
/// Silverman bandwidth evaluated at the loss quantile.
///
/// # Errors
/// [`FxVarError::Invalid`] for fewer than 10 scenarios or bad `alpha`.
pub fn var_standard_error(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    let n = pnl.len();
    if n < 10 {
        return Err(FxVarError::invalid("need at least 10 scenarios for a VaR standard error"));
    }
    // Loss quantile q with the same tail convention as empirical_var.
    let q = -empirical_var(pnl, alpha, None)?;

    let sd = sample_std(pnl);
    let mut sorted = pnl.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
    let quantile_sorted = |p: f64| {
        let pos = p * (sorted.len() - 1) as f64;
        let lo = pos as usize;
        let hi = (lo + 1).min(sorted.len() - 1);
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    };
    let iqr = quantile_sorted(0.75) - quantile_sorted(0.25);
    let mut spread = sd;
    if iqr > 0.0 {
        spread = sd.min(iqr / 1.34);
    }
    if !(spread > 0.0) {
        spread = 1e-300;
    }
    let h = 0.9 * spread * (n as f64).powf(-0.2);

    let mut dens = 0.0;
    for &x in pnl {
        dens += norm_pdf((q - x) / h);
    }
    dens /= n as f64 * h;
    dens = dens.max(1e-300);
    Ok((alpha * (1.0 - alpha) / n as f64).sqrt() / dens)
}

/// Bootstrap standard error of the empirical VaR estimate (distribution-free
/// cross-check for [`var_standard_error`]'s KDE estimate).
///
/// Resamples `pnl` with replacement `n_boot` times (via the crate's own
/// [`Rng`]) and applies [`empirical_var`] — the same order-statistic VaR
/// rule used by [`monte_carlo_var`]'s point estimate, uniform weights, so
/// the tail rank is fixed by `(n, alpha)` alone and identical across every
/// resample — to each resample, returning the standard deviation (`ddof =
/// 1`) of the resulting VaR estimates.
///
/// # Errors
/// [`FxVarError::Invalid`] for fewer than 10 scenarios or bad `alpha`.
pub fn var_standard_error_bootstrap(pnl: &[f64], alpha: f64, n_boot: usize, seed: u64) -> Result<f64> {
    validate_alpha(alpha)?;
    let n = pnl.len();
    if n < 10 {
        return Err(FxVarError::invalid("need at least 10 scenarios to bootstrap"));
    }

    let mut rng = Rng::new(seed);
    let mut boot_vars = Vec::with_capacity(n_boot);
    let mut resample = vec![0.0; n];
    for _ in 0..n_boot {
        for x in resample.iter_mut() {
            let idx = ((rng.uniform() * n as f64) as usize).min(n - 1);
            *x = pnl[idx];
        }
        boot_vars.push(empirical_var(&resample, alpha, None)?);
    }
    Ok(sample_std(&boot_vars))
}

/// Monte Carlo VaR/ES with full revaluation of the book. `cov` must cover
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
    let compiled = CompiledBook::new(book, market)?;

    let mut res = MonteCarloResult {
        var: 0.0,
        es: 0.0,
        alpha: opts.alpha,
        horizon_days: opts.horizon_days,
        dist: opts.dist,
        n_scenarios: opts.n_scenarios,
        se_var: 0.0,
        pnl: Vec::new(),
        cholesky_warning: String::new(),
    };
    if compiled.factors().is_empty() {
        res.pnl = vec![0.0; opts.n_scenarios];
        return Ok(res); // pure base-ccy cash: zero risk
    }

    let sub = cov.select(compiled.factors())?;
    let (scen, warning) = simulate_factor_returns(
        &sub,
        opts.n_scenarios,
        opts.dist,
        opts.df,
        &opts.jumps,
        opts.seed,
        opts.horizon_days,
    )?;
    res.cholesky_warning = warning;
    res.pnl = compiled.pnl_scenarios(&scen)?;
    let (var, es) = empirical_var_es(&res.pnl, opts.alpha, None)?;
    res.var = var;
    res.es = es;
    res.se_var = var_standard_error(&res.pnl, opts.alpha)?;
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
        let mut jumps = JumpSpec { prob: 1.0, ..Default::default() }; // always jump
        jumps.mean.insert("FX:EUR".to_string(), -0.10);
        let (scen, _) =
            simulate_factor_returns(&cov1(), 50, McDist::Jump, 6.0, &jumps, 3, 1.0).unwrap();
        for i in 0..50 {
            assert!(scen.data.get(i, 0) < -0.05);
        }
    }

    #[test]
    fn monte_carlo_var_normal_agrees_with_parametric() {
        let market = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
        let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))], "USD").unwrap();
        let cov = cov1();
        let opts = MonteCarloOptions { n_scenarios: 200_000, seed: 42, ..Default::default() };
        let mc = monte_carlo_var(&book, &market, &cov, &opts).unwrap();
        let exposures = CompiledBook::new(&book, &market).unwrap().linear_exposures(1e-6).unwrap();
        let closed = crate::parametric::var_covar(&exposures, &cov.cov, 0.99, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        assert!((mc.var - closed.var).abs() < 6.0 * mc.se_var.max(1.0));
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
        let a = var_standard_error_bootstrap(&pnl, 0.99, 200, 7).unwrap();
        let b = var_standard_error_bootstrap(&pnl, 0.99, 200, 7).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bootstrap_se_errors_on_too_few_scenarios() {
        let pnl = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0];
        assert!(var_standard_error_bootstrap(&pnl, 0.99, 100, 0).is_err());
    }

    #[test]
    fn bootstrap_se_errors_on_invalid_alpha() {
        let pnl = synthetic_pnl(50);
        assert!(var_standard_error_bootstrap(&pnl, 0.0, 100, 0).is_err());
        assert!(var_standard_error_bootstrap(&pnl, 1.0, 100, 0).is_err());
        assert!(var_standard_error_bootstrap(&pnl, 1.5, 100, 0).is_err());
    }

    #[test]
    fn bootstrap_se_agrees_with_kde_within_a_sane_multiple() {
        let pnl = synthetic_pnl(20_000);
        let kde_se = var_standard_error(&pnl, 0.999).unwrap();
        let boot_se = var_standard_error_bootstrap(&pnl, 0.999, 500, 1).unwrap();
        assert!(boot_se > 0.0 && kde_se > 0.0);
        assert!(boot_se < 3.0 * kde_se);
        assert!(kde_se < 3.0 * boot_se);
    }

    #[test]
    fn bootstrap_se_handles_degenerate_constant_pnl_without_nan_or_panic() {
        let pnl = vec![-5.0; 50];
        let se = var_standard_error_bootstrap(&pnl, 0.99, 100, 0).unwrap();
        assert!(se.is_finite());
        assert_eq!(se, 0.0);
    }
}
