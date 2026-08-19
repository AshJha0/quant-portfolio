//! Edge cases from the documentation contract: empty inputs, single-asset
//! portfolios, alpha at the boundaries of (0, 0.5), zero-variance assets,
//! degenerate P&L series, and returns/pnl-mapping validation. Each case here
//! is also documented in docs/VALIDATION.md.

use eq_var_engine::expected_shortfall::expected_shortfall;
use eq_var_engine::historical::{historical_var, linear_quantile, MIN_OBS};
use eq_var_engine::matrix::Matrix;
use eq_var_engine::monte_carlo::{monte_carlo_var, portfolio_pnl};
use eq_var_engine::parametric::{parametric_var, portfolio_sigma};
use eq_var_engine::stats::{excess_kurtosis, mean, normal_ppf, skewness};
use eq_var_engine::{EqVarError, TailModel};

#[test]
fn empty_inputs_error() {
    let empty: Vec<f64> = vec![];
    let cov = Matrix::from_vec(1, 1, vec![1e-4]).unwrap();
    assert!(historical_var(&empty, 0.01).is_err());
    assert!(expected_shortfall(&empty, 0.01).is_err());
    assert!(linear_quantile(&empty, 0.5).is_err());
    assert!(portfolio_sigma(&empty, &cov).is_err());
    assert!(parametric_var(&empty, &cov, 0.01, TailModel::Normal).is_err());
    assert!(monte_carlo_var(&empty, &cov, 0.01, 1000, TailModel::Normal, 0).is_err());
    assert!(portfolio_pnl(&Matrix::zeros(5, 1), &empty).is_err());
    assert!(mean(&empty).is_err());
}

#[test]
fn single_asset_portfolio_is_exact() {
    let (vol, w0) = (0.02, 1.0e6);
    let cov = Matrix::from_vec(1, 1, vec![vol * vol]).unwrap();
    let w = [w0];
    let sigma = w0 * vol;
    assert!((portfolio_sigma(&w, &cov).unwrap() - sigma).abs() < 1e-9);
    let var = parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap();
    assert!((var - (-normal_ppf(0.01).unwrap() * sigma)).abs() < 1e-9);
    let mc = monte_carlo_var(&w, &cov, 0.01, 100_000, TailModel::Normal, 5).unwrap();
    let se = eq_var_engine::monte_carlo::var_order_statistic_se(
        &eq_var_engine::monte_carlo::monte_carlo_pnl(&w, &cov, 100_000, TailModel::Normal, 5)
            .unwrap(),
        0.01,
    )
    .unwrap();
    assert!((mc - var).abs() < 3.0 * se);
    // Short single-asset exposure has identical risk (symmetry of the normal).
    let ws = [-w0];
    let var_short = parametric_var(&ws, &cov, 0.01, TailModel::Normal).unwrap();
    assert!((var_short - var).abs() < 1e-9);
}

#[test]
fn alpha_bounds_rejected_everywhere() {
    let pnl: Vec<f64> = (0..100).map(|t| (1.3 * t as f64).sin() * 100.0).collect();
    let cov = Matrix::from_vec(1, 1, vec![1e-4]).unwrap();
    let w = [1.0e6];
    for bad in [0.0, -0.01, 0.5, 0.75, 1.0] {
        assert!(historical_var(&pnl, bad).is_err(), "{bad}");
        assert!(expected_shortfall(&pnl, bad).is_err(), "{bad}");
        assert!(parametric_var(&w, &cov, bad, TailModel::Normal).is_err(), "{bad}");
    }
    assert!(historical_var(&pnl, 0.499).is_ok());
    assert!(historical_var(&pnl, 0.011).is_ok());
}

#[test]
fn zero_variance_asset_handled_throughout() {
    // Asset 2 is riskless: covariance is singular but PSD.
    let cov = Matrix::from_vec(2, 2, vec![4.0e-4, 0.0, 0.0, 0.0]).unwrap();
    let w = [1.0e6, 5.0e5];
    let sigma = portfolio_sigma(&w, &cov).unwrap();
    assert!((sigma - 1.0e6 * 0.02).abs() < 1e-9);
    assert!(cov.cholesky().is_err()); // plain Cholesky fails on the singular matrix
    let l = cov.cholesky_jitter(1e-10, 12).unwrap();
    assert_eq!(l.rows(), 2);
    let mc = monte_carlo_var(&w, &cov, 0.01, 100_000, TailModel::Normal, 17).unwrap();
    assert!((mc - (-normal_ppf(0.01).unwrap() * sigma)).abs() < 0.05 * sigma);
}

#[test]
fn fully_riskless_portfolio() {
    let cov = Matrix::zeros(2, 2);
    let w = [1.0e6, -1.0e6];
    assert_eq!(portfolio_sigma(&w, &cov).unwrap(), 0.0);
    assert_eq!(parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap(), 0.0);
    assert_eq!(
        eq_var_engine::expected_shortfall::normal_es(0.0, 0.01, 0.0).unwrap(),
        0.0
    );
}

#[test]
fn constant_pnl_series() {
    let flat = vec![25.0; 100];
    assert_eq!(historical_var(&flat, 0.05).unwrap(), -25.0);
    assert_eq!(expected_shortfall(&flat, 0.05).unwrap(), -25.0);
    let zeros = vec![0.0; 100];
    assert_eq!(
        eq_var_engine::historical::filtered_historical_var(&zeros, 0.05, 0.94).unwrap(),
        0.0
    );
    assert_eq!(skewness(&flat).unwrap(), 0.0);
    assert_eq!(excess_kurtosis(&flat).unwrap(), 0.0);
}

#[test]
fn non_finite_pnl_rejected() {
    let mut pnl = vec![1.0; 100];
    pnl[3] = f64::INFINITY;
    assert!(historical_var(&pnl, 0.05).is_err());
    pnl[3] = f64::NAN;
    assert!(expected_shortfall(&pnl, 0.05).is_err());
    assert!(eq_var_engine::historical::age_weighted_var(&pnl, 0.05, 0.98).is_err());
}

#[test]
fn portfolio_pnl_shape_mismatch() {
    let panel = Matrix::from_vec(10, 3, vec![0.01; 30]).unwrap();
    assert!(portfolio_pnl(&panel, &[1.0, 2.0]).is_err());
    let ok = [1.0, 2.0, 3.0];
    let pnl = portfolio_pnl(&panel, &ok).unwrap();
    assert_eq!(pnl.len(), 10);
    assert!((pnl[0] - 0.06).abs() < 1e-15);
}

#[test]
fn min_obs_guard_exact_boundary() {
    let ok: Vec<f64> = (0..MIN_OBS).map(|t| (0.9 * t as f64).sin()).collect();
    assert!(historical_var(&ok, 0.05).is_ok());
    let short_series = &ok[..ok.len() - 1];
    assert!(historical_var(short_series, 0.05).is_err());
}

#[test]
fn deep_tail_alpha_on_small_sample() {
    // alpha far below 1/n: the type-7 quantile interpolates just above the
    // sample minimum (h = (n-1) alpha < 1) — the tail is unresolvable and the
    // estimate is pinned to the worst observed loss, never extrapolated
    // beyond it. Documented failure mode (docs/VALIDATION.md).
    let pnl: Vec<f64> = (0..100)
        .map(|t| 10.0 * (2.3 * t as f64).sin() - if t == 50 { 500.0 } else { 0.0 })
        .collect();
    let v = historical_var(&pnl, 1e-6).unwrap();
    let mut sorted = pnl.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = 99.0 * 1e-6;
    let expected = -(sorted[0] + h * (sorted[1] - sorted[0]));
    assert!((v - expected).abs() < 1e-12 * expected.abs());
    assert!(v <= -sorted[0]); // never exceeds the worst observed loss
    assert!(v > -sorted[1]); // ... but stays pinned to it
}

/// A NaN anywhere in the risk pipeline must produce an error, never a
/// number.
///
/// This is the single most dangerous failure mode in a risk system. Guards
/// written as `if x < 0.0 { return Err(..) }` silently accept NaN, because
/// every IEEE-754 comparison against NaN is false — Rust is no different
/// from C here (`f64::NAN < 0.0` is `false`). Worse, `f64::max` propagates
/// the *other* operand, so `NaN.max(0.0)` is `0.0`: before this guard, a
/// NaN covariance entry made `portfolio_sigma` — and therefore the whole
/// parametric VaR/ES family — report exactly **zero risk**.
#[test]
fn non_finite_covariance_and_exposures_are_rejected_not_silently_zeroed() {
    use eq_var_engine::expected_shortfall::{normal_es, parametric_es, student_t_es};
    use eq_var_engine::matrix::{covariance_from_vols, sample_covariance};
    use eq_var_engine::monte_carlo::{monte_carlo_es, monte_carlo_pnl, simulate_factor_returns};
    use eq_var_engine::parametric::{cornish_fisher_var, parametric_var_full};

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Poisoned covariance: every consumer must error.
        let cov = Matrix::from_vec(2, 2, vec![4.0e-4, bad, bad, 1.0e-4]).unwrap();
        let w = [1.0e6, -5.0e5];
        assert!(portfolio_sigma(&w, &cov).is_err(), "sigma with cov={bad}");
        assert!(parametric_var(&w, &cov, 0.01, TailModel::Normal).is_err(), "var cov={bad}");
        assert!(parametric_es(&w, &cov, 0.025, TailModel::Normal).is_err(), "es cov={bad}");
        assert!(cov.cholesky().is_err(), "cholesky cov={bad}");
        assert!(cov.cholesky_jitter(1e-10, 12).is_err(), "cholesky_jitter cov={bad}");
        assert!(simulate_factor_returns(&cov, 100, TailModel::Normal, 1).is_err());
        assert!(monte_carlo_var(&w, &cov, 0.01, 1000, TailModel::Normal, 1).is_err());
        assert!(monte_carlo_es(&w, &cov, 0.01, 1000, TailModel::Normal, 1).is_err());
        assert!(monte_carlo_pnl(&w, &cov, 1000, TailModel::Normal, 1).is_err());

        // Poisoned exposures against a clean covariance.
        let clean = Matrix::from_vec(2, 2, vec![4.0e-4, 1.0e-5, 1.0e-5, 1.0e-4]).unwrap();
        let bad_w = [1.0e6, bad];
        assert!(portfolio_sigma(&bad_w, &clean).is_err(), "sigma with w={bad}");
        assert!(parametric_var(&bad_w, &clean, 0.01, TailModel::Normal).is_err());
        assert!(parametric_es(&bad_w, &clean, 0.025, TailModel::Normal).is_err());

        // Poisoned scalar arguments of the closed-form tail formulae.
        assert!(normal_es(bad, 0.025, 0.0).is_err(), "normal_es sigma={bad}");
        assert!(normal_es(1.0, 0.025, bad).is_err(), "normal_es mean={bad}");
        assert!(student_t_es(bad, 0.025, 5.0, 0.0).is_err());
        assert!(student_t_es(1.0, 0.025, bad, 0.0).is_err(), "student_t_es df={bad}");
        assert!(student_t_es(1.0, 0.025, 5.0, bad).is_err());
        assert!(
            parametric_var(&w, &clean, 0.01, TailModel::StudentT { df: bad }).is_err(),
            "StudentT df={bad}"
        );
        assert!(parametric_var_full(&w, &clean, 0.01, TailModel::Normal, bad, 1).is_err());
        assert!(cornish_fisher_var(bad, 0.01, 0.0, 0.0, 0.0, true).is_err());
        assert!(cornish_fisher_var(1.0, 0.01, bad, 0.0, 0.0, true).is_err());
        assert!(cornish_fisher_var(1.0, 0.01, 0.0, bad, 0.0, true).is_err());
        assert!(cornish_fisher_var(1.0, 0.01, 0.0, 0.0, bad, true).is_err());
        // ... and with the domain check switched OFF, which is the path a
        // caller in a hurry takes.
        assert!(cornish_fisher_var(1.0, 0.01, bad, 0.0, 0.0, false).is_err());

        // Poisoned return panel feeding the covariance estimators.
        let mut panel = Matrix::zeros(10, 2);
        for t in 0..10 {
            panel.set(t, 0, 0.01 * (t as f64).sin());
            panel.set(t, 1, 0.01 * (t as f64).cos());
        }
        panel.set(4, 1, bad);
        assert!(sample_covariance(&panel).is_err(), "sample_covariance {bad}");
        assert!(
            eq_var_engine::matrix::ewma_covariance(&panel, 0.94).is_err(),
            "ewma_covariance {bad}"
        );

        // Poisoned vols / correlations.
        let corr = Matrix::from_vec(2, 2, vec![1.0, bad, bad, 1.0]).unwrap();
        assert!(covariance_from_vols(&[0.01, 0.02], &corr).is_err(), "corr={bad}");
        let good_corr = Matrix::from_vec(2, 2, vec![1.0, 0.3, 0.3, 1.0]).unwrap();
        assert!(covariance_from_vols(&[0.01, bad], &good_corr).is_err(), "vol={bad}");
    }

    // Sanity: the same calls on clean inputs return strictly positive risk,
    // so the guards above are not simply rejecting everything.
    let clean = Matrix::from_vec(2, 2, vec![4.0e-4, 1.0e-5, 1.0e-5, 1.0e-4]).unwrap();
    let w = [1.0e6, -5.0e5];
    assert!(portfolio_sigma(&w, &clean).unwrap() > 0.0);
    assert!(parametric_var(&w, &clean, 0.01, TailModel::Normal).unwrap() > 0.0);
}

/// A NaN VaR or a NaN P&L must not be counted as "no exception".
#[test]
fn backtests_reject_non_finite_series_instead_of_scoring_them_green() {
    use eq_var_engine::backtest::{basel_traffic_light, exceptions_from_pnl, BaselZone};

    let mut pnl: Vec<f64> = (0..250).map(|t| 50.0 * (0.7 * t as f64).sin()).collect();
    // A genuinely broken model: losses far beyond the VaR every day.
    let breached: Vec<f64> = vec![-1000.0; 250];
    let ex = exceptions_from_pnl(&breached, &[100.0]).unwrap();
    assert_eq!(ex.iter().map(|&e| e as i64).sum::<i64>(), 250);
    assert_eq!(basel_traffic_light(250, 250).unwrap().zone, BaselZone::Red);

    // Now poison the VaR feed. `pnl < -NaN` is false on every day, so a
    // naive implementation reports ZERO exceptions and the traffic light
    // goes green — a broken model certified as sound.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            exceptions_from_pnl(&breached, &[bad]).is_err(),
            "scalar VaR = {bad} must be rejected"
        );
        let mut var_series = vec![100.0; 250];
        var_series[123] = bad;
        assert!(
            exceptions_from_pnl(&breached, &var_series).is_err(),
            "one bad VaR day ({bad}) must be rejected"
        );
        // ... and symmetrically, a NaN P&L day.
        pnl[7] = bad;
        assert!(
            exceptions_from_pnl(&pnl, &[100.0]).is_err(),
            "one bad P&L day ({bad}) must be rejected"
        );
        pnl[7] = 50.0 * (0.7 * 7.0f64).sin();
    }
    // A negative VaR is still rejected (positive-loss convention).
    assert!(exceptions_from_pnl(&pnl, &[-1.0]).is_err());
    // And the clean series still works.
    assert!(exceptions_from_pnl(&pnl, &[100.0]).is_ok());
}

/// The Cholesky jitter ladder repairs rounding noise, not real indefiniteness.
#[test]
fn jitter_repairs_singular_but_refuses_materially_indefinite_matrices() {
    use eq_var_engine::matrix::MAX_RELATIVE_JITTER;

    // (a) PSD but exactly singular: perfectly correlated factors. Repaired
    //     at the first rung, and the factor reproduces the input.
    let singular = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    assert!(singular.cholesky().is_err());
    let l = singular.cholesky_jitter(1e-10, 12).unwrap();
    let mut recon = 0.0f64;
    for i in 0..2 {
        for j in 0..2 {
            let s: f64 = (0..2).map(|k| l.get(i, k) * l.get(j, k)).sum();
            recon = recon.max((s - singular.get(i, j)).abs());
        }
    }
    assert!(recon < 1e-8, "singular repair changed the matrix by {recon:e}");

    // (b) Rank-deficient 3x3 (factor 3 = factor 1 + factor 2): still PSD,
    //     still repairable, and the resulting VaR is finite and positive.
    let mut rd = Matrix::zeros(3, 3);
    let base = [[4.0e-4, 1.0e-4], [1.0e-4, 9.0e-4]];
    for i in 0..2 {
        for j in 0..2 {
            rd.set(i, j, base[i][j]);
        }
    }
    for i in 0..2 {
        let s = base[i][0] + base[i][1];
        rd.set(i, 2, s);
        rd.set(2, i, s);
    }
    rd.set(2, 2, base[0][0] + 2.0 * base[0][1] + base[1][1]);
    assert!(rd.cholesky().is_err(), "rank-deficient matrix should defeat plain Cholesky");
    assert!(rd.cholesky_jitter(1e-10, 12).is_ok());
    let w = [1.0e6, -2.0e5, 3.0e5];
    let var = parametric_var(&w, &rd, 0.01, TailModel::Normal).unwrap();
    assert!(var > 0.0 && var.is_finite());
    let mc = monte_carlo_var(&w, &rd, 0.01, 50_000, TailModel::Normal, 3).unwrap();
    assert!((mc - var).abs() < 0.1 * var, "MC {mc} vs parametric {var} on a singular cov");

    // (c) Materially indefinite: a genuinely negative eigenvalue, an order
    //     of magnitude larger than the materiality cap. Repairing it would
    //     require inflating the diagonal by ~1e-2 of the mean variance,
    //     which changes the risk number — so it must be an error.
    //     Sigma = [[1, 0], [0, 1]] * 1e-4 with an off-diagonal of 1.1e-4
    //     has eigenvalues 2.1e-4 and -0.1e-4.
    let indef = Matrix::from_vec(2, 2, vec![1.0e-4, 1.1e-4, 1.1e-4, 1.0e-4]).unwrap();
    assert!(indef.cholesky().is_err());
    let res = indef.cholesky_jitter(1e-10, 12);
    assert!(
        matches!(res, Err(EqVarError::Numerical(_))),
        "materially indefinite matrix must not be silently repaired, got {res:?}"
    );
    // The quadratic form check catches it at the parametric entry point too,
    // for the exposures that expose the negative direction.
    let hedge = [1.0e6, -1.0e6];
    assert!(portfolio_sigma(&hedge, &indef).is_err());
    // And the Monte Carlo path refuses rather than simulating nonsense.
    assert!(monte_carlo_var(&hedge, &indef, 0.01, 1000, TailModel::Normal, 1).is_err());

    // (d) The cap is documented and small: a repair must never be a
    //     material fraction of the mean variance.
    assert!(MAX_RELATIVE_JITTER > 0.0 && MAX_RELATIVE_JITTER <= 1e-4);
    // A jitter argument that is itself nonsense is rejected up front.
    assert!(singular.cholesky_jitter(0.0, 12).is_err());
    assert!(singular.cholesky_jitter(f64::NAN, 12).is_err());
}

/// One- and two-element samples: the smallest inputs a VaR engine can see.
#[test]
fn one_and_two_element_samples_behave_predictably() {
    use eq_var_engine::expected_shortfall::expected_shortfall;
    use eq_var_engine::historical::{historical_var, linear_quantile};

    // linear_quantile is the only estimator defined on a single point: it
    // returns that point at every level (a degenerate but correct CDF).
    for q in [0.0, 0.01, 0.5, 0.99, 1.0] {
        assert_eq!(linear_quantile(&[42.0], q).unwrap(), 42.0);
    }
    // Two points: exact linear interpolation between the order statistics.
    assert_eq!(linear_quantile(&[-10.0, 10.0], 0.0).unwrap(), -10.0);
    assert_eq!(linear_quantile(&[-10.0, 10.0], 1.0).unwrap(), 10.0);
    assert!((linear_quantile(&[-10.0, 10.0], 0.25).unwrap() + 5.0).abs() < 1e-12);
    // Out-of-range levels are errors, not clamps.
    assert!(linear_quantile(&[1.0, 2.0], -1e-16).is_err());
    assert!(linear_quantile(&[1.0, 2.0], 1.0 + 1e-9).is_err());

    // Every higher-level estimator refuses samples below its documented
    // minimum rather than reporting a tail read off 1-2 points.
    for n in [0usize, 1, 2, 5] {
        let pnl = vec![-1.0; n];
        assert!(historical_var(&pnl, 0.01).is_err(), "historical_var n={n}");
        assert!(expected_shortfall(&pnl, 0.01).is_err(), "ES n={n}");
        assert!(
            eq_var_engine::historical::age_weighted_var(&pnl, 0.01, 0.98).is_err(),
            "BRW n={n}"
        );
        assert!(
            eq_var_engine::historical::filtered_historical_var(&pnl, 0.01, 0.94).is_err(),
            "FHS n={n}"
        );
    }
    // A single-column, two-row return panel is the smallest legal input to
    // the sample covariance (ddof = 1 needs 2 rows).
    let one_row = Matrix::from_vec(1, 1, vec![0.01]).unwrap();
    assert!(eq_var_engine::matrix::sample_covariance(&one_row).is_err());
    let two_rows = Matrix::from_vec(2, 1, vec![0.01, -0.01]).unwrap();
    let cov = eq_var_engine::matrix::sample_covariance(&two_rows).unwrap();
    assert!((cov.get(0, 0) - 2.0e-4).abs() < 1e-18); // ((0.02)^2)/2 * ... hand check
    // A one-element exposure vector against that covariance is a legal,
    // fully specified single-asset book.
    let var = parametric_var(&[1.0e6], &cov, 0.01, TailModel::Normal).unwrap();
    assert!(var > 0.0 && var.is_finite());
}
