//! Cross-language golden tests against the Python reference implementation.
//!
//! Provenance: every expected constant below was produced by the Python
//! research stack (the conceptual twin of this engine) and independently
//! re-verified against this Rust engine:
//!
//!   cd /home/claude/quant-portfolio/python/equity/03-var-es-engine
//!   PYTHONPATH=src python3 /tmp/gen_golden.py     (numpy, scipy)
//!
//! regenerated on 2026-08-18 from `eq_var` (historical_var / parametric_var /
//! expected_shortfall / backtesting modules). The values are identical to
//! those committed in `cpp/equity-var-engine/tests/test_cross_language.cpp`
//! (same Python invocation, same inputs) — all three engines (Python, C++,
//! Rust) agree to the stated tolerance. Inputs are closed-form (sin/cos
//! based) deterministic series so every language regenerates them
//! independently; agreement is required to 1e-9 relative (the residual is
//! libm-vs-numpy ulp noise in sin/cos, far below any risk tolerance).

use eq_var_engine::prelude::*;

/// Relative tolerance for cross-language agreement.
fn expect_rel(actual: f64, expected: f64, what: &str) {
    let tol = 1e-9 * expected.abs() + 1e-12;
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: actual={actual} expected={expected} (tol={tol})"
    );
}

/// Case A input: `pnl[t] = 100 sin(3t + 1) + 0.5 t cos(t)`, `t = 0..99`.
fn case_a_pnl() -> Vec<f64> {
    (0..100)
        .map(|t| {
            let td = t as f64;
            100.0 * (3.0 * td + 1.0).sin() + 0.5 * td * td.cos()
        })
        .collect()
}

/// Case C input: `r[t, j] = 0.01 sin(t + j) + 0.005 cos(2t - j)`,
/// `t = 0..59`, `j = 0..2`.
fn case_c_returns() -> Matrix {
    let mut r = Matrix::zeros(60, 3);
    for t in 0..60 {
        for j in 0..3 {
            let (td, jd) = (t as f64, j as f64);
            let v = 0.01 * (td + jd).sin() + 0.005 * (2.0 * td - jd).cos();
            r.set(t, j, v);
        }
    }
    r
}

#[test]
fn case_a_historical_family() {
    let pnl = case_a_pnl();
    expect_rel(historical_var(&pnl, 0.01).unwrap(), 1.224129222375264e+02, "A hist VaR 1%");
    expect_rel(historical_var(&pnl, 0.05).unwrap(), 1.045522835374927e+02, "A hist VaR 5%");
    expect_rel(expected_shortfall(&pnl, 0.05).unwrap(), 1.226373207703405e+02, "A ES 5%");
    expect_rel(expected_shortfall(&pnl, 0.01).unwrap(), 1.417568107549531e+02, "A ES 1%");
    expect_rel(
        age_weighted_var(&pnl, 0.05, 0.98).unwrap(),
        1.082348601407293e+02,
        "A BRW VaR 5%",
    );
    expect_rel(
        filtered_historical_var(&pnl, 0.05, 0.94).unwrap(),
        1.093777910164513e+02,
        "A FHS VaR 5%",
    );
}

#[test]
fn case_a_brw_weights() {
    let w = brw_weights(5, 0.98).unwrap();
    let expected = [
        1.920016255921656e-01,
        1.959200261144547e-01,
        1.999183939943416e-01,
        2.039983612187159e-01,
        2.081615930803223e-01,
    ];
    assert_eq!(w.len(), expected.len());
    for (a, e) in w.iter().zip(expected.iter()) {
        expect_rel(*a, *e, "A brw weight");
    }
}

#[test]
fn case_b_parametric_family() {
    // w = [1e6, -5e5, 2e5]; vols = [1%, 1.5%, 2%];
    // corr = [[1, .5, .25], [.5, 1, .3], [.25, .3, 1]].
    let w = [1.0e6, -5.0e5, 2.0e5];
    let vols = [0.01, 0.015, 0.02];
    let corr = Matrix::from_vec(3, 3, vec![1.0, 0.5, 0.25, 0.5, 1.0, 0.3, 0.25, 0.3, 1.0]).unwrap();
    let cov = covariance_from_vols(&vols, &corr).unwrap();
    let sigma = portfolio_sigma(&w, &cov).unwrap();
    expect_rel(sigma, 9.962429422585637e+03, "B sigma");
    expect_rel(
        parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap(),
        2.317607650751402e+04,
        "B VaR normal 1%",
    );
    expect_rel(
        parametric_var(&w, &cov, 0.01, TailModel::StudentT { df: 6.0 }).unwrap(),
        2.556337478743866e+04,
        "B VaR t6 1%",
    );
    expect_rel(
        parametric_var_full(&w, &cov, 0.01, TailModel::Normal, 0.0, 10).unwrap(),
        7.328918899006478e+04,
        "B VaR normal 1% 10d",
    );
    expect_rel(normal_es(sigma, 0.025, 0.0).unwrap(), 2.329019532123021e+04, "B ES normal 2.5%");
    expect_rel(
        student_t_es(sigma, 0.025, 6.0, 0.0).unwrap(),
        2.648647588170394e+04,
        "B ES t6 2.5%",
    );
    expect_rel(
        cornish_fisher_var(sigma, 0.01, -0.5, 1.0, 0.0, true).unwrap(),
        2.823062626871169e+04,
        "B CF VaR 1%",
    );
}

#[test]
fn case_c_covariance_estimators() {
    let r = case_c_returns();
    let s = sample_covariance(&r).unwrap();
    let e = ewma_covariance(&r, 0.94).unwrap();
    expect_rel(s.get(0, 0), 6.197010057346647e-05, "C sample cov 00");
    expect_rel(s.get(0, 1), 3.367477490492125e-05, "C sample cov 01");
    expect_rel(s.get(1, 2), 3.643904950084309e-05, "C sample cov 12");
    expect_rel(e.get(0, 0), 6.085653794066718e-05, "C ewma cov 00");
    expect_rel(e.get(0, 1), 3.159039791408869e-05, "C ewma cov 01");
    expect_rel(e.get(2, 2), 6.695_548_359_317_17e-5, "C ewma cov 22");

    let w = [2.0e6, -1.0e6, 5.0e5];
    expect_rel(
        parametric_var(&w, &s, 0.01, TailModel::Normal).unwrap(),
        2.403056180968714e+04,
        "C VaR normal 1%",
    );
    expect_rel(
        parametric_var(&w, &s, 0.05, TailModel::StudentT { df: 8.0 }).unwrap(),
        1.663517215790393e+04,
        "C VaR t8 5%",
    );
}

#[test]
fn case_c_backtest_statistics() {
    let k7 = kupiec_pof(250, 7, 0.01).unwrap();
    expect_rel(k7.lr, 5.496990447792683e+00, "C kupiec LR (250, 7)");
    expect_rel(k7.pvalue, 1.904923089052653e-02, "C kupiec p (250, 7)");
    let k0 = kupiec_pof(250, 0, 0.01).unwrap();
    expect_rel(k0.lr, 5.025167926750726e+00, "C kupiec LR (250, 0)");
    expect_rel(k0.pvalue, 2.498150305344973e-02, "C kupiec p (250, 0)");

    // Exception pattern: t % 40 == 0, plus days 100 and 101 (9 exceptions).
    let mut ex = vec![0u8; 250];
    let mut t = 0;
    while t < 250 {
        ex[t] = 1;
        t += 40;
    }
    ex[100] = 1;
    ex[101] = 1;
    let count: i32 = ex.iter().map(|&e| e as i32).sum();
    assert_eq!(count, 9);

    let ind = christoffersen_independence(&ex).unwrap();
    expect_rel(ind.lr, 1.189356349111719e+00, "C christoffersen ind LR");
    expect_rel(ind.pvalue, 2.754594267438438e-01, "C christoffersen ind p");
    let cc = christoffersen_cc(&ex, 0.01).unwrap();
    expect_rel(cc.lr, 1.141838698170947e+01, "C christoffersen cc LR");
    expect_rel(cc.pvalue, 3.315345323267836e-03, "C christoffersen cc p");
}
