//! Option VaR: full revaluation wired into historical simulation and Monte
//! Carlo. Golden values from `cpp/equity-var-engine/tools/gen_option_var_golden.py`
//! (Python reference `eq_var.revaluation` + S2 order statistic); the MC check
//! is against the closed-form quantile of a monotone one-factor book.

use eq_var_engine::matrix::Matrix;
use eq_var_engine::monte_carlo::monte_carlo_run;
use eq_var_engine::option_reval::*;
use eq_var_engine::tail::empirical_var_es;
use eq_var_engine::TailModel;

const N: usize = 250;

fn legs() -> Vec<OptionLeg> {
    vec![
        OptionLeg {
            quantity: 10.0,
            kind: OptionKind::Call,
            spot: 100.0,
            strike: 105.0,
            tau: 0.5,
            rate: 0.03,
            carry: 0.01,
            vol: 0.25,
            factor: 0,
        },
        OptionLeg {
            quantity: -5.0,
            kind: OptionKind::Put,
            spot: 50.0,
            strike: 48.0,
            tau: 0.25,
            rate: 0.03,
            carry: 0.0,
            vol: 0.30,
            factor: 1,
        },
    ]
}

const W: [f64; 2] = [1000.0, -500.0];

fn panel() -> Matrix {
    let mut d = Vec::with_capacity(2 * N);
    for i in 0..N {
        let t = i as f64;
        d.push(0.02 * (1.3 * t + 0.4).sin());
        d.push(0.015 * (0.7 * t).cos() + 0.005 * (2.9 * t).sin());
    }
    Matrix::from_vec(N, 2, d).unwrap()
}

fn rel(got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * want.abs(),
        "got {got}, want {want}"
    );
}

fn atm_call() -> Vec<OptionLeg> {
    vec![OptionLeg {
        quantity: 10.0,
        kind: OptionKind::Call,
        spot: 100.0,
        strike: 100.0,
        tau: 0.25,
        rate: 0.02,
        carry: 0.0,
        vol: 0.2,
        factor: 0,
    }]
}

#[test]
fn book_pnl_matches_python() {
    let p = option_book_pnl(&legs(), &W, &panel(), 0.0).unwrap();
    rel(p[0], 5.077728139045078, 1e-11);
    rel(p[7], -5.395781243949972, 1e-10);
}

#[test]
fn historical_matches_python() {
    let (v, e) = historical_option_var_es(&legs(), &[], &panel(), 0.01, 0.0).unwrap();
    rel(v, 9.667169426304056, 1e-11);
    rel(e, 9.719324275447953, 1e-11);
    let (v, e) = historical_option_var_es(&legs(), &W, &panel(), 0.01, 0.0).unwrap();
    rel(v, 33.444115781270625, 1e-11);
    rel(e, 34.649898183703485, 1e-11);
    let (v, e) = historical_option_var_es(&legs(), &W, &panel(), 0.01, 1.0 / 252.0).unwrap();
    rel(v, 33.63830117790198, 1e-11);
    rel(e, 34.84490535004737, 1e-11);
}

#[test]
fn no_legs_is_the_linear_path() {
    let x = panel();
    let lin = x.matvec(&W).unwrap();
    let want = empirical_var_es(&lin, 0.01).unwrap();
    assert_eq!(
        historical_option_var_es(&[], &W, &x, 0.01, 0.0).unwrap(),
        want
    );
    let cov = Matrix::from_vec(2, 2, vec![1e-4, 2e-5, 2e-5, 4e-4]).unwrap();
    let a =
        monte_carlo_option_run(&[], &W, &cov, 0.01, 5000, TailModel::Normal, 7, 50, 0.0).unwrap();
    let b = monte_carlo_run(&W, &cov, 0.01, 5000, TailModel::Normal, 7, 50).unwrap();
    assert_eq!(a, b);
}

#[test]
fn monte_carlo_matches_closed_form_quantile() {
    let cov = Matrix::from_vec(1, 1, vec![1e-4]).unwrap();
    let r = monte_carlo_option_run(
        &atm_call(),
        &[],
        &cov,
        0.01,
        200_000,
        TailModel::Normal,
        11,
        200,
        0.0,
    )
    .unwrap();
    assert!(r.se > 0.0);
    assert!((r.var - 11.352269313193162).abs() <= 4.0 * r.se, "{r:?}");
    rel(r.es, 12.787892115047633, 0.02);
}

#[test]
fn convexity_orders_long_short_and_delta() {
    let lng = atm_call();
    let mut sht = lng.clone();
    sht[0].quantity = -10.0;
    let delta = bsm_greeks(&lng[0]).unwrap().delta * 10.0 * 100.0;
    let cov = Matrix::from_vec(1, 1, vec![4e-4]).unwrap();
    let run = |l: &[OptionLeg]| {
        monte_carlo_option_run(l, &[], &cov, 0.01, 50_000, TailModel::Normal, 3, 50, 0.0)
            .unwrap()
            .var
    };
    let d = monte_carlo_run(&[delta], &cov, 0.01, 50_000, TailModel::Normal, 3, 50)
        .unwrap()
        .var;
    assert!(run(&lng) < d);
    assert!(run(&sht) > d);
}

#[test]
fn rejects_bad_input() {
    let x = panel();
    let short = Matrix::zeros(10, 2);
    assert!(historical_option_var_es(&legs(), &W, &short, 0.01, 0.0).is_err());
    assert!(historical_option_var_es(&legs(), &[1.0], &x, 0.01, 0.0).is_err());
    assert!(historical_option_var_es(&[], &[], &x, 0.01, 0.0).is_err());
    assert!(historical_option_var_es(&legs(), &W, &x, 0.7, 0.0).is_err());
    assert!(historical_option_var_es(&legs(), &W, &x, 0.01, -1.0).is_err());
    let mut bad = legs();
    bad[1].factor = 5;
    assert!(historical_option_var_es(&bad, &W, &x, 0.01, 0.0).is_err());
    let cov = Matrix::zeros(2, 2);
    assert!(monte_carlo_option_run(
        &legs(),
        &[1.0],
        &cov,
        0.01,
        500,
        TailModel::Normal,
        0,
        10,
        0.0
    )
    .is_err());
    assert!(
        monte_carlo_option_run(&legs(), &W, &cov, 0.01, 50, TailModel::Normal, 0, 10, 0.0).is_err()
    );
}
