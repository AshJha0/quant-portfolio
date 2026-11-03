//! Round 4: vol scenarios in the equity option VaR drivers, on the combined
//! axis (spot block, then one vol column per factor). Golden values from
//! `cpp/equity-var-engine/tools/gen_option_var_golden.py`.

use eq_var_engine::TailModel;
use eq_var_engine::matrix::Matrix;
use eq_var_engine::option_reval::*;

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

const SKEWS: [f64; 2] = [-0.1, -0.2];

fn combined(zero_vol: bool) -> Matrix {
    let x = panel();
    let mut d = Vec::with_capacity(4 * N);
    for i in 0..N {
        let t = i as f64;
        d.push(x.get(i, 0));
        d.push(x.get(i, 1));
        if zero_vol {
            d.extend_from_slice(&[0.0, 0.0]);
        } else {
            d.push(0.03 * (0.9 * t + 0.2).sin());
            d.push(0.04 * (1.7 * t).cos());
        }
    }
    Matrix::from_vec(N, 4, d).unwrap()
}

#[test]
fn historical_matches_python_all_modes() {
    let c = combined(false);
    let want = [
        (VolMode::StickyStrike, 39.22448503697698, 40.96304558554884),
        (VolMode::StickyDelta, 39.85818324867974, 41.62283824144462),
        (VolMode::HistoricalAtm, 34.80398626392455, 35.97429554496573),
    ];
    for (mode, var, es) in want {
        let (v, e) =
            historical_option_var_es_vol(&legs(), &W, &c, 0.01, 0.0, mode, &SKEWS).unwrap();
        rel(v, var, 1e-11);
        rel(e, es, 1e-11);
    }
    let p = option_book_pnl_vol(&legs(), &W, &c, 0.0, VolMode::StickyDelta, &SKEWS).unwrap();
    rel(p[0], 5.032911200617955, 1e-11);
    rel(p[7], -5.154584861517498, 1e-10);
}

#[test]
fn zero_vol_block_sticky_strike_is_the_old_path() {
    let a = historical_option_var_es_vol(
        &legs(),
        &W,
        &combined(true),
        0.01,
        1.0 / 252.0,
        VolMode::StickyStrike,
        &[],
    )
    .unwrap();
    let b = historical_option_var_es(&legs(), &W, &panel(), 0.01, 1.0 / 252.0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn monte_carlo_matches_closed_form_vol_quantile() {
    let cov = Matrix::from_vec(2, 2, vec![1e-12, 0.0, 0.0, 0.0025]).unwrap();
    let r = monte_carlo_option_run_vol(
        &atm_call(),
        &[],
        &cov,
        0.01,
        200_000,
        TailModel::Normal,
        5,
        200,
        0.0,
        VolMode::HistoricalAtm,
        &[],
    )
    .unwrap();
    assert!((r.var - 4.35873555333572).abs() <= 4.0 * r.se, "{}", r.var);
    rel(r.es, 4.948184300200496, 0.02);
}

#[test]
fn rejects_bad_input() {
    let c = combined(false);
    let odd = Matrix::zeros(N, 3);
    assert!(
        historical_option_var_es_vol(&legs(), &W, &odd, 0.01, 0.0, VolMode::StickyStrike, &[])
            .is_err()
    );
    assert!(
        historical_option_var_es_vol(&[], &W, &c, 0.01, 0.0, VolMode::StickyStrike, &[]).is_err()
    );
    assert!(
        historical_option_var_es_vol(
            &legs(),
            &[1.0, 2.0, 3.0, 4.0],
            &c,
            0.01,
            0.0,
            VolMode::StickyStrike,
            &[]
        )
        .is_err()
    );
    assert!(
        historical_option_var_es_vol(&legs(), &W, &c, 0.01, 0.0, VolMode::StickyDelta, &[1.0])
            .is_err()
    );
    let cov3 = Matrix::zeros(3, 3);
    assert!(
        monte_carlo_option_run_vol(
            &legs(),
            &W,
            &cov3,
            0.01,
            5000,
            TailModel::Normal,
            1,
            50,
            0.0,
            VolMode::StickyStrike,
            &[]
        )
        .is_err()
    );
}
