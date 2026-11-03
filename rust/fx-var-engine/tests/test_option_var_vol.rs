//! Round 4: vol scenarios in the FX option VaR drivers. `"VOL:CCY"` columns
//! sit beside `"FX:CCY"` on the one named factor axis. Golden values from
//! `cpp/equity-var-engine/tools/gen_option_var_golden.py` (Python reference
//! `fx_var.revaluation.full_revaluation_pnl(..., vol_mode=...)` + the S2
//! order statistic); the MC check is a closed-form quantile.

use fx_var_engine::book::{Book, CashPosition, Market, Position};
use fx_var_engine::historical::{
    HistoricalOptions, historical_option_var, historical_option_var_vol,
};
use fx_var_engine::matrix::Matrix;
use fx_var_engine::monte_carlo::{MonteCarloOptions, monte_carlo_option_var_vol};
use fx_var_engine::option_reval::{
    OptionKind, OptionLeg, VolMode, VolScenarios, layout_option_book, option_combined_factors,
};
use fx_var_engine::returns::{FactorCov, ReturnsMatrix};

const N: usize = 250;

#[allow(clippy::too_many_arguments)]
fn leg(q: f64, kind: OptionKind, s: f64, k: f64, t: f64, rf: f64, v: f64, f: usize) -> OptionLeg {
    OptionLeg {
        quantity: q,
        kind,
        spot: s,
        strike: k,
        tau: t,
        rate: 0.045,
        carry: rf,
        vol: v,
        factor: f,
    }
}

fn legs() -> Vec<OptionLeg> {
    vec![
        leg(1.0e6, OptionKind::Call, 1.10, 1.12, 0.25, 0.03, 0.08, 0),
        leg(-2.0e6, OptionKind::Put, 0.75, 0.74, 0.5, 0.04, 0.10, 1),
    ]
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn market() -> Market {
    Market::new([("EUR", 1.10), ("AUD", 0.75)], [("USD", 0.045)]).unwrap()
}

fn spot_panel() -> ReturnsMatrix {
    let mut m = Matrix::zeros(N, 2);
    for i in 0..N {
        let t = i as f64;
        m.set(i, 1, 0.02 * (1.3 * t + 0.4).sin());
        m.set(i, 0, 0.015 * (0.7 * t).cos() + 0.005 * (2.9 * t).sin());
    }
    ReturnsMatrix::new(names(&["FX:AUD", "FX:EUR"]), m).unwrap()
}

fn vol_panel(zero_vol: bool) -> ReturnsMatrix {
    let p = spot_panel();
    let mut m = Matrix::zeros(N, 4);
    for i in 0..N {
        let t = i as f64;
        m.set(i, 1, p.data.get(i, 0));
        m.set(i, 2, p.data.get(i, 1));
        if !zero_vol {
            m.set(i, 3, 0.03 * (0.9 * t + 0.2).sin());
            m.set(i, 0, 0.04 * (1.7 * t).cos());
        }
    }
    ReturnsMatrix::new(names(&["VOL:AUD", "FX:AUD", "FX:EUR", "VOL:EUR"]), m).unwrap()
}

fn plain() -> HistoricalOptions {
    HistoricalOptions {
        alpha: 0.01,
        warn_pegs: false,
        ..Default::default()
    }
}

fn rel(got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * want.abs(),
        "got {got}, want {want}"
    );
}

fn cash_eur() -> Book {
    Book::new(vec![Position::Cash(CashPosition::new("EUR", 1.0e6))], "USD").unwrap()
}

fn vs(mode: VolMode) -> VolScenarios {
    VolScenarios {
        mode,
        skews: vec![-0.05, 0.08],
    }
}

#[test]
fn historical_matches_python_all_modes() {
    let want = [
        (
            VolMode::StickyStrike,
            47004.75041376564,
            50058.34738396022,
            5845.693777989148,
        ),
        (
            VolMode::StickyDelta,
            47689.026521634754,
            50653.063630506396,
            6400.832090265918,
        ),
        (
            VolMode::HistoricalAtm,
            37267.63851698485,
            38043.48960210825,
            18537.15494683804,
        ),
    ];
    for (mode, var, es, pnl0) in want {
        let r = historical_option_var_vol(
            &cash_eur(),
            &market(),
            &legs(),
            &names(&["FX:EUR", "FX:AUD"]),
            &vol_panel(false),
            &plain(),
            1.0 / 252.0,
            &vs(mode),
        )
        .unwrap();
        rel(r.pnl[0], pnl0, 1e-10);
        rel(r.var, var, 1e-11);
        rel(r.es, es, 1e-11);
    }
}

#[test]
fn zero_vol_columns_sticky_strike_is_the_old_path() {
    let n = names(&["FX:EUR", "FX:AUD"]);
    let a = historical_option_var_vol(
        &cash_eur(),
        &market(),
        &legs(),
        &n,
        &vol_panel(true),
        &plain(),
        1.0 / 252.0,
        &VolScenarios::default(),
    )
    .unwrap();
    let b = historical_option_var(
        &cash_eur(),
        &market(),
        &legs(),
        &n,
        &spot_panel(),
        &plain(),
        1.0 / 252.0,
    )
    .unwrap();
    assert_eq!((a.var, a.es), (b.var, b.es));
    assert_eq!(a.pnl, b.pnl);
}

#[test]
fn layout_puts_vol_factors_after_spot() {
    let layout = layout_option_book(
        &Book::empty(),
        &market(),
        &legs(),
        &names(&["FX:EUR", "FX:AUD"]),
    )
    .unwrap();
    assert_eq!(
        option_combined_factors(&layout),
        names(&["FX:EUR", "FX:AUD", "VOL:EUR", "VOL:AUD"])
    );
}

#[test]
fn monte_carlo_matches_closed_form_vol_quantile() {
    let cov = FactorCov {
        factors: names(&["FX:EUR", "VOL:EUR"]),
        cov: Matrix::from_rows(&[vec![1e-12, 0.0], vec![0.0, 0.0025]]).unwrap(),
    };
    let mo = MonteCarloOptions {
        n_scenarios: 200_000,
        seed: 5,
        ..Default::default()
    };
    let one = vec![leg(
        1.0e6,
        OptionKind::Call,
        1.10,
        1.10,
        0.25,
        0.03,
        0.08,
        0,
    )];
    let r = monte_carlo_option_var_vol(
        &Book::empty(),
        &market(),
        &one,
        &names(&["FX:EUR"]),
        &cov,
        &mo,
        0.0,
        &VolScenarios {
            mode: VolMode::HistoricalAtm,
            skews: vec![],
        },
    )
    .unwrap();
    assert!(
        (r.var - 1899.770347803309).abs() <= 4.0 * r.se_var,
        "var {}",
        r.var
    );
    rel(r.es, 2156.5055802076868, 0.02);
}

#[test]
fn rejects_bad_input() {
    let n = names(&["FX:EUR", "FX:AUD"]);
    let d = VolScenarios::default();
    // A spot-only panel has no VOL: columns.
    assert!(
        historical_option_var_vol(
            &cash_eur(),
            &market(),
            &legs(),
            &n,
            &spot_panel(),
            &plain(),
            0.0,
            &d
        )
        .is_err()
    );
    assert!(
        historical_option_var_vol(
            &cash_eur(),
            &market(),
            &[],
            &[],
            &vol_panel(false),
            &plain(),
            0.0,
            &d
        )
        .is_err()
    );
    let bad = VolScenarios {
        mode: VolMode::StickyDelta,
        skews: vec![1.0],
    };
    assert!(
        historical_option_var_vol(
            &cash_eur(),
            &market(),
            &legs(),
            &n,
            &vol_panel(false),
            &plain(),
            0.0,
            &bad
        )
        .is_err()
    );
    let cov = FactorCov {
        factors: names(&["FX:AUD", "FX:EUR"]),
        cov: Matrix::from_rows(&[vec![1e-4, 0.0], vec![0.0, 1e-4]]).unwrap(),
    };
    assert!(
        monte_carlo_option_var_vol(
            &cash_eur(),
            &market(),
            &legs(),
            &n,
            &cov,
            &MonteCarloOptions::default(),
            0.0,
            &d
        )
        .is_err()
    );
}
