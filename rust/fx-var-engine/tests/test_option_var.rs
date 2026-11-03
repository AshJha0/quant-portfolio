//! Option VaR: Garman-Kohlhagen full revaluation wired into historical
//! simulation and Monte Carlo. Golden values from
//! `cpp/equity-var-engine/tools/gen_option_var_golden.py` (Python reference
//! `fx_var.revaluation` + the S2 order statistic); the MC check is against
//! the closed-form quantile of a monotone one-factor book.

use fx_var_engine::book::{Book, CashPosition, Market, Position, SpotPosition};
use fx_var_engine::historical::{
    HistoricalOptions, HsMethod, historical_option_var, historical_var,
};
use fx_var_engine::matrix::Matrix;
use fx_var_engine::monte_carlo::{MonteCarloOptions, monte_carlo_option_var, monte_carlo_var};
use fx_var_engine::option_reval::{OptionKind, OptionLeg};
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

fn panel() -> ReturnsMatrix {
    let mut m = Matrix::zeros(N, 2);
    for i in 0..N {
        let t = i as f64;
        m.set(i, 1, 0.02 * (1.3 * t + 0.4).sin());
        m.set(i, 0, 0.015 * (0.7 * t).cos() + 0.005 * (2.9 * t).sin());
    }
    ReturnsMatrix::new(names(&["FX:AUD", "FX:EUR"]), m).unwrap()
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

#[test]
fn historical_options_only_matches_python() {
    let r = historical_option_var(
        &Book::empty(),
        &market(),
        &legs(),
        &names(&["FX:EUR", "FX:AUD"]),
        &panel(),
        &plain(),
        0.0,
    )
    .unwrap();
    rel(r.pnl[0], 11397.749935234513, 1e-11);
    rel(r.pnl[7], 3718.275663457116, 1e-10);
    rel(r.var, 16760.147839431305, 1e-11);
    rel(r.es, 17352.525540976494, 1e-11);
}

#[test]
fn historical_with_linear_book_matches_python() {
    let r = historical_option_var(
        &cash_eur(),
        &market(),
        &legs(),
        &names(&["FX:EUR", "FX:AUD"]),
        &panel(),
        &plain(),
        1.0 / 252.0,
    )
    .unwrap();
    rel(r.pnl[0], 19969.669477855823, 1e-11);
    rel(r.var, 36908.47316233292, 1e-11);
    rel(r.es, 37902.526440186884, 1e-11);
}

#[test]
fn no_options_is_the_linear_path() {
    let book = Book::new(
        vec![Position::Spot(SpotPosition::new("EURUSD", 5e6, None))],
        "USD",
    )
    .unwrap();
    let a = historical_option_var(&book, &market(), &[], &[], &panel(), &plain(), 0.0).unwrap();
    let b = historical_var(&book, &market(), &panel(), &plain()).unwrap();
    assert_eq!((a.var, a.es), (b.var, b.es));
    let cov = FactorCov {
        factors: names(&["FX:AUD", "FX:EUR"]),
        cov: Matrix::from_rows(&[vec![4e-5, 1e-5], vec![1e-5, 3.6e-5]]).unwrap(),
    };
    let mo = MonteCarloOptions {
        n_scenarios: 5000,
        n_boot: 20,
        seed: 9,
        ..Default::default()
    };
    let c = monte_carlo_option_var(&book, &market(), &[], &[], &cov, &mo, 0.0).unwrap();
    let d = monte_carlo_var(&book, &market(), &cov, &mo).unwrap();
    assert_eq!((c.var, c.es, c.se_var), (d.var, d.es, d.se_var));
}

#[test]
fn monte_carlo_matches_closed_form_quantile() {
    let cov = FactorCov {
        factors: names(&["FX:EUR"]),
        cov: Matrix::from_rows(&[vec![0.006 * 0.006]]).unwrap(),
    };
    let mo = MonteCarloOptions {
        n_scenarios: 200_000,
        seed: 11,
        ..Default::default()
    };
    let one = [leg(
        1.0e6,
        OptionKind::Call,
        1.10,
        1.10,
        0.25,
        0.03,
        0.08,
        0,
    )];
    let r = monte_carlo_option_var(
        &Book::empty(),
        &market(),
        &one,
        &names(&["FX:EUR"]),
        &cov,
        &mo,
        0.0,
    )
    .unwrap();
    assert!(r.se_var > 0.0);
    assert!(
        (r.var - 7204.544744658981).abs() <= 4.0 * r.se_var,
        "{}",
        r.var
    );
    rel(r.es, 8054.686967487454, 0.02);
}

#[test]
fn age_and_filtered_methods_run() {
    for m in [HsMethod::Age, HsMethod::Filtered] {
        let o = HistoricalOptions {
            method: m,
            ..plain()
        };
        let r = historical_option_var(
            &cash_eur(),
            &market(),
            &legs(),
            &names(&["FX:EUR", "FX:AUD"]),
            &panel(),
            &o,
            0.0,
        )
        .unwrap();
        assert!(r.var > 0.0 && r.es >= r.var);
        assert_eq!(r.weights.len(), N);
    }
}

#[test]
fn rejects_bad_input() {
    let nm = names(&["FX:EUR", "FX:AUD"]);
    let h10 = HistoricalOptions {
        horizon_days: 10.0,
        ..plain()
    };
    let run = |b: &Book, n: &[String], o: &HistoricalOptions| {
        historical_option_var(b, &market(), &legs(), n, &panel(), o, 0.0)
    };
    assert!(run(&cash_eur(), &nm, &h10).is_err());
    let eur_base = Book::new(vec![Position::Cash(CashPosition::new("AUD", 1.0e6))], "EUR").unwrap();
    assert!(run(&eur_base, &nm, &plain()).is_err());
    assert!(run(&cash_eur(), &names(&["FX:EUR"]), &plain()).is_err());
    assert!(run(&cash_eur(), &names(&["EUR", "AUD"]), &plain()).is_err());
    assert!(run(&cash_eur(), &names(&["FX:EUR", "FX:CHF"]), &plain()).is_err());
}
