//! Cross-language golden tests against the Python reference engine.
//!
//! # Provenance
//!
//! Every constant below was produced by the Python reference package
//! `python/fx/03-var-es-engine` (`fx_var`), running:
//!
//! ```text
//! cd /home/claude/quant-portfolio/python/fx/03-var-es-engine
//! PYTHONPATH=src python3 <golden generator, see docs/VALIDATION.md>
//! ```
//!
//! on 2026-08-18 with numpy/scipy doubles, printed via `repr()` (17
//! significant digits). The same script and fixture were used to generate
//! the C++ engine's `tests/test_golden_python.cpp` (independently
//! re-confirmed against a live Python run for this crate). The three cases
//! are fully deterministic (no RNG):
//!
//! * **CASE A** — book revaluation + plain/BRW historical VaR/ES on a
//!   sinusoidal synthetic history reproduced bit-for-bit here;
//! * **CASE B** — parametric closed-form normal / Student-t VaR & ES from
//!   fixed exposures and covariance (`fx_var.parametric_var.var_covar`);
//! * **CASE C** — Kupiec / Christoffersen / Basel statistics
//!   (`fx_var.backtesting`).
//!
//! Tolerances: the only cross-language differences are libm sin/cos/exp
//! rounding (~1 ulp per call) and the special-function implementations
//! (SciPy Cephes vs this crate's), so P&L-scale figures agree to ~1e-6
//! absolute / ~1e-8 relative and probability-scale figures to ~1e-9
//! absolute — matching the C++ engine's documented tolerances.

use fx_var_engine::prelude::*;

fn golden_market() -> Market {
    Market::new([("EUR", 1.10), ("JPY", 0.0090), ("GBP", 1.27)], [("USD", 0.050), ("EUR", 0.030), ("JPY", 0.001)])
        .unwrap()
}

fn golden_book() -> Book {
    Book::new(
        vec![
            Position::Spot(SpotPosition::new("EURUSD", 10_000_000.0, None)), // long 10m EUR at market
            Position::Forward(ForwardPosition::new("USDJPY", 5_000_000.0, 0.5, None)), // ATM CIP forward
            Position::Spot(SpotPosition::new("EURJPY", -3_000_000.0, None)), // short 3m EUR cross
        ],
        "USD",
    )
    .unwrap()
}

/// Deterministic history: factors sorted as the book enumerates them
/// (FX:EUR, FX:JPY, IR:JPY, IR:USD), `r[t][j] = s_j (sin(0.1 t + j) + 0.5
/// cos(0.05 t (j+1)))` — identical formula in the Python generator.
fn golden_returns() -> ReturnsMatrix {
    let factors = vec!["FX:EUR".to_string(), "FX:JPY".to_string(), "IR:JPY".to_string(), "IR:USD".to_string()];
    let scales = [0.006, 0.007, 0.0004, 0.0005];
    let n = 300;
    let mut data = Matrix::zeros(n, 4);
    for t in 0..n {
        for j in 0..4 {
            let td = t as f64;
            let jd = j as f64;
            data.set(t, j, scales[j] * ((0.1 * td + jd).sin() + 0.5 * (0.05 * td * (jd + 1.0)).cos()));
        }
    }
    ReturnsMatrix::new(factors, data).unwrap()
}

#[test]
fn case_a_book_pnl_single_scenario() {
    // Python: book.pnl(market, returns.iloc[17]) = 58177.37489810074
    let m = golden_market();
    let cb = CompiledBook::new(&golden_book(), &m).unwrap();
    let rets = golden_returns();
    let got = cb.pnl(rets.data.row(17)).unwrap();
    assert!((got - 58177.37489810074_f64).abs() < 1e-6, "got {got}");
}

#[test]
fn case_a_plain_historical_var_es() {
    // Python fx_var.historical_var(..., alpha=.99/.975, method="plain"):
    //   var99  = 61919.80890587624   es99  = 62006.12006224847
    //   var975 = 61237.42600889597   es975 = 61777.93608271857
    let m = golden_market();
    let rets = golden_returns();
    let o99 = HistoricalOptions { alpha: 0.99, ..Default::default() };
    let r99 = historical_var(&golden_book(), &m, &rets, &o99).unwrap();
    assert!((r99.var - 61919.80890587624_f64).abs() < 1e-6);
    assert!((r99.es - 62006.12006224847_f64).abs() < 1e-6);
    let o975 = HistoricalOptions { alpha: 0.975, ..Default::default() };
    let r975 = historical_var(&golden_book(), &m, &rets, &o975).unwrap();
    assert!((r975.var - 61237.42600889597_f64).abs() < 1e-6);
    assert!((r975.es - 61777.93608271857_f64).abs() < 1e-6);
}

#[test]
fn case_a_age_weighted_historical_var_es() {
    // Python fx_var.historical_var(..., method="age", decay=0.995):
    //   var99 = 61874.26268531149   es99 = 61977.52496594109
    let m = golden_market();
    let o = HistoricalOptions { alpha: 0.99, method: HsMethod::Age, decay: 0.995, ..Default::default() };
    let r = historical_var(&golden_book(), &m, &golden_returns(), &o).unwrap();
    assert!((r.var - 61874.26268531149_f64).abs() < 1e-6);
    assert!((r.es - 61977.52496594109_f64).abs() < 1e-6);
}

#[test]
fn case_b_parametric_closed_form() {
    // Python fx_var.parametric_var.var_covar on fixed exposures/cov:
    //   w   = {FX:EUR: 11e6, FX:JPY: -4.5e6, IR:USD: -2.4e6}
    //   cov = [[3.6e-5, 1.1e-5, -2e-6],
    //          [1.1e-5, 4.9e-5, -1e-6],
    //          [-2e-6, -1e-6, 2.5e-7]]
    //   normal 99%:  var = 153339.50441962917  es = 175675.6297200285
    //   t(5)   99%:  var = 171803.12389091405  es = 227327.5314974144
    //   normal 99% 10d: var = 484902.0892474838 es = 555535.1192996583
    let w = [11e6, -4.5e6, -2.4e6];
    let cov = Matrix::from_rows(&[
        vec![3.6e-5, 1.1e-5, -2.0e-6],
        vec![1.1e-5, 4.9e-5, -1.0e-6],
        vec![-2.0e-6, -1.0e-6, 2.5e-7],
    ])
    .unwrap();
    let n1 = var_covar(&w, &cov, 0.99, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
    assert!((n1.var - 153339.50441962917_f64).abs() < 1e-8 * n1.var);
    assert!((n1.es - 175675.6297200285_f64).abs() < 1e-8 * n1.es);
    let t5 = var_covar(&w, &cov, 0.99, 1.0, TailDist::StudentT, 5.0, 0.0).unwrap();
    assert!((t5.var - 171803.12389091405_f64).abs() < 1e-8 * t5.var);
    assert!((t5.es - 227327.5314974144_f64).abs() < 1e-8 * t5.es);
    let n10 = var_covar(&w, &cov, 0.99, 10.0, TailDist::Normal, 6.0, 0.0).unwrap();
    assert!((n10.var - 484902.0892474838_f64).abs() < 1e-8 * n10.var);
    assert!((n10.es - 555535.1192996583_f64).abs() < 1e-8 * n10.es);
}

#[test]
fn case_c_backtest_statistics() {
    // Python fx_var.backtesting:
    //   kupiec_pof(8, 250, 0.99)   -> LR = 7.7335507244945205
    //                                 p  = 0.0054204051941277994
    //   christoffersen on the fixed pattern (t % 37 == 5, plus 100,101;
    //   9 exceptions in 250)       -> LR = 1.0063610339314124
    //                                 p  = 0.3157762037622499
    //   basel cum prob: P(X<=5)  = 0.9588168159301514  (yellow, 3.40)
    //                   P(X<=4)  = 0.8921876269036249  (green,  3.00)
    //                   P(X<=10) = 0.999946101370953   (red,    4.00)
    let k = kupiec_pof(8, 250, 0.99).unwrap();
    assert!((k.lr - 7.7335507244945205_f64).abs() < 1e-10);
    assert!((k.p - 0.0054204051941277994_f64).abs() < 1e-11);

    let mut e = vec![0_i32; 250];
    for (t, v) in e.iter_mut().enumerate() {
        if t % 37 == 5 {
            *v = 1;
        }
    }
    e[100] = 1;
    e[101] = 1;
    let count: i32 = e.iter().sum();
    assert_eq!(count, 9); // matches the Python generator's pattern count
    let c = christoffersen_independence(&e).unwrap();
    assert!((c.lr - 1.0063610339314124_f64).abs() < 1e-10);
    assert!((c.p - 0.3157762037622499_f64).abs() < 1e-10);

    let t5 = basel_traffic_light(5, 250, 0.99).unwrap();
    assert!((t5.cumulative_prob - 0.9588168159301514_f64).abs() < 1e-12);
    assert_eq!(t5.zone, Zone::Yellow);
    assert!((t5.multiplier - 3.40_f64).abs() < 1e-12);
    let t4 = basel_traffic_light(4, 250, 0.99).unwrap();
    assert!((t4.cumulative_prob - 0.8921876269036249_f64).abs() < 1e-12);
    assert_eq!(t4.zone, Zone::Green);
    let t10 = basel_traffic_light(10, 250, 0.99).unwrap();
    assert!((t10.cumulative_prob - 0.999946101370953_f64).abs() < 1e-12);
    assert_eq!(t10.zone, Zone::Red);
    assert!((t10.multiplier - 4.0_f64).abs() < 1e-12);
}
