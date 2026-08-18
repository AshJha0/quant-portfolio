//! Real-life scenarios and edge cases (CONVENTIONS.md documentation
//! contract item 6): hand-exact empirical quantiles, peg blindness vs the
//! jump-mixture Monte Carlo overlay, T -> 0 / zero / negative rates,
//! singular pegged covariances, missing/NaN data, and crisis stress
//! replays end to end.

use fx_var_engine::prelude::*;
use std::collections::HashMap;

fn market_eur_jpy_hkd() -> Market {
    Market::new([("EUR", 1.10), ("JPY", 0.0090), ("HKD", 0.1282)], [("USD", 0.050), ("EUR", 0.030), ("JPY", 0.001)])
        .unwrap()
}

// -----------------------------------------------------------------------
// Hand-exact empirical VaR / ES (mirrors C++ test_expected_shortfall.cpp)
// -----------------------------------------------------------------------

#[test]
fn hand_exact_empirical_var_quantiles() {
    // 10 P&Ls; losses sorted desc: 50, 40, 30, 20, 10, 0, -5, -15, -25, -35.
    let pnl = [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 5.0, 15.0, 25.0, 35.0];
    assert_eq!(empirical_var(&pnl, 0.90, None).unwrap(), 50.0); // m=ceil(1)=worst
    assert_eq!(empirical_var(&pnl, 0.80, None).unwrap(), 40.0); // m=2
    assert_eq!(empirical_var(&pnl, 0.75, None).unwrap(), 30.0); // m=2.5->3rd
    assert_eq!(empirical_var(&pnl, 0.99, None).unwrap(), 50.0); // only 10 obs
}

#[test]
fn hand_exact_acerbi_tasche_es() {
    let pnl = [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 5.0, 15.0, 25.0, 35.0];
    assert_eq!(empirical_es(&pnl, 0.80, None).unwrap(), 45.0); // 2 worst losses averaged
    // alpha=0.75: tail mass 0.25 = 0.1+0.1+0.05 fractional atom share.
    let es = empirical_es(&pnl, 0.75, None).unwrap();
    assert!((es - 42.0).abs() < 1e-9);
    for a in [0.5, 0.75, 0.9, 0.99] {
        let (v, e) = empirical_var_es(&pnl, a, None).unwrap();
        assert!(e >= v);
    }
}

#[test]
fn weighted_empirical_var_es() {
    // BRW-style weighting emphasising specific scenarios.
    let pnl = [-100.0, -10.0, 0.0, -50.0];
    let w = [0.05, 0.15, 0.30, 0.50];
    // Losses desc: 100 (w .05), 50 (w .50), 10 (w .15), 0 (w .30).
    assert!((empirical_var(&pnl, 0.90, Some(&w)).unwrap() - 50.0).abs() < 1e-12);
    assert!((empirical_es(&pnl, 0.90, Some(&w)).unwrap() - 75.0).abs() < 1e-12);
}

#[test]
fn closed_form_normal_textbook_numbers() {
    // sigma=1, alpha=0.99: VaR = 2.3263478740, ES = 2.6652142306 (classic).
    assert!((normal_var(1.0, 0.99, 0.0).unwrap() - 2.326_347_874_040_840_8).abs() < 1e-9);
    assert!((normal_es(1.0, 0.99, 0.0).unwrap() - 2.665_214_220_345_808).abs() < 1e-9);
    assert!(normal_es(1.0, 0.95, 0.0).unwrap() > normal_var(1.0, 0.95, 0.0).unwrap());
}

#[test]
fn student_t_converges_to_normal_as_df_grows() {
    let sigma = 1.0;
    assert!(student_t_var(sigma, 0.99, 4.0, 0.0).unwrap() > normal_var(sigma, 0.99, 0.0).unwrap());
    let far = student_t_var(sigma, 0.99, 1.0e7, 0.0).unwrap();
    let normal = normal_var(sigma, 0.99, 0.0).unwrap();
    assert!((far - normal).abs() < 1e-4);
}

// -----------------------------------------------------------------------
// T -> 0 / zero / negative rates
// -----------------------------------------------------------------------

#[test]
fn forward_at_zero_expiry_collapses_to_spot_difference() {
    // A forward with expiry = 0 is economically a spot trade struck at K:
    // no discounting, so zero rate-factor *sensitivity* even though the
    // book's factor set (driven by the pair, not the expiry) still lists
    // IR:EUR / IR:USD.
    let m = market_eur_jpy_hkd();
    let book =
        Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 5e6, 0.0, Some(1.05)))], "USD").unwrap();
    let cb = CompiledBook::new(&book, &m).unwrap();
    let expect = 5e6 * 1.10 - 5e6 * 1.05; // undiscounted spot legs
    assert!((cb.value0_usd() - expect).abs() < 1e-6 * expect.abs());
    let w = cb.linear_exposures(1e-6).unwrap();
    for (f, wi) in cb.factors().iter().zip(&w) {
        if f.starts_with("IR:") {
            assert!(wi.abs() < 1e-3, "expected zero IR sensitivity at T=0, got {wi} for {f}");
        }
    }
}

#[test]
fn zero_and_negative_rates_are_valid_inputs() {
    // A negative-rate currency (e.g. EUR/JPY in the 2015-2021 era) is a
    // legitimate, non-exceptional market input.
    let m = Market::new([("EUR", 1.10)], [("USD", 0.0), ("EUR", -0.005)]).unwrap();
    let book = Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 5e6, 1.0, None))], "USD").unwrap();
    let cb = CompiledBook::new(&book, &m).unwrap();
    // ATM CIP forward at zero USD rate and negative EUR rate: still prices
    // and revalues cleanly with zero initial value.
    assert!(cb.value0_usd().abs() < 1e-8 * 5e6);
}

// -----------------------------------------------------------------------
// Peg blindness: HS misses it, jump-mixture MC sees it
// -----------------------------------------------------------------------

#[test]
fn peg_break_jump_produces_loss_historical_simulation_misses() {
    // A pegged HKD short-USD book: the historical window shows ~zero vol,
    // so HS VaR is negligible - but the jump-mixture MC with a peg-break
    // overlay reports a material loss. This is the engine's
    // peg-blindness story end to end.
    let m = market_eur_jpy_hkd();
    let book = Book::new(vec![Position::Spot(SpotPosition::new("USDHKD", 100e6, None))], "USD").unwrap(); // long USD vs HKD

    let mut data = Matrix::zeros(250, 1);
    for i in 0..250 {
        data.set(i, 0, 1e-4 * (0.5 * i as f64).sin()); // band-bound noise
    }
    let rets = ReturnsMatrix::new(vec!["FX:HKD".to_string()], data).unwrap();
    let hs = historical_var(&book, &m, &rets, &HistoricalOptions::default()).unwrap();
    assert_eq!(hs.flagged_peg_factors.len(), 1); // engine warns

    let cov = sample_cov(&rets).unwrap();
    let mut jumps = JumpSpec { prob: 0.02, ..Default::default() }; // revaluation event
    jumps.mean.insert("FX:HKD".to_string(), 0.10); // HKD +10% (log) vs USD
    let opts = MonteCarloOptions { n_scenarios: 50_000, seed: 5, dist: McDist::Jump, jumps, ..Default::default() };
    let mc = monte_carlo_var(&book, &m, &cov, &opts).unwrap();
    // Short 100m USD of HKD: a +10% HKD reval loses ~10m USD.
    assert!(hs.var < 0.1e6); // HS blind: < 0.1% of notional
    assert!(mc.var > 5e6); // jump MC sees the break
    assert!(mc.var > 20.0 * hs.var);
}

// -----------------------------------------------------------------------
// Singular / pegged covariance
// -----------------------------------------------------------------------

#[test]
fn singular_pegged_covariance_runs_with_jitter_end_to_end() {
    let m = Market::new([("HKD", 0.1282), ("AED", 0.2723)], std::iter::empty::<(&str, f64)>()).unwrap();
    let book = Book::new(
        vec![
            Position::Spot(SpotPosition::new("USDHKD", 10e6, None)),
            Position::Spot(SpotPosition::new("USDAED", 5e6, None)),
        ],
        "USD",
    )
    .unwrap();
    let cov = FactorCov {
        factors: vec!["FX:AED".to_string(), "FX:HKD".to_string()],
        cov: Matrix::from_rows(&[vec![6.25e-2, 3.125e-2], vec![3.125e-2, 1.5625e-2]]).unwrap(),
    };
    let opts = MonteCarloOptions { n_scenarios: 1000, ..Default::default() };
    let res = monte_carlo_var(&book, &m, &cov, &opts).unwrap();
    assert!(!res.cholesky_warning.is_empty());
    assert!(res.var >= 0.0);
}

// -----------------------------------------------------------------------
// Missing / NaN data
// -----------------------------------------------------------------------

#[test]
fn nan_history_is_refused_not_dropped() {
    let m = market_eur_jpy_hkd();
    let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 1e6, None))], "USD").unwrap();
    let mut data = Matrix::zeros(120, 1);
    data.set(5, 0, f64::NAN);
    let rets = ReturnsMatrix::new(vec!["FX:EUR".to_string()], data).unwrap();
    assert!(historical_var(&book, &m, &rets, &HistoricalOptions::default()).is_err());
}

#[test]
fn missing_factor_column_is_reported_not_silently_zeroed() {
    let m = market_eur_jpy_hkd();
    let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 1e6, None))], "USD").unwrap();
    let mut data = Matrix::zeros(120, 1);
    for t in 0..120 {
        data.set(t, 0, 0.001 * (t as f64).sin());
    }
    // Wrong factor column: "FX:JPY" instead of the required "FX:EUR".
    let rets = ReturnsMatrix::new(vec!["FX:JPY".to_string()], data).unwrap();
    let err = historical_var(&book, &m, &rets, &HistoricalOptions::default()).unwrap_err();
    assert!(matches!(err, FxVarError::Invalid(_)));
}

// -----------------------------------------------------------------------
// Crisis regime replay (canned stress scenarios)
// -----------------------------------------------------------------------

#[test]
fn multi_currency_book_survives_brexit_and_chf_replay() {
    let m = Market::new(
        [("EUR", 1.10), ("GBP", 1.27), ("JPY", 0.0090), ("CHF", 1.12)],
        [("USD", 0.050), ("EUR", 0.030), ("GBP", 0.045), ("JPY", 0.001), ("CHF", -0.005)],
    )
    .unwrap();
    let book = Book::new(
        vec![
            Position::Spot(SpotPosition::new("GBPUSD", 8e6, None)),
            Position::Spot(SpotPosition::new("USDCHF", 5e6, None)),
            Position::Forward(ForwardPosition::new("EURUSD", 3e6, 0.5, None)),
        ],
        "USD",
    )
    .unwrap();
    let scenarios = historical_scenarios();
    let rows = run_stress(&book, &m, &scenarios).unwrap();
    assert_eq!(rows.len(), scenarios.len());
    // Every scenario must be a full revaluation (finite P&L), and the
    // report must be sorted worst-first.
    for r in &rows {
        assert!(r.pnl.is_finite());
    }
    for w in rows.windows(2) {
        assert!(w[0].pnl <= w[1].pnl);
    }
}

// -----------------------------------------------------------------------
// Cross-currency correlation regime dependence (documented in
// docs/METHODOLOGY.md): sample vs EWMA covariance diverge sharply right
// after a volatility regime shift.
// -----------------------------------------------------------------------

#[test]
fn ewma_covariance_reacts_faster_than_sample_after_regime_shift() {
    let n = 150;
    let mut data = Matrix::zeros(n, 1);
    for t in 0..n {
        // Calm regime for the first 120 days, then a vol spike.
        let vol = if t < 120 { 0.001 } else { 0.02 };
        data.set(t, 0, vol * (((t as f64) * 13.0).sin()));
    }
    let rets = ReturnsMatrix::new(vec!["FX:EUR".to_string()], data).unwrap();
    let sample = sample_cov(&rets).unwrap();
    let ewma = ewma_cov(&rets, 0.90).unwrap(); // fast decay to feel the shift
    assert!(ewma.cov.get(0, 0) > sample.cov.get(0, 0));
}

// -----------------------------------------------------------------------
// USD-pivot invariant: shocking FX:USD is always rejected.
// -----------------------------------------------------------------------

#[test]
fn shocking_fx_usd_is_always_rejected() {
    let m = market_eur_jpy_hkd();
    let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 1e6, None))], "USD").unwrap();
    let cb = CompiledBook::new(&book, &m).unwrap();
    let mut shocks = HashMap::new();
    shocks.insert("FX:USD".to_string(), 0.05);
    assert!(cb.pnl_map(&shocks).is_err());
}
