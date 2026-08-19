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

// -----------------------------------------------------------------------
// NaN / Inf rejection at every public entry point (CONVENTIONS item 6).
//
// A validation guard written as `if x <= 0.0 { return Err(..) }` silently
// ACCEPTS NaN, because every IEEE-754 comparison against NaN is false, and
// Rust is no different from C here (`f64::NAN < 0.0` is `false`). Worse,
// `f64::max` propagates the *other* operand, so `NaN.max(0.0)` is `0.0` —
// which turned a corrupt covariance into a portfolio sigma, VaR and ES of
// exactly ZERO. A NaN risk number breaches no limit and colours no traffic
// light (`NaN <= limit` is false, and so is `NaN > limit`); a zero one
// looks like a perfectly hedged book. Both must be errors.
// -----------------------------------------------------------------------

#[test]
fn non_finite_market_and_book_inputs_are_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Market: spot and rate.
        assert!(Market::new([("EUR", bad)], [("USD", 0.05)]).is_err(), "spot {bad}");
        assert!(Market::new([("EUR", 1.10)], [("USD", bad)]).is_err(), "rate {bad}");
        // Positions: notionals, entry rates, strikes, expiries.
        let m = market_eur_jpy_hkd();
        let cases: Vec<Position> = vec![
            Position::Cash(CashPosition::new("EUR", bad)),
            Position::Spot(SpotPosition::new("EURUSD", bad, None)),
            Position::Spot(SpotPosition::new("EURUSD", 1e6, Some(bad))),
            Position::Forward(ForwardPosition::new("EURUSD", bad, 0.5, None)),
            Position::Forward(ForwardPosition::new("EURUSD", 1e6, bad, None)),
            Position::Forward(ForwardPosition::new("EURUSD", 1e6, 0.5, Some(bad))),
        ];
        for (i, p) in cases.into_iter().enumerate() {
            let res = Book::new(vec![p], "USD");
            assert!(res.is_err(), "position case {i} with {bad} must be rejected");
        }
        // A clean book still builds, so the guards are not rejecting all input.
        assert!(Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 1e6, None))],
            "USD"
        )
        .is_ok());
        let _ = m;
    }
}

#[test]
fn non_finite_returns_covariances_and_scalars_are_rejected() {
    let bads = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let m = market_eur_jpy_hkd();
    let book =
        Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 1e6, None))], "USD").unwrap();

    for bad in bads {
        // Return history: NaN *and* Inf must both be refused. An
        // `is_nan()`-only screen (the shape this engine used to have) lets
        // +/-inf straight through into an infinite covariance.
        let mut data = Matrix::zeros(120, 3);
        for t in 0..120 {
            for j in 0..3 {
                data.set(t, j, 0.001 * ((t + j) as f64).sin());
            }
        }
        data.set(37, 1, bad);
        let rets = ReturnsMatrix::new(
            vec!["FX:EUR".to_string(), "IR:EUR".to_string(), "IR:USD".to_string()],
            data,
        )
        .unwrap();
        assert!(
            validate_returns(&rets, &["FX:EUR".to_string()], 60).is_err(),
            "validate_returns {bad}"
        );
        assert!(sample_cov(&rets).is_err(), "sample_cov {bad}");
        assert!(ewma_cov(&rets, 0.94).is_err(), "ewma_cov {bad}");
        assert!(ewma_volatility(&rets, 0.94).is_err(), "ewma_volatility {bad}");
        assert!(
            historical_var(&book, &m, &rets, &HistoricalOptions::default()).is_err(),
            "historical_var {bad}"
        );
        assert!(
            parametric_var(&book, &m, &rets, &ParametricOptions::default()).is_err(),
            "parametric_var {bad}"
        );

        // Covariance and exposures straight into the linear-algebra layer.
        let cov = Matrix::from_rows(&[vec![1e-4, bad], vec![bad, 4e-4]]).unwrap();
        assert!(portfolio_sigma(&[1e6, -2e6], &cov).is_err(), "portfolio_sigma cov {bad}");
        assert!(cov.cholesky().is_err(), "cholesky {bad}");
        assert!(cov.cholesky_with_jitter(8).is_err(), "cholesky_with_jitter {bad}");
        assert!(!cov.all_finite());
        assert!(!cov.is_symmetric(1e-12), "a NaN matrix must not count as symmetric");
        let clean = Matrix::from_rows(&[vec![1e-4, 2e-5], vec![2e-5, 4e-4]]).unwrap();
        assert!(portfolio_sigma(&[1e6, bad], &clean).is_err(), "portfolio_sigma w {bad}");
        assert!(reverse_stress_linear(&[1e6, bad], &clean, 3.0).is_err());
        assert!(reverse_stress_linear(&[1e6, -2e6], &clean, bad).is_err());
        assert!(reverse_stress_for_loss(&[1e6, bad], &clean, 1e5).is_err());
        assert!(reverse_stress_numerical(&[1e6, -2e6], &clean, bad, 1).is_err());

        // Closed-form tail formulae.
        assert!(normal_var(bad, 0.99, 0.0).is_err(), "normal_var sigma {bad}");
        assert!(normal_var(1.0, 0.99, bad).is_err(), "normal_var mean {bad}");
        assert!(normal_es(bad, 0.99, 0.0).is_err());
        assert!(normal_es(1.0, 0.99, bad).is_err());
        assert!(student_t_var(bad, 0.99, 6.0, 0.0).is_err());
        assert!(student_t_var(1.0, 0.99, bad, 0.0).is_err(), "student_t df {bad}");
        assert!(student_t_var(1.0, 0.99, 6.0, bad).is_err());
        assert!(student_t_es(1.0, 0.99, bad, 0.0).is_err());
        assert!(var_covar(&[1e6], &Matrix::from_rows(&[vec![1e-4]]).unwrap(),
                          0.99, 1.0, TailDist::Normal, 6.0, bad).is_err());
        assert!(var_covar(&[1e6], &Matrix::from_rows(&[vec![1e-4]]).unwrap(),
                          0.99, bad, TailDist::Normal, 6.0, 0.0).is_err(),
                "horizon {bad}");
        assert!(var_covar(&[1e6], &Matrix::from_rows(&[vec![1e-4]]).unwrap(),
                          bad, 1.0, TailDist::Normal, 6.0, 0.0).is_err(), "alpha {bad}");

        // Empirical estimators and their weights.
        let mut pnl: Vec<f64> = (0..100).map(|t| 1000.0 * (0.3 * t as f64).sin()).collect();
        pnl[11] = bad;
        assert!(empirical_var(&pnl, 0.99, None).is_err(), "empirical_var {bad}");
        assert!(empirical_es(&pnl, 0.99, None).is_err(), "empirical_es {bad}");
        let clean_pnl: Vec<f64> = (0..100).map(|t| 1000.0 * (0.3 * t as f64).sin()).collect();
        let mut w = vec![0.01; 100];
        w[3] = bad;
        assert!(empirical_var(&clean_pnl, 0.99, Some(&w)).is_err(), "weights {bad}");

        // Backtest series.
        assert!(
            evaluate_var_backtest(&pnl, &vec![1000.0; 100], 0.99).is_err(),
            "backtest pnl {bad}"
        );
        let mut vf = vec![1000.0; 100];
        vf[42] = bad;
        assert!(
            evaluate_var_backtest(&clean_pnl, &vf, 0.99).is_err(),
            "backtest var forecast {bad}"
        );

        // Stress scenario construction.
        assert!(simple_to_log(bad).is_err(), "simple_to_log {bad}");
        assert!(usd_broad_move(&["EUR".to_string()], bad).is_err(), "usd_broad_move {bad}");
        assert!(
            peg_break_scenario("HKD", bad, &HashMap::new()).is_err(),
            "peg_break jump {bad}"
        );
        assert!(
            peg_break_scenario("HKD", -0.30, &HashMap::from([("CNY".to_string(), bad)])).is_err(),
            "peg_break contagion {bad}"
        );
    }

    // A negative VaR forecast is a data error, not a very safe day.
    let clean_pnl: Vec<f64> = (0..100).map(|t| 1000.0 * (0.3 * t as f64).sin()).collect();
    let mut vf = vec![1000.0; 100];
    vf[9] = -1.0;
    assert!(evaluate_var_backtest(&clean_pnl, &vf, 0.99).is_err());
    assert!(evaluate_var_backtest(&clean_pnl, &vec![1000.0; 100], 0.99).is_ok());
}

#[test]
fn simple_to_log_rejects_its_domain_boundary() {
    // ln(1 + pct) is -inf at pct = -100% and NaN below it. Returning
    // either silently would put an infinite/NaN shock into a scenario and
    // an infinite/NaN row into the stress report.
    assert!(simple_to_log(-1.0).is_err(), "-100% must be rejected (ln 0 = -inf)");
    assert!(simple_to_log(-1.000_001).is_err());
    assert!(simple_to_log(-2.0).is_err());
    assert!(simple_to_log(f64::NAN).is_err());
    assert!(simple_to_log(f64::INFINITY).is_err());
    // Just inside the domain it is finite and matches ln1p exactly.
    for pct in [-0.999, -0.5, -0.081, 0.0, 0.149, 10.0] {
        let v = simple_to_log(pct).unwrap();
        assert!(v.is_finite(), "pct={pct} gave {v}");
        assert_eq!(v, (pct as f64).ln_1p());
    }
    // And it round-trips: exp(log) - 1 == pct.
    for pct in [-0.5, -0.081, 0.149] {
        let back = simple_to_log(pct).unwrap().exp() - 1.0;
        assert!((back - pct).abs() < 1e-14, "round trip {pct} -> {back}");
    }
    // A -100% peg break is rejected at the scenario builder too.
    assert!(peg_break_scenario("HKD", -1.0, &HashMap::new()).is_err());
    // ... while a 99.9% devaluation (finite, extreme, legal) is not.
    let sc = peg_break_scenario("ARS", -0.999, &HashMap::new()).unwrap();
    assert!(sc.shocks.values().all(|v| v.is_finite()));
}

// -----------------------------------------------------------------------
// Single-currency books and identity triangulation.
// -----------------------------------------------------------------------

#[test]
fn single_currency_book_has_no_fx_risk() {
    // A USD-only cash book held in USD carries no factor at all: no FX
    // factor (USD is the pivot, its USD price is identically 1) and no
    // rate factor (cash has no discounting leg). Its VaR is exactly zero,
    // and it must not error out on the way there.
    let m = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
    let usd_only =
        Book::new(vec![Position::Cash(CashPosition::new("USD", 250e6))], "USD").unwrap();
    let cb = CompiledBook::new(&usd_only, &m).unwrap();
    assert!(
        cb.factors().iter().all(|f| !f.starts_with("FX:")),
        "USD-only book should carry no FX factor, got {:?}",
        cb.factors()
    );
    assert!((cb.value0_usd() - 250e6).abs() < 1e-6);
    // Any shock at all leaves it flat.
    let shocks = HashMap::from([
        ("FX:EUR".to_string(), 0.10),
        ("IR:USD".to_string(), 0.01),
        ("IR:EUR".to_string(), -0.02),
    ]);
    assert!(cb.pnl_map(&shocks).unwrap().abs() < 1e-6);

    // A single *foreign* currency book, reported in that same currency,
    // is also flat: EUR cash valued in EUR has no exposure to EURUSD.
    let eur_book =
        Book::new(vec![Position::Cash(CashPosition::new("EUR", 100e6))], "EUR").unwrap();
    let cb_eur = CompiledBook::new(&eur_book, &m).unwrap();
    assert!(
        cb_eur.pnl_map(&HashMap::from([("FX:EUR".to_string(), 0.10)])).unwrap().abs()
            < 1e-6 * 100e6,
        "EUR cash reported in EUR must be FX-flat"
    );
    // ... but the same position reported in USD is fully exposed.
    let eur_in_usd =
        Book::new(vec![Position::Cash(CashPosition::new("EUR", 100e6))], "USD").unwrap();
    let cb_usd = CompiledBook::new(&eur_in_usd, &m).unwrap();
    let pnl = cb_usd.pnl_map(&HashMap::from([("FX:EUR".to_string(), 0.10)])).unwrap();
    let expected = 100e6 * 1.10 * ((0.10_f64).exp() - 1.0);
    assert!((pnl - expected).abs() < 1e-6 * expected.abs(), "{pnl} vs {expected}");
}

#[test]
fn identity_triangulation_is_exact() {
    // Every cross is triangulated through USD. The identity cases must
    // come back exactly: CCYUSD is the currency's USD price, USDUSD is 1,
    // and CCYCCY is 1 for every currency in the market.
    let m = Market::new(
        [("EUR", 1.10), ("JPY", 0.0090), ("GBP", 1.27)],
        [("USD", 0.050), ("EUR", 0.030), ("JPY", 0.001), ("GBP", 0.045)],
    )
    .unwrap();
    assert_eq!(m.spot("USD").unwrap(), 1.0);
    assert_eq!(m.cross("EURUSD").unwrap(), m.spot("EUR").unwrap());
    // A pair with identical legs is not a tradeable pair: rather than
    // returning a vacuous 1.0 that would silently create a zero-risk
    // "position", the engine rejects it. That is the identity case of the
    // triangulation rule, and it is enforced, not assumed.
    for c in ["EUR", "JPY", "GBP", "USD"] {
        let pair = format!("{c}{c}");
        assert!(
            matches!(m.cross(&pair), Err(FxVarError::Invalid(_))),
            "{pair} (identical legs) must be rejected, not triangulated to 1"
        );
        assert!(
            Book::new(vec![Position::Spot(SpotPosition::new(&pair, 1e6, None))], "USD").is_err(),
            "a {pair} position must be rejected"
        );
    }
    // USDCCY is the exact reciprocal of CCYUSD, and the cross is the exact
    // ratio of the two USD legs (this is what makes the factor set
    // arbitrage-consistent: there is no independent cross factor).
    for (a, b) in [("EUR", "JPY"), ("GBP", "EUR"), ("JPY", "GBP")] {
        let direct = m.cross(&format!("{a}{b}")).unwrap();
        let via_usd = m.spot(a).unwrap() / m.spot(b).unwrap();
        assert!(
            (direct - via_usd).abs() <= 1e-15 * via_usd.abs(),
            "{a}{b}: {direct} vs {via_usd}"
        );
        // Reciprocal consistency both ways.
        let inverse = m.cross(&format!("{b}{a}")).unwrap();
        assert!((direct * inverse - 1.0).abs() < 1e-14, "{a}{b} x {b}{a} = {}", direct * inverse);
    }
    // The forward at T = 0 is the spot cross, exactly.
    for pair in ["EURUSD", "EURJPY", "USDJPY"] {
        assert_eq!(m.forward(pair, 0.0).unwrap(), m.cross(pair).unwrap());
    }
}

// -----------------------------------------------------------------------
// Scale: a large-notional book must not be rejected for being large.
// -----------------------------------------------------------------------

#[test]
fn large_notional_books_are_not_rejected_by_absolute_tolerances() {
    // A 50bn-notional book produces a P&L covariance with entries ~1e14.
    // Symmetry and PSD gates written with ABSOLUTE tolerances reject such
    // a matrix purely because its two triangles were accumulated in
    // different orders and differ in the last ulp — which at that scale is
    // ~1e-2, ten orders of magnitude above a 1e-12 absolute gate. Both
    // gates in this engine are scale-relative; this pins that.
    let sigma_a = 5.0e11_f64; // P&L std of leg A, currency units
    let sigma_b = 2.0e11_f64;
    let rho = 0.35;
    let off = rho * sigma_a * sigma_b;
    // Perturb one triangle by a single ulp to mimic a different summation
    // order in the estimator.
    let off_b = f64::from_bits(off.to_bits() + 1);
    let cov = Matrix::from_rows(&[
        vec![sigma_a * sigma_a, off],
        vec![off_b, sigma_b * sigma_b],
    ])
    .unwrap();
    assert_ne!(off, off_b);
    assert!(
        (off - off_b).abs() > 1e-12,
        "the ulp perturbation must exceed an absolute 1e-12 gate for this test to bite"
    );
    assert!(
        cov.is_symmetric_rel(1e-12),
        "a one-ulp asymmetry at scale 1e22 must still count as symmetric"
    );
    assert!(
        !cov.is_symmetric(1e-12),
        "the absolute gate is the one that (wrongly) rejects it — kept for callers who know their units"
    );
    let (l, jitter) = cov.cholesky_with_jitter(8).unwrap();
    assert_eq!(jitter, 0.0, "a healthy large-notional covariance needs no jitter");
    assert!(l.get(0, 0) > 0.0);

    // Portfolio sigma / VaR on the same scale: finite, positive, and
    // proportional to notional (no absolute floor bites).
    let w = [1.0, 1.0];
    let s = portfolio_sigma(&w, &cov).unwrap();
    let expect = (sigma_a * sigma_a + 2.0 * off + sigma_b * sigma_b).sqrt();
    assert!((s - expect).abs() <= 1e-12 * expect, "{s} vs {expect}");
    let big = var_covar(&w, &cov, 0.99, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
    let small_cov = Matrix::from_rows(&[
        vec![1.0e-6 * sigma_a * sigma_a, 1.0e-6 * off],
        vec![1.0e-6 * off, 1.0e-6 * sigma_b * sigma_b],
    ])
    .unwrap();
    let small = var_covar(&w, &small_cov, 0.99, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
    assert!((big.var / small.var - 1.0e3).abs() < 1e-6 * 1.0e3, "VaR must scale with sigma");
}

// -----------------------------------------------------------------------
// Boundary confidence levels and minimum-size samples.
// -----------------------------------------------------------------------

#[test]
fn boundary_alpha_and_tiny_samples() {
    let pnl: Vec<f64> = (0..100).map(|t| 1000.0 * (0.37 * t as f64).sin()).collect();
    // alpha is a CONFIDENCE level here (0.99 = 99% VaR), so the open
    // interval is (0, 1). The closed boundaries and anything outside must
    // be rejected at every entry point, not clamped.
    for bad in [0.0, 1.0, -0.01, 1.0 + 1e-12, 2.0, f64::NAN] {
        assert!(empirical_var(&pnl, bad, None).is_err(), "empirical_var alpha={bad}");
        assert!(empirical_es(&pnl, bad, None).is_err(), "empirical_es alpha={bad}");
        assert!(normal_var(1.0, bad, 0.0).is_err(), "normal_var alpha={bad}");
        assert!(student_t_es(1.0, bad, 6.0, 0.0).is_err(), "student_t_es alpha={bad}");
        assert!(
            evaluate_var_backtest(&pnl, &vec![1000.0; 100], bad).is_err(),
            "backtest alpha={bad}"
        );
    }
    // Just inside both ends still works, and VaR is monotone in alpha.
    let lo = empirical_var(&pnl, 0.01, None).unwrap();
    let hi = empirical_var(&pnl, 0.999, None).unwrap();
    assert!(hi >= lo, "VaR must be non-decreasing in the confidence level");
    assert!(normal_var(1.0, 1e-9, 0.0).unwrap() < normal_var(1.0, 1.0 - 1e-9, 0.0).unwrap());
    // Horizon must be finite and positive.
    let cov1 = Matrix::from_rows(&[vec![1e-4]]).unwrap();
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            var_covar(&[1e6], &cov1, 0.99, bad, TailDist::Normal, 6.0, 0.0).is_err(),
            "horizon={bad}"
        );
    }

    // Minimum-size samples: 1 and 2 observations.
    assert_eq!(empirical_var(&[-42.0], 0.99, None).unwrap(), 42.0);
    assert_eq!(empirical_es(&[-42.0], 0.99, None).unwrap(), 42.0);
    let two = [-100.0, 20.0];
    assert_eq!(empirical_var(&two, 0.99, None).unwrap(), 100.0);
    // At 40% confidence the tail mass is 0.6, which reaches past the
    // worst scenario (weight 0.5) onto the second: VaR is -20, i.e. the
    // book is expected to be UP 20 at that (useless) confidence level.
    assert_eq!(empirical_var(&two, 0.4, None).unwrap(), -20.0);
    assert!(empirical_var(&[] as &[f64], 0.99, None).is_err());
    // Covariance and EWMA need 2 rows; a single row is refused.
    let one_row = ReturnsMatrix::new(
        vec!["FX:EUR".to_string()],
        Matrix::from_rows(&[vec![0.01]]).unwrap(),
    )
    .unwrap();
    assert!(sample_cov(&one_row).is_err());
    assert!(ewma_volatility(&one_row, 0.94).is_err());
    let two_rows = ReturnsMatrix::new(
        vec!["FX:EUR".to_string()],
        Matrix::from_rows(&[vec![0.01], vec![-0.01]]).unwrap(),
    )
    .unwrap();
    let cov = sample_cov(&two_rows).unwrap();
    assert!((cov.cov.get(0, 0) - 2.0e-4).abs() < 1e-18);
    // The backtest needs at least 2 days.
    assert!(evaluate_var_backtest(&[-1.0], &[1.0], 0.99).is_err());
    assert!(evaluate_var_backtest(&[-1.0, 2.0], &[1.0, 1.0], 0.99).is_ok());
}

// -----------------------------------------------------------------------
// Jitter materiality: repair rounding noise, refuse real indefiniteness.
// -----------------------------------------------------------------------

#[test]
fn jitter_repairs_pegged_blocks_but_refuses_indefinite_covariances() {
    use fx_var_engine::matrix::MAX_RELATIVE_JITTER;

    // Two currencies pegged to the same anchor: perfectly correlated, so
    // the covariance is exactly singular but still PSD. Repaired at the
    // first rung, and the jitter reported to the caller is tiny.
    // (Use a power of two so the singularity is exact in binary floating
    // point rather than one ulp away from it.)
    let v = 2.0_f64.powi(-26);
    let pegged = Matrix::from_rows(&[vec![v, v], vec![v, v]]).unwrap();
    assert!(pegged.cholesky().is_err(), "an exactly singular block must defeat plain Cholesky");
    let (_l, jitter) = pegged.cholesky_with_jitter(8).unwrap();
    assert!(
        jitter > 0.0 && jitter <= MAX_RELATIVE_JITTER * v,
        "peg block needed jitter {jitter}, cap is {}",
        MAX_RELATIVE_JITTER * v
    );

    // A materially indefinite matrix (correlation > 1) must NOT be
    // silently patched into something factorisable: repairing it would
    // change the simulated risk with no diagnostic.
    let indefinite = Matrix::from_rows(&[vec![1e-4, 1.4e-4], vec![1.4e-4, 1e-4]]).unwrap();
    assert!(indefinite.cholesky().is_err());
    let res = indefinite.cholesky_with_jitter(30);
    assert!(
        matches!(res, Err(FxVarError::Numerical(_))),
        "materially indefinite covariance must not be silently repaired, got {res:?}"
    );
    // The quadratic form catches it too, in the direction that exposes the
    // negative eigenvalue.
    assert!(portfolio_sigma(&[1e6, -1e6], &indefinite).is_err());
    assert!(MAX_RELATIVE_JITTER > 0.0 && MAX_RELATIVE_JITTER <= 1e-4);
}
