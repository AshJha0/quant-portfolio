//! Benchmark: 250-position, 50-factor FX book — historical, parametric and
//! 100k-scenario Monte Carlo VaR wall times.
//!
//! The book is deterministic: 45 non-USD currencies (45 FX factors),
//! forwards on 4 currency pairs adding 5 IR factors (incl. IR:USD) for a
//! 50-factor set; 200 spot positions + 50 forwards = 250 positions. History
//! is 500 days of sinusoidal factor returns. Mirrors
//! `cpp/fx-var-engine/bench/bench_main.cpp` in spirit; run with
//! `cargo run --release --bin bench`.

use std::time::Instant;

use fx_var_engine::prelude::*;

fn main() {
    // ---- deterministic 45-currency universe --------------------------
    let mut ccys = Vec::new();
    for i in 0..45 {
        let c = format!("C{}{}", (b'A' + (i / 26) as u8) as char, (b'A' + (i % 26) as u8) as char);
        ccys.push(c);
    }
    let mut spots: Vec<(String, f64)> = Vec::new();
    let mut rates: Vec<(String, f64)> = vec![("USD".to_string(), 0.05)];
    for (i, c) in ccys.iter().enumerate() {
        spots.push((c.clone(), 0.5 + 0.02 * i as f64));
        rates.push((c.clone(), 0.01 + 0.001 * (i % 10) as f64));
    }
    let market = Market::new(spots, rates).unwrap();

    // ---- 250 positions: 200 spots + 50 forwards (5 fwd currencies) ---
    let mut positions = Vec::new();
    for i in 0..200 {
        let pair = format!("{}USD", ccys[i % 45]);
        let notional = if i % 2 == 1 { -1.0 } else { 1.0 } * (1.0e6 + 2.0e4 * i as f64);
        positions.push(Position::Spot(SpotPosition::new(pair, notional, None)));
    }
    for i in 0..50 {
        let pair = format!("{}USD", ccys[i % 4]);
        let notional = if i % 2 == 1 { -1.0 } else { 1.0 } * 2.0e6;
        positions.push(Position::Forward(ForwardPosition::new(pair, notional, 0.25 + 0.05 * i as f64, None)));
    }
    let book = Book::new(positions, "USD").unwrap();
    let compiled = CompiledBook::new(&book, &market).unwrap();
    let factors = compiled.factors().to_vec();
    println!("book: {} positions, {} factors", book.positions().len(), factors.len());

    // ---- 500-day deterministic history --------------------------------
    let n_days = 500;
    let mut data = Matrix::zeros(n_days, factors.len());
    for t in 0..n_days {
        for (j, f) in factors.iter().enumerate() {
            let scale = if f.starts_with("FX:") { 0.006 } else { 0.0004 };
            data.set(t, j, scale * (0.1 * t as f64 + 0.37 * j as f64).sin());
        }
    }
    let rets = ReturnsMatrix::new(factors.clone(), data).unwrap();

    // ---- historical VaR (500 scenarios, full reval) --------------------
    {
        let opts = HistoricalOptions { warn_pegs: false, ..Default::default() };
        let reps = 20;
        let t0 = Instant::now();
        let mut r = None;
        for _ in 0..reps {
            r = Some(historical_var(&book, &market, &rets, &opts).unwrap());
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let r = r.unwrap();
        println!("historical  VaR (500 scen): var={:.0} es={:.0}   {:8.3} ms", r.var, r.es, ms);
    }

    // ---- parametric VaR -------------------------------------------------
    {
        let opts = ParametricOptions { warn_pegs: false, ..Default::default() };
        let reps = 20;
        let t0 = Instant::now();
        let mut r = None;
        for _ in 0..reps {
            r = Some(parametric_var(&book, &market, &rets, &opts).unwrap());
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let r = r.unwrap();
        println!("parametric  VaR (normal)  : var={:.0} es={:.0}   {:8.3} ms", r.var, r.es, ms);
    }

    // ---- Monte Carlo, 100k scenarios, full reval ------------------------
    let cov = sample_cov(&rets).unwrap();
    for (dist, label) in [(McDist::Normal, "normal"), (McDist::StudentT, "t(5)  ")] {
        let opts = MonteCarloOptions { n_scenarios: 100_000, seed: 42, dist, df: 5.0, ..Default::default() };
        let reps = 3;
        let t0 = Instant::now();
        let mut r = None;
        for _ in 0..reps {
            r = Some(monte_carlo_var(&book, &market, &cov, &opts).unwrap());
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let r = r.unwrap();
        println!("monte carlo VaR 100k {label}: var={:.0} es={:.0}   {:8.1} ms", r.var, r.es, ms);
    }
}
