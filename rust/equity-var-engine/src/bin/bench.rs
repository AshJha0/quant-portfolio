//! bench — end-to-end timings for the `eq_var_engine` on a realistic
//! desk-sized problem: a 250-day x 100-asset returns panel (one year of
//! history, 100 risk factors). Mirrors `cpp/equity-var-engine/bench/bench_main.cpp`
//! in spirit and stage list.
//!
//! Measured stages:
//! 1. sample + EWMA covariance estimation from the panel (100 x 100);
//! 2. historical VaR family (plain / BRW / FHS) on the portfolio P&L;
//! 3. parametric VaR (normal + Student-t);
//! 4. Monte Carlo VaR + ES, 100'000 paths, normal and Student-t (the
//!    dominant cost: 100k x 100 correlated draws through the Cholesky
//!    factor).
//!
//! Deterministic inputs: the panel is a one-factor market return plus
//! idiosyncratic noise drawn from the crate's seeded RNG, so the sample
//! covariance is full rank (250 days > 100 assets). The Monte Carlo stage
//! simulates once and derives VaR, ES and the bootstrap SE from the same
//! scenarios. Each stage is repeated and the best-of-R wall time is
//! reported (single thread, CPU bound).
//!
//! Run with `cargo run --release --bin bench`.

use std::time::Instant;

use eq_var_engine::prelude::*;

fn best_of_ms<F: FnMut() -> f64>(reps: usize, sink: &mut f64, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        *sink += f();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best {
            best = ms;
        }
    }
    best
}

fn row(stage: &str, ms: f64, result: &str) {
    println!("| {stage:<42} | {ms:>10.3} | {result:<28} |");
}

fn money(v: f64) -> String {
    format!("{v:.0}")
}

fn main() {
    const DAYS: usize = 250;
    const ASSETS: usize = 100;
    const PATHS: usize = 100_000;
    const ALPHA: f64 = 0.01;

    // Deterministic 250 x 100 daily returns panel: per-asset vol ramps from
    // 10 bp to ~2 %, phase-shifted sin/cos terms induce rich correlation.
    let mut panel = Matrix::zeros(DAYS, ASSETS);
    let mut rng = Rng::new(20_261_101);
    for t in 0..DAYS {
        let market = rng.standard_normal();
        for j in 0..ASSETS {
            let vol = 0.001 + 0.0002 * j as f64;
            let v = vol * (0.6 * market + 0.8 * rng.standard_normal());
            panel.set(t, j, v);
        }
    }
    // Long-biased book: exposures alternate long/short, 100k..1.09M per name.
    let exposures: Vec<f64> = (0..ASSETS)
        .map(|j| {
            let sign = if j % 3 == 2 { -1.0 } else { 1.0 };
            sign * (1.0e5 + 1.0e4 * j as f64)
        })
        .collect();

    println!(
        "eq_var_engine benchmark - {DAYS}-day x {ASSETS}-asset panel, alpha = {ALPHA:.2}, {PATHS} MC paths, single thread\n"
    );
    println!("| {:<42} | {:>10} | {:<28} |", "stage", "best ms", "result");
    println!("|{}|{}|{}|", "-".repeat(44), "-".repeat(12), "-".repeat(30));

    let mut sink = 0.0;

    // --- covariance estimation ---------------------------------------------
    let mut cov = Matrix::zeros(ASSETS, ASSETS);
    let ms_cov = best_of_ms(5, &mut sink, || {
        cov = sample_covariance(&panel).unwrap();
        cov.get(0, 0)
    });
    row("sample covariance (250 x 100 -> 100 x 100)", ms_cov, "");

    let mut ecov = Matrix::zeros(ASSETS, ASSETS);
    let ms_ecov = best_of_ms(5, &mut sink, || {
        ecov = ewma_covariance(&panel, 0.94).unwrap();
        ecov.get(0, 0)
    });
    row("EWMA covariance (lam = 0.94)", ms_ecov, "");

    // --- historical family --------------------------------------------------
    let pnl = portfolio_pnl(&panel, &exposures).unwrap();
    let mut v = 0.0;
    let ms_hist = best_of_ms(20, &mut sink, || {
        v = historical_var(&pnl, ALPHA).unwrap();
        v
    });
    row("historical VaR 99% (250 scenarios)", ms_hist, &money(v));

    let ms_brw = best_of_ms(20, &mut sink, || {
        v = age_weighted_var(&pnl, ALPHA, 0.98).unwrap();
        v
    });
    row("BRW age-weighted VaR 99% (lam = 0.98)", ms_brw, &money(v));

    let ms_fhs = best_of_ms(20, &mut sink, || {
        v = filtered_historical_var_factors(&panel, &exposures, ALPHA, 0.94).unwrap();
        v
    });
    row("filtered (FHS, per factor) VaR 99%", ms_fhs, &money(v));

    let ms_es = best_of_ms(20, &mut sink, || {
        v = expected_shortfall(&pnl, 0.025).unwrap();
        v
    });
    row("empirical ES 97.5%", ms_es, &money(v));

    // --- parametric ----------------------------------------------------------
    let ms_par_n = best_of_ms(20, &mut sink, || {
        v = parametric_var(&exposures, &cov, ALPHA, TailModel::Normal).unwrap();
        v
    });
    row("parametric VaR 99% (normal)", ms_par_n, &money(v));

    let ms_par_t = best_of_ms(20, &mut sink, || {
        v = parametric_var(&exposures, &cov, ALPHA, TailModel::StudentT { df: 6.0 }).unwrap();
        v
    });
    row(
        "parametric VaR 99% (Student-t, df = 6)",
        ms_par_t,
        &money(v),
    );

    let sigma = portfolio_sigma(&exposures, &cov).unwrap();
    let ms_nes = best_of_ms(20, &mut sink, || {
        v = normal_es(sigma, 0.025, 0.0).unwrap();
        v
    });
    row("closed-form ES 97.5% (normal)", ms_nes, &money(v));

    // --- Cholesky (isolated) -------------------------------------------------
    let ms_chol = best_of_ms(5, &mut sink, || {
        let l = cov.cholesky_repair(PsdRepair::None).unwrap();
        assert_eq!(l.jitter, 0.0, "bench covariance must be full rank");
        l.factor.get(0, 0)
    });
    row("Cholesky 100 x 100", ms_chol, "");

    // --- Monte Carlo ---------------------------------------------------------
    let mut run = None;
    let ms_mc_n = best_of_ms(3, &mut sink, || {
        let r = monte_carlo_run(&exposures, &cov, ALPHA, PATHS, TailModel::Normal, 42, 200).unwrap();
        let v = r.var;
        run = Some(r);
        v
    });
    let rn = run.take().unwrap();
    row(
        "MC VaR+ES+SE normal (100k x 100, 1 sim)",
        ms_mc_n,
        &format!("{} (SE {})", money(rn.var), money(rn.se)),
    );
    let ms_mc_t = best_of_ms(3, &mut sink, || {
        let r = monte_carlo_run(&exposures, &cov, ALPHA, PATHS, TailModel::StudentT { df: 6.0 }, 42, 200)
            .unwrap();
        let v = r.var;
        run = Some(r);
        v
    });
    let rt = run.take().unwrap();
    row(
        "MC VaR+ES+SE Student-t df=6 (1 sim)",
        ms_mc_t,
        &format!("{} (SE {})", money(rt.var), money(rt.se)),
    );
    let (mc_var_n, mc_es_n) = (rn.var, rn.es);

    let total = ms_cov + ms_ecov + ms_hist + ms_brw + ms_fhs + ms_es + ms_par_n + ms_par_t + ms_mc_n;
    println!(
        "\nfull daily batch (cov + historical family + parametric + 100k-path MC): {total:.1} ms"
    );
    let exact = parametric_var(&exposures, &cov, ALPHA, TailModel::Normal).unwrap();
    println!("MC normal vs parametric normal: {} vs {} (MC SE {} bootstrap, {} KDE)", money(mc_var_n), money(exact), money(rn.se), money(rn.se_kde));
    println!(
        "MC normal ES 99%: {}  |  closed-form: {}",
        money(mc_es_n),
        money(normal_es(sigma, ALPHA, 0.0).unwrap())
    );
    println!("[checksum {sink:.6}]");
}
