//! bench: per-call latency distribution of the `eq_var_engine` kernels.
//!
//! Each kernel runs a warm-up, then many timed calls over a rotating set of
//! inputs (different P&L samples, books and seeds). The tool reports p50,
//! p90, p99, p99.9 and max in nanoseconds per call.
//!
//! Kernels that finish in under a microsecond are timed in batches of
//! `calls_per_sample` calls and the batch time is divided by that count.
//! The timer (`std::time::Instant`) resolves about 20-30 ns, so a single
//! call of that size would read as noise.
//!
//! Every kernel runs twice: through the main API (allocating, `String`
//! errors) and through `eq_var_engine::fast` (no allocation). The main API
//! code is unchanged by the latency work, so its rows are the "before"
//! numbers and the `fast` rows are the "after" numbers.
//!
//! Output: a table on stdout and `bench/results.json` (crate-relative).
//! Run with `cargo run --release --bin bench`.

use std::hint::black_box;
use std::time::Instant;

use eq_var_engine::fast::*;
use eq_var_engine::prelude::*;

struct Row {
    kernel: String,
    n_iter: usize,
    k: usize,
    p: [f64; 5],
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

/// Time `n_iter` samples of `k` calls each after `warm` warm-up calls.
/// `f(i)` runs call number `i`; it picks its input from `i`.
fn measure<F: FnMut(usize) -> f64>(
    kernel: &str,
    n_iter: usize,
    k: usize,
    warm: usize,
    mut f: F,
) -> Row {
    let mut sink = 0.0;
    for i in 0..warm {
        sink += f(i);
    }
    let mut t = Vec::with_capacity(n_iter);
    let mut c = 0usize;
    for _ in 0..n_iter {
        let t0 = Instant::now();
        for _ in 0..k {
            sink += f(black_box(c));
            c += 1;
        }
        t.push(t0.elapsed().as_nanos() as f64 / k as f64);
    }
    black_box(sink);
    t.sort_by(f64::total_cmp);
    let p = [
        pct(&t, 0.5),
        pct(&t, 0.9),
        pct(&t, 0.99),
        pct(&t, 0.999),
        t[t.len() - 1],
    ];
    println!(
        "| {kernel:<44} | {:>9.0} | {:>9.0} | {:>9.0} | {:>9.0} | {:>10.0} | {k:>3} |",
        p[0], p[1], p[2], p[3], p[4]
    );
    Row {
        kernel: kernel.to_string(),
        n_iter,
        k,
        p,
    }
}

fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn today() -> String {
    // Days since the Unix epoch to a civil date (Howard Hinnant's algorithm).
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let z = secs / 86_400 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02}")
}

fn book(n: usize, seed: u64) -> (Vec<f64>, Matrix) {
    let mut rng = Rng::new(seed);
    let mut cov = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let vi = 0.001 + 0.0002 * i as f64;
            let vj = 0.001 + 0.0002 * j as f64;
            cov.set(i, j, if i == j { vi * vi } else { 0.36 * vi * vj });
        }
    }
    let w = (0..n)
        .map(|_| 1.0e5 * (1.0 + rng.uniform()) * if rng.uniform() < 0.3 { -1.0 } else { 1.0 })
        .collect();
    (w, cov)
}

fn main() {
    const SETS: usize = 16;
    // Historical inputs: 16 samples of 250 days (one year) and of 1000 days.
    let mut rng = Rng::new(20_261_101);
    let p250: Vec<Vec<f64>> = (0..SETS)
        .map(|_| (0..250).map(|_| 1.0e4 * rng.standard_normal()).collect())
        .collect();
    let p1000: Vec<Vec<f64>> = (0..SETS)
        .map(|_| (0..1000).map(|_| 1.0e4 * rng.standard_normal()).collect())
        .collect();
    let mut work = vec![0.0; 1000];
    let books: Vec<(Vec<f64>, Matrix)> = (0..SETS).map(|s| book(100, s as u64)).collect();

    println!("eq_var_engine per-call latency, ns (single thread)\n");
    println!(
        "| {:<44} | {:>9} | {:>9} | {:>9} | {:>9} | {:>10} | {:>3} |",
        "kernel", "p50", "p90", "p99", "p99.9", "max", "K"
    );
    println!(
        "|{}|{}|{}|{}|{}|{}|{}|",
        "-".repeat(46),
        "-".repeat(11),
        "-".repeat(11),
        "-".repeat(11),
        "-".repeat(11),
        "-".repeat(12),
        "-".repeat(5)
    );

    let mut rows = Vec::new();
    let n = 20_000;
    let a = 0.01;
    rows.push(measure(
        "historical_var n=250 (main API)",
        n,
        8,
        2000,
        |i| historical_var(&p250[i % SETS], a).unwrap_or(0.0),
    ));
    rows.push(measure("historical_var_ws n=250 (fast)", n, 8, 2000, |i| {
        historical_var_ws(&p250[i % SETS], a, &mut work).unwrap_or(0.0)
    }));
    rows.push(measure(
        "historical_var n=1000 (main API)",
        n,
        2,
        2000,
        |i| historical_var(&p1000[i % SETS], a).unwrap_or(0.0),
    ));
    rows.push(measure(
        "historical_var_ws n=1000 (fast)",
        n,
        2,
        2000,
        |i| historical_var_ws(&p1000[i % SETS], a, &mut work).unwrap_or(0.0),
    ));
    rows.push(measure(
        "expected_shortfall n=250 a=2.5% (main API)",
        n,
        8,
        2000,
        |i| expected_shortfall(&p250[i % SETS], 0.025).unwrap_or(0.0),
    ));
    rows.push(measure(
        "expected_shortfall_ws n=250 a=2.5% (fast)",
        n,
        8,
        2000,
        |i| expected_shortfall_ws(&p250[i % SETS], 0.025, &mut work).unwrap_or(0.0),
    ));
    rows.push(measure(
        "parametric_var normal 100 assets (main API)",
        n,
        2,
        2000,
        |i| {
            let (w, c) = &books[i % SETS];
            parametric_var(w, c, a, TailModel::Normal).unwrap_or(0.0)
        },
    ));
    rows.push(measure(
        "parametric_var_fast normal 100 assets (fast)",
        n,
        2,
        2000,
        |i| {
            let (w, c) = &books[i % SETS];
            parametric_var_fast(w, c, a, TailModel::Normal).unwrap_or(0.0)
        },
    ));
    rows.push(measure(
        "parametric_var t df=6 100 assets (main API)",
        n,
        1,
        2000,
        |i| {
            let (w, c) = &books[i % SETS];
            parametric_var(w, c, a, TailModel::StudentT { df: 6.0 }).unwrap_or(0.0)
        },
    ));
    rows.push(measure(
        "parametric_var_fast t df=6 100 assets (fast)",
        n,
        1,
        2000,
        |i| {
            let (w, c) = &books[i % SETS];
            parametric_var_fast(w, c, a, TailModel::StudentT { df: 6.0 }).unwrap_or(0.0)
        },
    ));
    let mut est = vec![0.0; 200];
    rows.push(measure(
        "var_bootstrap_se n=250 B=200 (main API)",
        2000,
        1,
        100,
        |i| var_bootstrap_se(&p250[i % SETS], a, 200, i as u64).unwrap_or(0.0),
    ));
    rows.push(measure(
        "var_bootstrap_se_ws n=250 B=200 (fast)",
        2000,
        1,
        100,
        |i| {
            var_bootstrap_se_ws(&p250[i % SETS], a, 200, i as u64, &mut work, &mut est)
                .unwrap_or(0.0)
        },
    ));

    // Monte Carlo: 20-factor book, 10k paths, 50 bootstrap resamples.
    let (mw, mc) = book(20, 77);
    let paths = 10_000;
    rows.push(measure(
        "monte_carlo_run 20x10k B=50 (main API)",
        300,
        1,
        10,
        |i| {
            monte_carlo_run(&mw, &mc, a, paths, TailModel::Normal, i as u64, 50)
                .map(|r| r.var)
                .unwrap_or(0.0)
        },
    ));
    let mut ws = McWorkspace::new(20, paths, 50);
    let _ = ws.prepare(&mc, TailModel::Normal);
    rows.push(measure(
        "McWorkspace::run 20x10k B=50 (fast)",
        300,
        1,
        10,
        |i| {
            ws.run(&mw, a, paths, i as u64, 50)
                .map(|r| r.var)
                .unwrap_or(0.0)
        },
    ));
    rows.push(measure(
        "monte_carlo_var 20x10k (main API)",
        300,
        1,
        10,
        |i| monte_carlo_var(&mw, &mc, a, paths, TailModel::Normal, i as u64).unwrap_or(0.0),
    ));
    rows.push(measure("McWorkspace::var 20x10k (fast)", 300, 1, 10, |i| {
        ws.var(&mw, a, paths, i as u64).unwrap_or(0.0)
    }));

    let cpu = cpu_name();
    let date = today();
    let flags = "release: opt-level=3, lto=fat, codegen-units=1; RUSTFLAGS unset";
    let toolchain = "rustc 1.99.1";
    let mut json = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        json.push_str(&format!(
            "  {{\"engine\": \"rust/equity-var-engine\", \"kernel\": \"{}\", \"toolchain\": \"{toolchain}\", \"flags\": \"{flags}\", \"cpu\": \"{cpu} (WSL2)\", \"date\": \"{date}\", \"n_iter\": {}, \"calls_per_sample\": {}, \"unit\": \"ns\", \"p50\": {:.1}, \"p90\": {:.1}, \"p99\": {:.1}, \"p999\": {:.1}, \"max\": {:.1}}}{}\n",
            r.kernel, r.n_iter, r.k, r.p[0], r.p[1], r.p[2], r.p[3], r.p[4],
            if i + 1 < rows.len() { "," } else { "" }
        ));
    }
    json.push_str("]\n");
    let _ = std::fs::create_dir_all("bench");
    match std::fs::write("bench/results.json", &json) {
        Ok(()) => println!("\nwrote bench/results.json"),
        Err(e) => println!("\ncould not write bench/results.json: {e}"),
    }
    println!("{json}");
}
