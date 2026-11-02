//! Per-call latency benchmark for the fx-var-engine kernels.
//!
//! Book: 250 positions on 50 factors (45 FX + 5 IR): 200 spots and 50
//! forwards. History: 500 days of seeded factor returns, full-rank
//! covariance. Each kernel is warmed up, then timed call by call with
//! `std::time::Instant` and `std::hint::black_box`, cycling through varied
//! inputs. Kernels faster than about 1 us are timed in batches of
//! `calls_per_sample` calls and reported per call.
//!
//! Output: a JSON array (one object per kernel with p50, p90, p99, p99.9
//! and max in ns) printed to stdout and written to `bench/results.json`
//! under the crate root (or to `--out <path>`). Run with `cargo run --release --bin bench`.

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use fx_var_engine::prelude::*;

struct Stat {
    kernel: &'static str,
    n_iter: usize,
    calls_per_sample: usize,
    p: [f64; 5],
}

/// Time `n_iter` samples of `calls` calls each after `warm` warm-up
/// samples. `f(i)` runs call number `i` (the caller varies inputs by `i`).
fn measure(
    kernel: &'static str,
    warm: usize,
    n_iter: usize,
    calls: usize,
    mut f: impl FnMut(usize),
) -> Stat {
    let mut i = 0usize;
    for _ in 0..warm * calls {
        f(i);
        i += 1;
    }
    let mut ns = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let t0 = Instant::now();
        for _ in 0..calls {
            f(i);
            i += 1;
        }
        ns.push(t0.elapsed().as_nanos() as f64 / calls as f64);
    }
    ns.sort_unstable_by(f64::total_cmp);
    let q = |p: f64| ns[((p * (ns.len() - 1) as f64).round() as usize).min(ns.len() - 1)];
    let p = [q(0.5), q(0.9), q(0.99), q(0.999), ns[ns.len() - 1]];
    eprintln!(
        "{kernel:<34} p50 {:>12.0}  p99 {:>12.0}  p99.9 {:>12.0} ns",
        p[0], p[2], p[3]
    );
    Stat {
        kernel,
        n_iter,
        calls_per_sample: calls,
        p,
    }
}

fn cmd(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

fn cpu() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    // ---- deterministic 45-currency universe -----------------------------
    let ccys: Vec<String> = (0..45)
        .map(|i| {
            format!(
                "C{}{}",
                (b'A' + (i / 26) as u8) as char,
                (b'A' + (i % 26) as u8) as char
            )
        })
        .collect();
    let mut spots: Vec<(String, f64)> = Vec::new();
    let mut rates: Vec<(String, f64)> = vec![("USD".to_string(), 0.05)];
    for (i, c) in ccys.iter().enumerate() {
        spots.push((c.clone(), 0.5 + 0.02 * i as f64));
        rates.push((c.clone(), 0.01 + 0.001 * (i % 10) as f64));
    }
    let market = Market::new(spots, rates).expect("market");
    let mut positions = Vec::new();
    for i in 0..200 {
        let notional = if i % 2 == 1 { -1.0 } else { 1.0 } * (1.0e6 + 2.0e4 * i as f64);
        positions.push(Position::Spot(SpotPosition::new(
            format!("{}USD", ccys[i % 45]),
            notional,
            None,
        )));
    }
    for i in 0..50 {
        let notional = if i % 2 == 1 { -1.0 } else { 1.0 } * 2.0e6;
        positions.push(Position::Forward(ForwardPosition::new(
            format!("{}USD", ccys[i % 4]),
            notional,
            0.25 + 0.05 * i as f64,
            None,
        )));
    }
    let book = Book::new(positions, "USD").expect("book");
    let compiled = CompiledBook::new(&book, &market).expect("compile");
    let factors = compiled.factors().to_vec();
    let k = factors.len();

    // ---- 500-day deterministic history ----------------------------------
    let n_days = 500;
    let mut data = Matrix::zeros(n_days, k);
    let mut rng = Rng::new(20_261_101);
    for t in 0..n_days {
        for (j, f) in factors.iter().enumerate() {
            let scale = if f.starts_with("FX:") { 0.006 } else { 0.0004 };
            data.set(
                t,
                j,
                scale * (0.5 * (0.1 * t as f64 + 0.37 * j as f64).sin() + rng.normal()),
            );
        }
    }
    let rets = ReturnsMatrix::new(factors.clone(), data).expect("returns");
    let cov = sample_cov(&rets).expect("cov");
    let exposures = compiled.linear_exposures(1e-6).expect("exposures");

    // Eight distinct 100k-scenario P&L samples for the tail kernels.
    let samples: Vec<Vec<f64>> = (0..8)
        .map(|s| {
            let mut r = Rng::new(100 + s);
            (0..100_000).map(|_| 4.0e5 * r.normal()).collect()
        })
        .collect();

    let mut stats = Vec::new();

    stats.push(measure("pnl_one_scenario_250pos", 1_000, 100_000, 1, |i| {
        black_box(compiled.pnl(black_box(rets.data.row(i % n_days))).ok());
    }));
    stats.push(measure("portfolio_sigma_50f", 1_000, 100_000, 1, |i| {
        let w = &exposures;
        let _ = i;
        black_box(portfolio_sigma(black_box(w), black_box(&cov.cov)).ok());
    }));
    stats.push(measure("normal_var_closed_form", 1_000, 100_000, 64, |i| {
        black_box(normal_var(black_box(1.0e5 + i as f64), 0.01, 0.0).ok());
    }));
    stats.push(measure("empirical_var_es_100k", 20, 2_000, 1, |i| {
        black_box(empirical_var_es(black_box(&samples[i % 8]), 0.01, None).ok());
    }));
    // new-api-begin
    let mut ws = vec![0.0; 100_000];
    stats.push(measure("empirical_var_es_into_100k", 20, 2_000, 1, |i| {
        black_box(empirical_var_es_into(black_box(&samples[i % 8]), 0.01, &mut ws).ok());
    }));
    let mut out = vec![0.0; n_days];
    stats.push(measure("pnl_aligned_into_500scen", 20, 2_000, 1, |_| {
        black_box(
            compiled
                .pnl_aligned_into(black_box(&rets.data), &mut out)
                .ok(),
        );
    }));
    // new-api-end
    let hopts = HistoricalOptions {
        warn_pegs: false,
        ..Default::default()
    };
    stats.push(measure("historical_var_500scen", 20, 2_000, 1, |_| {
        black_box(historical_var(&book, &market, black_box(&rets), &hopts).ok());
    }));
    let popts = ParametricOptions {
        warn_pegs: false,
        ..Default::default()
    };
    stats.push(measure("parametric_var_normal", 20, 2_000, 1, |_| {
        black_box(parametric_var(&book, &market, black_box(&rets), &popts).ok());
    }));
    stats.push(measure("var_se_bootstrap_100k_x200", 2, 50, 1, |i| {
        black_box(
            var_standard_error_bootstrap(black_box(&samples[i % 8]), 0.01, 200, i as u64).ok(),
        );
    }));
    for (dist, name) in [
        (McDist::Normal, "monte_carlo_var_100k_normal"),
        (McDist::StudentT, "monte_carlo_var_100k_t5"),
    ] {
        stats.push(measure(name, 2, 20, 1, |i| {
            let o = MonteCarloOptions {
                n_scenarios: 100_000,
                seed: 42 + i as u64,
                dist,
                df: 5.0,
                ..Default::default()
            };
            black_box(monte_carlo_var(&book, &market, black_box(&cov), &o).ok());
        }));
    }

    // ---- JSON --------------------------------------------------------------
    let toolchain = cmd("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_string());
    let flags = std::env::var("BENCH_FLAGS")
        .unwrap_or_else(|_| "release: opt-level=3, lto=fat, codegen-units=1".to_string());
    let cpu = format!("{} (WSL2)", cpu());
    let date = cmd("date", &["-u", "+%Y-%m-%d"]).unwrap_or_else(|| "unknown".to_string());
    let mut json = String::from("[\n");
    for (n, s) in stats.iter().enumerate() {
        let _ = writeln!(
            json,
            "  {{\"engine\": \"fx-var-engine\", \"kernel\": \"{}\", \"toolchain\": \"{}\", \"flags\": \"{}\", \
             \"cpu\": \"{}\", \"date\": \"{}\", \"n_iter\": {}, \"calls_per_sample\": {}, \"unit\": \"ns\", \
             \"p50\": {:.0}, \"p90\": {:.0}, \"p99\": {:.0}, \"p999\": {:.0}, \"max\": {:.0}}}{}",
            s.kernel,
            toolchain,
            flags,
            cpu,
            date,
            s.n_iter,
            s.calls_per_sample,
            s.p[0],
            s.p[1],
            s.p[2],
            s.p[3],
            s.p[4],
            if n + 1 < stats.len() { "," } else { "" }
        );
    }
    json.push_str("]\n");
    print!("{json}");
    write_results(&out_path(), &json);
}

/// Where to write the JSON: `--out <path>` (or `--out=<path>`) if given,
/// otherwise `bench/results.json` inside this crate.
fn out_path() -> std::path::PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--out" {
            if let Some(p) = args.next() {
                return p.into();
            }
        } else if let Some(p) = a.strip_prefix("--out=") {
            return p.into();
        }
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("bench")
        .join("results.json")
}

/// Write `json` to `path`, creating the parent directory.
fn write_results(path: &std::path::Path, json: &str) {
    let written = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => std::fs::create_dir_all(dir),
        _ => Ok(()),
    }
    .and_then(|()| std::fs::write(path, json));
    match written {
        Ok(()) => eprintln!("wrote {}", path.display()),
        Err(e) => eprintln!("could not write {}: {e}", path.display()),
    }
}
