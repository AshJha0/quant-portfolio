//! Per-call latency benchmark for the pricing kernels (std only).
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --bin bench
//! ```
//!
//! Each kernel runs a warm-up, then `n_iter` timed samples over a cycle of
//! 1,024 varied inputs. A sample times `calls_per_sample` back-to-back
//! calls with `std::time::Instant`; the reported value is the sample time
//! divided by `calls_per_sample`. Kernels near the timer resolution
//! (about 20-30 ns under WSL2) use 16 calls per sample, so their
//! percentiles describe 16-call averages, not single calls.
//! `std::hint::black_box` keeps the optimiser from deleting the work.
//!
//! Writes `bench/results.json` (crate-relative) and prints it.
//!
//! Built with `RUSTFLAGS="--cfg before"`, only the kernels of the
//! pre-change API are measured, so the same source gives comparable
//! numbers for the previous version of the crate.

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use eq_options_engine::{
    black76_price, bs_greeks, bs_price, crr_price, implied_vol, mc_price, Exercise, OptionType,
};

/// One varied input row.
#[derive(Clone, Copy)]
struct Row {
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    v: f64,
    q: f64,
    ot: OptionType,
    /// Black-Scholes price of the row, for the implied-vol kernel.
    p: f64,
}

const N_ROWS: usize = 1024;

fn rows() -> Vec<Row> {
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    let mut unit = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / 9_007_199_254_740_992.0
    };
    (0..N_ROWS)
        .map(|i| {
            let s = 100.0;
            let k = 70.0 + 60.0 * unit();
            let t = 0.02 + 2.0 * unit();
            let r = 0.03 * unit();
            let v = 0.1 + 0.5 * unit();
            let q = 0.02 * unit();
            let ot = if i % 2 == 0 {
                OptionType::Call
            } else {
                OptionType::Put
            };
            let p = bs_price(s, k, t, r, v, q, ot).unwrap_or(0.0);
            Row {
                s,
                k,
                t,
                r,
                v,
                q,
                ot,
                p,
            }
        })
        .collect()
}

struct Stat {
    kernel: &'static str,
    n_iter: usize,
    calls_per_sample: usize,
    p50: f64,
    p90: f64,
    p99: f64,
    p999: f64,
    max: f64,
}

/// Time `n_iter` samples of `calls` calls of `f(row_index)` after `warm`
/// warm-up samples.
fn measure(
    kernel: &'static str,
    warm: usize,
    n_iter: usize,
    calls: usize,
    mut f: impl FnMut(usize) -> f64,
) -> Stat {
    let mut idx = 0usize;
    let mut sample = |f: &mut dyn FnMut(usize) -> f64| -> u64 {
        let start = Instant::now();
        let mut acc = 0.0;
        for _ in 0..calls {
            acc += f(idx % N_ROWS);
            idx = idx.wrapping_add(1);
        }
        black_box(acc);
        start.elapsed().as_nanos() as u64
    };
    for _ in 0..warm {
        sample(&mut f);
    }
    let mut ns: Vec<u64> = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        ns.push(sample(&mut f));
    }
    ns.sort_unstable();
    let pick = |q: f64| -> f64 {
        let i = ((q * n_iter as f64).ceil() as usize).clamp(1, n_iter) - 1;
        ns[i] as f64 / calls as f64
    };
    Stat {
        kernel,
        n_iter,
        calls_per_sample: calls,
        p50: pick(0.50),
        p90: pick(0.90),
        p99: pick(0.99),
        p999: pick(0.999),
        max: ns[n_iter - 1] as f64 / calls as f64,
    }
}

/// UTC calendar date from the system clock (days-from-civil inverse).
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

fn toolchain() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    let rs = rows();
    let mut stats: Vec<Stat> = Vec::new();
    const FAST: usize = 100_000;
    const K: usize = 16;

    // Closed-form kernels, classic API.
    stats.push(measure("bs_price", 10_000, FAST, K, |i| {
        let x = rs[i];
        bs_price(black_box(x.s), x.k, x.t, x.r, x.v, x.q, x.ot).unwrap_or(0.0)
    }));
    stats.push(measure("bs_greeks", 10_000, FAST, K, |i| {
        let x = rs[i];
        let g = bs_greeks(black_box(x.s), x.k, x.t, x.r, x.v, x.q, x.ot);
        g.map(|g| g.delta + g.gamma + g.vega + g.theta + g.rho + g.vanna + g.volga)
            .unwrap_or(0.0)
    }));
    stats.push(measure("black76_price", 10_000, FAST, K, |i| {
        let x = rs[i];
        black76_price(black_box(x.s), x.k, x.t, x.r, x.v, x.ot).unwrap_or(0.0)
    }));
    stats.push(measure("implied_vol", 10_000, 200_000, 1, |i| {
        let x = rs[i];
        implied_vol(black_box(x.p), x.s, x.k, x.t, x.r, x.q, x.ot).unwrap_or(0.0)
    }));
    for (name, ex, n, iters) in [
        ("crr_price n=100 european", Exercise::European, 100, 20_000),
        ("crr_price n=100 american", Exercise::American, 100, 20_000),
        ("crr_price n=1000 european", Exercise::European, 1000, 2_000),
        ("crr_price n=1000 american", Exercise::American, 1000, 2_000),
    ] {
        stats.push(measure(name, iters / 10, iters, 1, |i| {
            let x = rs[i];
            crr_price(
                black_box(x.s),
                x.k,
                x.t,
                x.r,
                x.v,
                x.q,
                OptionType::Put,
                ex,
                n,
            )
            .unwrap_or(0.0)
        }));
    }
    stats.push(measure("mc_price 10k paths anti+cv", 200, 2_000, 1, |i| {
        let x = rs[i];
        mc_price(
            black_box(x.s),
            x.k,
            x.t,
            x.r,
            x.v,
            x.q,
            x.ot,
            10_000,
            true,
            true,
            i as u64,
        )
        .map(|m| m.price)
        .unwrap_or(0.0)
    }));
    stats.push(measure("mc_price 1M paths anti+cv", 3, 30, 1, |i| {
        let x = rs[i];
        mc_price(
            black_box(x.s),
            x.k,
            x.t,
            x.r,
            x.v,
            x.q,
            x.ot,
            1_000_000,
            true,
            true,
            i as u64,
        )
        .map(|m| m.price)
        .unwrap_or(0.0)
    }));

    #[cfg(not(before))]
    new_api(&rs, &mut stats);

    let date = today();
    let tc = toolchain();
    let flags = "--release; opt-level=3, lto=fat, codegen-units=1";
    let cpu = "Intel Core 5 120U (12 threads), WSL2 on Windows 11";
    let mut json = String::from("[\n");
    for (i, s) in stats.iter().enumerate() {
        let _ = writeln!(
            json,
            "  {{\"engine\": \"rust/equity-options-engine\", \"kernel\": \"{}\", \
             \"toolchain\": \"{tc}\", \"flags\": \"{flags}\", \"cpu\": \"{cpu}\", \
             \"date\": \"{date}\", \"n_iter\": {}, \"calls_per_sample\": {}, \"unit\": \"ns\", \
             \"p50\": {:.1}, \"p90\": {:.1}, \"p99\": {:.1}, \"p999\": {:.1}, \"max\": {:.1}}}{}\n",
            s.kernel,
            s.n_iter,
            s.calls_per_sample,
            s.p50,
            s.p90,
            s.p99,
            s.p999,
            s.max,
            if i + 1 < stats.len() { "," } else { "" }
        );
    }
    json.push_str("]\n");
    print!("{json}");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench");
    let written = std::fs::create_dir_all(&dir)
        .and_then(|()| std::fs::write(dir.join("results.json"), &json));
    if let Err(e) = written {
        eprintln!("could not write bench/results.json: {e}");
    }
}

/// Kernels of the non-allocating API (absent from the previous version).
#[cfg(not(before))]
fn new_api(rs: &[Row], stats: &mut Vec<Stat>) {
    use eq_options_engine::fast::{
        bs_greeks_batch, bs_price_batch, try_bs_greeks, try_bs_price, try_crr_price,
        try_implied_vol, try_mc_price, BsBatch, GreeksOut,
    };
    use eq_options_engine::{CrrWorkspace, McWorkspace};

    const FAST: usize = 100_000;
    const K: usize = 16;
    stats.push(measure("try_bs_price", 10_000, FAST, K, |i| {
        let x = rs[i];
        try_bs_price(black_box(x.s), x.k, x.t, x.r, x.v, x.q, x.ot).unwrap_or(0.0)
    }));
    stats.push(measure("try_bs_greeks", 10_000, FAST, K, |i| {
        let x = rs[i];
        let g = try_bs_greeks(black_box(x.s), x.k, x.t, x.r, x.v, x.q, x.ot);
        g.map(|g| g.delta + g.gamma + g.vega + g.theta + g.rho + g.vanna + g.volga)
            .unwrap_or(0.0)
    }));
    stats.push(measure("try_implied_vol", 10_000, 200_000, 1, |i| {
        let x = rs[i];
        try_implied_vol(black_box(x.p), x.s, x.k, x.t, x.r, x.q, x.ot).unwrap_or(0.0)
    }));

    // Batch pricers: one sample = one 1,024-row batch; per-row figure.
    let col = |f: fn(&Row) -> f64| rs.iter().map(f).collect::<Vec<f64>>();
    let (s, k, t, r, v, q) = (
        col(|x| x.s),
        col(|x| x.k),
        col(|x| x.t),
        col(|x| x.r),
        col(|x| x.v),
        col(|x| x.q),
    );
    let ot: Vec<OptionType> = rs.iter().map(|x| x.ot).collect();
    let b = BsBatch {
        s: &s,
        k: &k,
        t: &t,
        r: &r,
        sigma: &v,
        q: &q,
        option_type: &ot,
    };
    let mut out = vec![0.0; N_ROWS];
    stats.push(per_row(measure(
        "bs_price_batch (per row)",
        200,
        5_000,
        1,
        |_| {
            let ok = bs_price_batch(black_box(&b), &mut out).is_ok();
            f64::from(u8::from(ok)) + out[0]
        },
    )));
    let mut cols = vec![vec![0.0; N_ROWS]; 8];
    stats.push(per_row(measure(
        "bs_greeks_batch (per row)",
        200,
        5_000,
        1,
        |_| {
            let [c0, c1, c2, c3, c4, c5, c6, c7] = &mut cols[..] else {
                return 0.0;
            };
            let mut go = GreeksOut {
                price: c0,
                delta: c1,
                gamma: c2,
                vega: c3,
                theta: c4,
                rho: c5,
                vanna: c6,
                volga: c7,
            };
            let ok = bs_greeks_batch(black_box(&b), &mut go).is_ok();
            f64::from(u8::from(ok)) + go.delta[0]
        },
    )));

    let mut ws = CrrWorkspace::with_steps(1000);
    for (name, ex, n, iters) in [
        (
            "try_crr_price n=100 european",
            Exercise::European,
            100,
            20_000,
        ),
        (
            "try_crr_price n=100 american",
            Exercise::American,
            100,
            20_000,
        ),
        (
            "try_crr_price n=1000 european",
            Exercise::European,
            1000,
            2_000,
        ),
        (
            "try_crr_price n=1000 american",
            Exercise::American,
            1000,
            2_000,
        ),
    ] {
        stats.push(measure(name, iters / 10, iters, 1, |i| {
            let x = rs[i];
            try_crr_price(
                black_box(x.s),
                x.k,
                x.t,
                x.r,
                x.v,
                x.q,
                OptionType::Put,
                ex,
                n,
                &mut ws,
            )
            .unwrap_or(0.0)
        }));
    }
    let mut mc = McWorkspace::new();
    for (name, n_paths, threads, warm, iters) in [
        (
            "try_mc_price 10k paths anti+cv 1 thread",
            10_000,
            1,
            200,
            2_000,
        ),
        (
            "try_mc_price 1M paths anti+cv 1 thread",
            1_000_000,
            1,
            3,
            30,
        ),
        (
            "try_mc_price 1M paths anti+cv 8 threads",
            1_000_000,
            8,
            3,
            30,
        ),
    ] {
        stats.push(measure(name, warm, iters, 1, |i| {
            let x = rs[i];
            try_mc_price(
                black_box(x.s),
                x.k,
                x.t,
                x.r,
                x.v,
                x.q,
                x.ot,
                n_paths,
                true,
                true,
                i as u64,
                threads,
                &mut mc,
            )
            .map(|m| m.price)
            .unwrap_or(0.0)
        }));
    }
}

/// Convert a whole-batch stat to a per-row stat.
#[cfg(not(before))]
fn per_row(mut s: Stat) -> Stat {
    let n = N_ROWS as f64;
    s.calls_per_sample = N_ROWS;
    s.p50 /= n;
    s.p90 /= n;
    s.p99 /= n;
    s.p999 /= n;
    s.max /= n;
    s
}
