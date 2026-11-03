//! Per-call latency of the feed handler: parse + arbitrate + apply, per
//! Binance diff-depth message, on a seeded generated day.
//!
//! Software in-process latency, not wire-to-wire. Figures from a laptop or a
//! hosted runner are shared-machine indicative only; the CPU model is printed
//! with every run.
//!
//! Method (the same as the W30 per-call harness): every call is timed on its
//! own and every sample is kept, then sorted for exact percentiles. Two
//! loads are run. Closed loop: the next call starts when the last one ends
//! (`end - start`). Fixed rate: call `i` is due at `t0 + i * period` and its
//! latency is `end - due`, so a stall is charged to every call queued behind
//! it (Gil Tene, "How NOT to Measure Latency", 2015, coordinated omission).
//! The timer is `std::time::Instant`; its own cost is measured and printed so
//! it can be subtracted by the reader. Seam: at integration this switches to
//! `qp_ull_bench` (rdtscp/lfence clock and the shared histogram) so the Rust
//! and C++ figures come out of the same code path.
//!
//! Usage: `feed-bench [messages] [period_ns]` (defaults 200000, 20000).

use std::hint::black_box;
use std::time::Instant;

use qp_ull_feed::generator::Generator;
use qp_ull_feed::{Handler, Venue};

fn pct(sorted: &[u64], q: f64) -> u64 {
    let i = ((sorted.len() as f64 * q).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[i]
}

fn report(name: &str, mut v: Vec<u64>) {
    v.sort_unstable();
    println!(
        "{name:<28} n={:<8} p50={:>7} p90={:>7} p99={:>7} p99.9={:>8} max={:>9} ns",
        v.len(),
        pct(&v, 0.50),
        pct(&v, 0.90),
        pct(&v, 0.99),
        pct(&v, 0.999),
        v[v.len() - 1]
    );
}

fn cpu_model() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo")
        && let Some(l) = s.lines().find(|l| l.starts_with("model name"))
    {
        return l.split(':').nth(1).unwrap_or("").trim().to_string();
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let period: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    println!("software in-process latency, not wire-to-wire; shared-machine indicative");
    println!("cpu: {}", cpu_model());
    println!("messages: {n}, fixed-rate period: {period} ns, seed 1");

    let mut g = Generator::new(1, 1);
    let snap = g.snapshot_json();
    let msgs: Vec<Vec<u8>> = (0..n).map(|_| g.next_event().raw.into_bytes()).collect();

    let mut t = Vec::with_capacity(100_000);
    for _ in 0..100_000 {
        let a = Instant::now();
        t.push(black_box(Instant::now()).duration_since(a).as_nanos() as u64);
    }
    report("timer overhead", t);

    let fresh = || {
        let mut h = Handler::new(Venue::BinanceDiffDepth, 8, 4096);
        h.on_snapshot(snap.as_bytes());
        h
    };

    // Warm-up pass (caches, branch predictors, map nodes).
    let mut h = fresh();
    for m in &msgs {
        h.on_stream(0, m);
    }
    assert!(!h.is_stale());

    let mut h = fresh();
    let mut v = Vec::with_capacity(n);
    for m in &msgs {
        let a = Instant::now();
        black_box(h.on_stream(0, black_box(m)));
        v.push(a.elapsed().as_nanos() as u64);
    }
    assert!(!h.is_stale());
    report("closed loop (end-start)", v);

    let mut h = fresh();
    let mut v = Vec::with_capacity(n);
    let t0 = Instant::now();
    for (i, m) in msgs.iter().enumerate() {
        let due = i as u64 * period;
        while (t0.elapsed().as_nanos() as u64) < due {
            std::hint::spin_loop();
        }
        black_box(h.on_stream(0, black_box(m)));
        v.push((t0.elapsed().as_nanos() as u64).saturating_sub(due));
    }
    report("fixed rate (end-due, CO)", v);
}
