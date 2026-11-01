//! Micro-benchmark for the pricing kernels (std::time::Instant, no deps).
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --bin bench
//! ```
//!
//! Reports single-run wall times; results feed the README and
//! docs/VALIDATION.md tables. `std::hint::black_box` prevents the
//! optimiser from deleting the priced work.

use std::hint::black_box;
use std::time::Instant;

use eq_options_engine::{
    bs_greeks, bs_price, crr_price, implied_vol, mc_price, Exercise, OptionType,
};

fn main() {
    println!("eq-options-engine bench (release, single run, 1 thread)");
    println!("{}", "-".repeat(72));

    // 1) 1,000,000 analytic Black-Scholes prices over a parameter sweep.
    {
        const N: usize = 1_000_000;
        let start = Instant::now();
        let mut acc = 0.0_f64;
        for i in 0..N {
            let x = (i % 1000) as f64;
            let s = 50.0 + 0.1 * x;
            let k = 100.0;
            let t = 0.05 + 0.001 * x;
            let sigma = 0.1 + 0.0004 * x;
            let ot = if i % 2 == 0 {
                OptionType::Call
            } else {
                OptionType::Put
            };
            acc += bs_price(s, k, t, 0.03, sigma, 0.01, ot).unwrap();
        }
        black_box(acc);
        let dt = start.elapsed();
        let ms = dt.as_secs_f64() * 1e3;
        println!(
            "Black-Scholes analytic     | {:>9} prices | {:8.1} ms | {:6.1}M prices/sec",
            N,
            ms,
            N as f64 / dt.as_secs_f64() / 1e6
        );
    }

    // 2) 100,000 full analytic Greek sets.
    {
        const N: usize = 100_000;
        let start = Instant::now();
        let mut acc = 0.0_f64;
        for i in 0..N {
            let x = (i % 1000) as f64;
            let g = bs_greeks(
                80.0 + 0.04 * x,
                100.0,
                0.5 + 0.001 * x,
                0.03,
                0.15 + 0.0002 * x,
                0.01,
                OptionType::Call,
            )
            .unwrap();
            acc += g.delta + g.gamma + g.vega + g.theta + g.rho + g.vanna + g.volga;
        }
        black_box(acc);
        let dt = start.elapsed();
        println!(
            "Full analytic Greek set    | {:>9} evals  | {:8.1} ms | {:6.1}M evals/sec",
            N,
            dt.as_secs_f64() * 1e3,
            N as f64 / dt.as_secs_f64() / 1e6
        );
    }

    // 3) CRR tree, 1000 steps, European and American.
    for (label, exercise) in [
        ("CRR tree n=1000, European", Exercise::European),
        ("CRR tree n=1000, American", Exercise::American),
    ] {
        const REPS: usize = 100;
        let start = Instant::now();
        let mut acc = 0.0_f64;
        for i in 0..REPS {
            let s = 95.0 + 0.1 * (i as f64);
            acc += crr_price(s, 100.0, 1.0, 0.05, 0.2, 0.02, OptionType::Put, exercise, 1000)
                .unwrap();
        }
        black_box(acc);
        let dt = start.elapsed();
        let per_tree_ms = dt.as_secs_f64() * 1e3 / REPS as f64;
        println!(
            "{label}  | {:>9} trees  | {:8.2} ms/tree | {:6.0} trees/sec",
            REPS,
            per_tree_ms,
            1e3 / per_tree_ms
        );
    }

    // 4) Monte Carlo, 1,000,000 paths, antithetic + control variate.
    {
        const N: usize = 1_000_000;
        let start = Instant::now();
        let res = mc_price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.2,
            0.0,
            OptionType::Call,
            N,
            true,
            true,
            42,
        )
        .unwrap();
        black_box(res.price);
        let dt = start.elapsed();
        println!(
            "MC 1M paths (anti + CV)    | {:>9} paths  | {:8.1} ms | {:6.1}M paths/sec",
            N,
            dt.as_secs_f64() * 1e3,
            N as f64 / dt.as_secs_f64() / 1e6
        );
        println!(
            "    price {:.6}  se {:.6}  ci [{:.6}, {:.6}]  (BS: 10.450584)",
            res.price, res.std_error, res.ci_low, res.ci_high
        );
    }
    // Implied vol: 8 quotes (ATM/OTM/ITM, 1 week to 1 year), 100,000 solves.
    {
        let cases = [
            (100.0, 100.0, 1.0, 0.2),
            (100.0, 120.0, 0.5, 0.25),
            (100.0, 80.0, 0.25, 0.3),
            (100.0, 90.0, 0.02, 0.1),
        ];
        let mut quotes = Vec::new();
        for &(s, k, t, v) in &cases {
            for ot in [OptionType::Call, OptionType::Put] {
                quotes.push((bs_price(s, k, t, 0.03, v, 0.0, ot).unwrap(), s, k, t, ot));
            }
        }
        const N: usize = 100_000;
        let start = Instant::now();
        let mut acc = 0.0_f64;
        for i in 0..N {
            let (p, s, k, t, ot) = quotes[i % quotes.len()];
            acc += implied_vol(black_box(p), s, k, t, 0.03, 0.0, ot).unwrap_or(0.0);
        }
        black_box(acc);
        let dt = start.elapsed();
        println!(
            "Implied vol                | {:>9} solves | {:8.1} ms | {:6.0} ns/solve",
            N,
            dt.as_secs_f64() * 1e3,
            dt.as_secs_f64() * 1e9 / N as f64
        );
    }
}
