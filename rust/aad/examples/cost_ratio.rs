//! AAD cost ratio: time for price plus all Greeks over time for the price.
//!
//! `cargo run --release --example cost_ratio`. Best of `REPS` timings each;
//! the plan's bar (W19) is "all Greeks under 5x one PV".

use qp_aad::bs;
use qp_aad::mc::{self, Payoff};
use qp_aad::swap::{self, Swap};
use qp_rates_curves::{FixedPeriod, FloatPeriod};
use std::hint::black_box;
use std::time::Instant;

const REPS: usize = 21;

fn best(mut f: impl FnMut(), inner: usize) -> f64 {
    (0..REPS)
        .map(|_| {
            let t0 = Instant::now();
            for _ in 0..inner {
                f();
            }
            t0.elapsed().as_secs_f64() / inner as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let b = [100.0, 100.0, 1.0, 0.05, 0.02, 0.20];
    let p = best(
        || {
            black_box(bs::price(
                black_box(b[0]),
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
                1.0,
            ));
        },
        100_000,
    );
    let g = best(
        || {
            black_box(bs::greeks(
                black_box(b[0]),
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
                true,
            ));
        },
        100_000,
    );
    println!(
        "bs     price {:>10.3} us  aad {:>10.3} us  ratio {:.2}",
        p * 1e6,
        g * 1e6,
        g / p
    );

    let steps: usize = std::env::args().nth(1).map_or(52, |a| a.parse().unwrap());
    let z = mc::normals(20_000, steps, 1);
    for payoff in [Payoff::European, Payoff::Asian] {
        let p = best(
            || {
                black_box(mc::price(
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    b[4],
                    b[5],
                    black_box(&z),
                    steps,
                    payoff,
                ));
            },
            3,
        );
        let g = best(
            || {
                black_box(mc::greeks(
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    b[4],
                    b[5],
                    black_box(&z),
                    steps,
                    payoff,
                ));
            },
            3,
        );
        println!(
            "mc {payoff:?} price {:>10.3} ms  aad {:>10.3} ms  ratio {:.2}",
            p * 1e3,
            g * 1e3,
            g / p
        );
    }

    // 10Y annual swap on an 18-pillar curve shaped like the USD OIS fixture.
    let days: Vec<i64> = vec![
        3, 13, 39, 98, 173, 264, 355, 446, 557, 739, 1104, 1469, 1833, 2565, 3660, 5487, 7314,
        10966,
    ];
    let dfs: Vec<f64> = days
        .iter()
        .map(|&d| (-0.04 * d as f64 / 365.0).exp())
        .collect();
    let fixed: Vec<FixedPeriod> = (1..=10)
        .map(|y| FixedPeriod {
            pay: 365 * y + 5,
            accrual: 1.0139,
        })
        .collect();
    let floating: Vec<FloatPeriod> = (1..=10)
        .map(|y| FloatPeriod {
            start: 365 * (y - 1),
            end: 365 * y,
            pay: 365 * y + 5,
            accrual: 1.0139,
        })
        .collect();
    let sw = Swap {
        fixed,
        floating,
        rate: 0.04,
        notional: 1e6,
        payer: true,
    };
    let p = best(
        || {
            black_box(swap::pv_f64(&days, black_box(&dfs), &sw));
        },
        100_000,
    );
    let g = best(
        || {
            black_box(swap::pillar_greeks(&days, black_box(&dfs), &sw));
        },
        100_000,
    );
    println!(
        "swap   pv    {:>10.3} us  aad {:>10.3} us  ratio {:.2}",
        p * 1e6,
        g * 1e6,
        g / p
    );
}
