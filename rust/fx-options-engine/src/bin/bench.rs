//! Throughput benchmarks for the FX options engine.
//!
//! 1. 1,000,000 Garman–Kohlhagen vanilla prices (varying strike/vol so
//!    the compiler cannot hoist the computation).
//! 2. Strike-from-delta solves across the four quoting conventions
//!    (analytic for spot/forward, Brent for premium-adjusted).
//! 3. One 1,000,000-path Monte Carlo price (antithetic + control variate,
//!    xoshiro256** + inverse-CDF normals, single-threaded).
//!
//! Run: `cargo run --release --bin bench`

use std::time::Instant;

use fx_options_engine::{
    gk_price, mc_price, strike_from_delta, DeltaConvention, OptionType,
};

fn main() {
    let (s, t, rd, rf) = (1.10, 0.5, 0.0425, 0.0290);

    // ---- 1M GK prices ---------------------------------------------------
    {
        const N: usize = 1_000_000;
        let mut acc = 0.0;
        let t0 = Instant::now();
        for i in 0..N {
            let k = 0.90 + 0.40 * (i % 1000) as f64 / 1000.0;
            let sig = 0.06 + 0.10 * (i % 97) as f64 / 97.0;
            let ty = if i % 2 == 0 {
                OptionType::Call
            } else {
                OptionType::Put
            };
            acc += gk_price(s, k, t, rd, rf, sig, ty).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "GK vanilla pricing : {N} prices in {dt:.3} s -> {:.2} M prices/s (checksum {acc:.6})",
            N as f64 / dt / 1e6
        );
    }

    // ---- strike-from-delta solves ---------------------------------------
    {
        const N_PER_CONV: usize = 25_000;
        let convs = [
            (DeltaConvention::Spot, "spot"),
            (DeltaConvention::Forward, "forward"),
            (DeltaConvention::SpotPa, "spot_pa"),
            (DeltaConvention::ForwardPa, "forward_pa"),
        ];
        for (conv, name) in convs {
            let mut acc = 0.0;
            let t0 = Instant::now();
            for i in 0..N_PER_CONV {
                let d = 0.05 + 0.40 * (i % 100) as f64 / 100.0;
                let sig = 0.07 + 0.08 * (i % 53) as f64 / 53.0;
                acc += strike_from_delta(d, s, t, rd, rf, sig, OptionType::Call, conv).unwrap();
                acc += strike_from_delta(-d, s, t, rd, rf, sig, OptionType::Put, conv).unwrap();
            }
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "strike-from-delta  : {name:<10} {} solves in {dt:.3} s -> {:.2} k solves/s (checksum {acc:.4})",
                2 * N_PER_CONV,
                (2 * N_PER_CONV) as f64 / dt / 1e3
            );
        }
    }

    // ---- 1M-path Monte Carlo --------------------------------------------
    {
        const N_PATHS: u64 = 1_000_000;
        let t0 = Instant::now();
        let r = mc_price(s, 1.12, t, rd, rf, 0.0925, OptionType::Call, N_PATHS, 0, true, true)
            .unwrap();
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "Monte Carlo        : {N_PATHS} paths in {dt:.3} s -> {:.2} M paths/s (price {:.8}, SE {:.2e}, {})",
            N_PATHS as f64 / dt / 1e6,
            r.price,
            r.std_error,
            r.method
        );
    }
}
