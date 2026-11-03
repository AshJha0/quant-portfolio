//! Latency timing shared by the engine bench binaries.
//!
//! The method is the same as the C++ harness (`cpp/core/include/qp/bench/
//! latency.hpp`), so C++ and Rust figures are measured the same way:
//!
//! 1. Warm up with `warm_calls` untimed calls.
//! 2. Pick the batch size B: the smallest power of two, at most 1024, for
//!    which one batch of B back-to-back calls takes at least 2 us on the
//!    monotonic clock (`std::time::Instant`), judged on the fastest of 5
//!    trial batches.
//! 3. Take N = max(31, `call_budget` / B) samples; each sample is one batch
//!    divided by B, in ns per call. Nothing is subtracted.
//! 4. Report the median with p10 and p90 as the spread, plus p99, p99.9,
//!    max and mean, using the nearest-rank percentile (element
//!    ceil(q * N) - 1 of the sorted samples).
//!
//! When B > 1 every sample is the mean of B calls, so p99, p99.9 and max are
//! tails of B-call batch means, not of single calls: one 1 ms stall inside a
//! batch of 1024 adds only about 1 us to its sample. They are labelled
//! "p99 of B-call batch means" (the JSON field `tail_basis`, from
//! [`tail_basis`]) and must not be quoted as per-call tail latency. When
//! B = 1 the tails are per call.
//!
//! `f(i)` runs call number `i`; `i` counts every call, warm-up included, so
//! the caller can vary inputs by `i`.

use std::time::Instant;

/// Shortest batch, in ns, before B stops doubling.
pub const MIN_SAMPLE_NS: f64 = 2_000.0;
/// Largest batch size.
pub const MAX_BATCH: usize = 1024;
/// Trial batches per candidate size during calibration (fastest counts).
pub const CALIBRATION_TRIALS: usize = 5;
/// Fewest samples per kernel.
pub const MIN_SAMPLES: usize = 31;

/// Per-call latency summary in ns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    /// Number of samples N.
    pub samples: usize,
    /// Batch size B (calls per sample).
    pub calls_per_sample: usize,
    /// 10th percentile, ns per call.
    pub p10: f64,
    /// Median, ns per call.
    pub p50: f64,
    /// 90th percentile, ns per call.
    pub p90: f64,
    /// 99th percentile of the samples (of B-call batch means when B > 1).
    pub p99: f64,
    /// 99.9th percentile of the samples (of batch means when B > 1).
    pub p999: f64,
    /// Slowest sample, ns per call (a batch mean when B > 1).
    pub max: f64,
    /// Mean, ns per call.
    pub mean: f64,
}

/// What the tail figures describe, for the JSON field `tail_basis`.
///
/// `"per-call"` when `calls_per_sample` is 1, else
/// `"p99 of B-call batch means"` with B written out, the same wording as
/// the C++ harness.
#[must_use]
pub fn tail_basis(calls_per_sample: usize) -> String {
    if calls_per_sample == 1 {
        "per-call".to_string()
    } else {
        format!("p99 of {calls_per_sample}-call batch means")
    }
}

/// Nearest-rank percentile of sorted data: element ceil(q * n) - 1.
///
/// # Panics
/// If `sorted` is empty.
#[must_use]
pub fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let i = ((q * n as f64).ceil() as usize).clamp(1, n) - 1;
    sorted[i]
}

/// Summarise per-call samples (any order).
///
/// # Panics
/// If `ns` is empty.
#[must_use]
pub fn summarise(mut ns: Vec<f64>, calls_per_sample: usize) -> Summary {
    ns.sort_unstable_by(f64::total_cmp);
    let n = ns.len();
    Summary {
        samples: n,
        calls_per_sample,
        p10: nearest_rank(&ns, 0.10),
        p50: nearest_rank(&ns, 0.50),
        p90: nearest_rank(&ns, 0.90),
        p99: nearest_rank(&ns, 0.99),
        p999: nearest_rank(&ns, 0.999),
        max: ns[n - 1],
        mean: ns.iter().sum::<f64>() / n as f64,
    }
}

/// Time `f` with the shared method (see the module docs).
pub fn measure(warm_calls: usize, call_budget: usize, mut f: impl FnMut(usize)) -> Summary {
    let mut i = 0usize;
    for _ in 0..warm_calls {
        f(i);
        i += 1;
    }
    let mut batch_ns = |b: usize| -> f64 {
        let t0 = Instant::now();
        for _ in 0..b {
            f(std::hint::black_box(i));
            i += 1;
        }
        t0.elapsed().as_nanos() as f64
    };
    // The fastest of a few trial batches decides, so one preempted trial
    // cannot stop the doubling early.
    let mut b = 1usize;
    while b < MAX_BATCH
        && (0..CALIBRATION_TRIALS)
            .map(|_| batch_ns(b))
            .fold(f64::INFINITY, f64::min)
            < MIN_SAMPLE_NS
    {
        b *= 2;
    }
    let n = (call_budget / b).max(MIN_SAMPLES);
    let ns: Vec<f64> = (0..n).map(|_| batch_ns(b) / b as f64).collect();
    summarise(ns, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_basis_matches_cpp_wording() {
        assert_eq!(tail_basis(1), "per-call");
        assert_eq!(tail_basis(32), "p99 of 32-call batch means");
    }

    #[test]
    fn nearest_rank_matches_definition() {
        let v: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(nearest_rank(&v, 0.10), 1.0);
        assert_eq!(nearest_rank(&v, 0.50), 5.0);
        assert_eq!(nearest_rank(&v, 0.90), 9.0);
        assert_eq!(nearest_rank(&v, 1.0), 10.0);
        assert_eq!(nearest_rank(&v, 0.0), 1.0);
    }

    #[test]
    fn measure_counts_calls_and_bounds_batch() {
        let mut calls = 0usize;
        let s = measure(5, 100, |_| calls += 1);
        assert!(s.calls_per_sample.is_power_of_two() && s.calls_per_sample <= MAX_BATCH);
        assert!(s.samples >= MIN_SAMPLES);
        assert!(s.p10 <= s.p50 && s.p50 <= s.p90 && s.p90 <= s.max);
        assert!(calls >= 5 + s.samples * s.calls_per_sample);
    }

    #[test]
    fn slow_call_uses_batch_of_one() {
        let s = measure(0, 31, |_| {
            std::thread::sleep(std::time::Duration::from_micros(5))
        });
        assert_eq!(s.calls_per_sample, 1);
        assert_eq!(s.samples, 31);
        assert!(s.p50 >= 5_000.0);
    }
}
