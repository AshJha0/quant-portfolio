//! Per-call latency harness.
//!
//! Plain idea: time each call on its own, put every sample in a histogram,
//! and report tail percentiles. Two traps are handled explicitly.
//!
//! 1. Timer cost and reordering. On x86-64 the start stamp is
//!    `lfence; rdtsc; lfence` and the end stamp is `rdtscp; lfence`, so the
//!    measured code cannot drift outside the stamps (Paoloni, "How to
//!    Benchmark Code Execution Times on Intel IA-32 and IA-64 Instruction Set
//!    Architectures", Intel white paper 324264, 2010, uses the same fence
//!    discipline with `cpuid`; `lfence` is the lighter serialiser Intel
//!    documents for `rdtsc` in the SDM vol. 2B). Ticks are converted to ns
//!    against `Instant` over a calibration window. Elsewhere `Instant` is used.
//! 2. Coordinated omission (Gil Tene, "How NOT to Measure Latency", 2015).
//!    A closed loop that waits for each call before issuing the next one
//!    stops issuing while the system is stalled, so the stall is recorded
//!    once instead of for every request that would have queued behind it.
//!    `run_fixed_rate` issues call `i` at its intended time `t0 + i*period`
//!    and records `end - intended`, which charges queueing delay to every
//!    delayed call. It also records the naive `end - start` so the two can be
//!    compared.
//!
//! Formal statement for the stall test: with period `P`, one stall of
//! length `S >> P` every `K` calls, and negligible service time, the CO-aware
//! latencies after a stall are about `S - jP` for `j = 0..S/P`, so the
//! fraction of calls above `x` is `(S - x) / (P K)` for `0 <= x <= S`, and the
//! q-quantile is `S - (1 - q) P K`. The naive histogram sees one slow call per
//! `K`, so its quantiles above `1 - 1/K` only show the stall.
//!
//! Worked example: `S` = 1 ms, `P` = 10 us, `K` = 10,000. CO-aware p99.9 is
//! 1 ms - 0.001 x 10 us x 10,000 = 0.9 ms; naive p99.9 shows only the service
//! time, and the stall appears in the naive histogram only above p99.99. The
//! test `stall_shows_at_the_right_rate_sim` checks this on a simulated clock
//! exactly. `stall_real_clock` (ignored by default; run locally) repeats it on
//! the real clock, where host noise only adds latency, so it checks a lower
//! bound and prints the uninjected baseline beside it.

#![warn(missing_docs)]

pub mod env;

use std::time::Instant;

/// Log-linear histogram in the HdrHistogram layout (Tene): values below
/// 2^11 are exact; above that each power-of-two range is split into 1,024
/// equal sub-buckets, so relative error is below 2^-10 (about 0.1%).
/// All memory is allocated in `new`; `record` never allocates.
#[derive(Clone, Debug)]
pub struct Histogram {
    counts: Box<[u64]>,
    total: u64,
    max: u64,
    min: u64,
    sum: u128,
}

const SUB: u64 = 2048;
const HALF: u64 = 1024;
const SUB_BITS: u32 = 11;

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Empty histogram covering 0..2^63.
    pub fn new() -> Self {
        let n = Self::index(u64::MAX >> 1) + 1;
        Histogram {
            counts: vec![0; n].into_boxed_slice(),
            total: 0,
            max: 0,
            min: u64::MAX,
            sum: 0,
        }
    }

    #[inline]
    fn index(v: u64) -> usize {
        if v < SUB {
            return v as usize;
        }
        let shift = (64 - v.leading_zeros()) - SUB_BITS;
        (SUB + (shift as u64 - 1) * HALF + ((v >> shift) - HALF)) as usize
    }

    /// Largest value that maps to bucket `i`.
    fn upper(i: usize) -> u64 {
        let i = i as u64;
        if i < SUB {
            return i;
        }
        let shift = (i - SUB) / HALF + 1;
        let m = (i - SUB) % HALF + HALF;
        ((m + 1) << shift) - 1
    }

    /// Records one value (clamped to 2^63 - 1).
    #[inline]
    pub fn record(&mut self, v: u64) {
        let v = v.min(u64::MAX >> 1);
        self.counts[Self::index(v)] += 1;
        self.total += 1;
        self.max = self.max.max(v);
        self.min = self.min.min(v);
        self.sum += v as u128;
    }

    /// Sample count.
    pub fn count(&self) -> u64 {
        self.total
    }

    /// Largest recorded value.
    pub fn max(&self) -> u64 {
        self.max
    }

    /// Mean of recorded values.
    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.sum as f64 / self.total as f64
        }
    }

    /// Value at percentile `p` in [0, 100]: the upper edge of the bucket that
    /// holds the sample of rank `ceil(p/100 * n)` (rank at least 1), capped at
    /// the recorded max. This matches HdrHistogram's `getValueAtPercentile`
    /// reporting of the highest equivalent value.
    pub fn percentile(&self, p: f64) -> u64 {
        if self.total == 0 {
            return 0;
        }
        // The 1e-9 guard stops 0.999 * 1000 = 999.0000000000001 rounding up to 1000.
        let rank = ((p * self.total as f64) / 100.0 - 1e-9).ceil().max(1.0) as u64;
        let mut seen = 0;
        for (i, c) in self.counts.iter().enumerate() {
            seen += c;
            if seen >= rank {
                return Self::upper(i).min(self.max);
            }
        }
        self.max
    }

    /// Adds another histogram's samples.
    pub fn merge(&mut self, o: &Histogram) {
        for (a, b) in self.counts.iter_mut().zip(o.counts.iter()) {
            *a += b;
        }
        self.total += o.total;
        self.max = self.max.max(o.max);
        self.min = self.min.min(o.min);
        self.sum += o.sum;
    }
}

/// A nanosecond clock the harness can wait on.
pub trait Clock {
    /// Current time, ns.
    fn now(&mut self) -> u64;
    /// Returns once `now() >= t` (busy-poll for the real clock).
    fn wait_until(&mut self, t: u64);
}

/// Real clock: TSC with fences on x86-64, `Instant` elsewhere.
#[derive(Debug)]
pub struct RealClock {
    origin: Instant,
    #[allow(dead_code)]
    tsc0: u64,
    /// ns per TSC tick (1.0 when the TSC is not used).
    pub ns_per_tick: f64,
    /// True when stamps come from the TSC.
    pub uses_tsc: bool,
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn tsc_start() -> u64 {
    use std::arch::x86_64::{_mm_lfence, _rdtsc};
    // SAFETY: lfence and rdtsc are available on every x86-64 CPU.
    unsafe {
        _mm_lfence();
        let t = _rdtsc();
        _mm_lfence();
        t
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn tsc_end() -> u64 {
    use std::arch::x86_64::{__rdtscp, _mm_lfence};
    let mut aux = 0u32;
    // SAFETY: rdtscp is present on every x86-64 CPU sold since about 2006;
    // `RealClock::calibrate` is the only constructor and runs it first.
    unsafe {
        let t = __rdtscp(&mut aux);
        _mm_lfence();
        t
    }
}

impl RealClock {
    /// Calibrates the TSC against `Instant` over `window_ms`.
    pub fn calibrate(window_ms: u64) -> Self {
        let origin = Instant::now();
        #[cfg(target_arch = "x86_64")]
        {
            let t0 = tsc_start();
            let i0 = Instant::now();
            while i0.elapsed().as_millis() < window_ms as u128 {
                std::hint::spin_loop();
            }
            let t1 = tsc_end();
            let ns = i0.elapsed().as_nanos() as f64;
            RealClock {
                origin,
                tsc0: t0,
                ns_per_tick: ns / (t1 - t0) as f64,
                uses_tsc: true,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = window_ms;
            RealClock {
                origin,
                tsc0: 0,
                ns_per_tick: 1.0,
                uses_tsc: false,
            }
        }
    }

    /// Start stamp (ns).
    #[inline(always)]
    pub fn start(&self) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            ((tsc_start() - self.tsc0) as f64 * self.ns_per_tick) as u64
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.origin.elapsed().as_nanos() as u64
        }
    }

    /// End stamp (ns).
    #[inline(always)]
    pub fn end(&self) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            ((tsc_end() - self.tsc0) as f64 * self.ns_per_tick) as u64
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.origin.elapsed().as_nanos() as u64
        }
    }

    /// Seconds since calibration, from `Instant` (for wall-clock checks).
    pub fn wall_secs(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

impl Clock for RealClock {
    #[inline(always)]
    fn now(&mut self) -> u64 {
        self.end()
    }
    #[inline]
    fn wait_until(&mut self, t: u64) {
        while self.end() < t {
            std::hint::spin_loop();
        }
    }
}

/// Simulated clock for exact harness tests: time moves only when told.
#[derive(Debug, Default)]
pub struct SimClock {
    /// Current time, ns.
    pub t: u64,
}

impl Clock for SimClock {
    fn now(&mut self) -> u64 {
        self.t
    }
    fn wait_until(&mut self, t: u64) {
        self.t = self.t.max(t);
    }
}

/// Results of a fixed-rate run.
#[derive(Clone, Debug)]
pub struct RateRun {
    /// `end - intended start`: coordinated-omission-aware.
    pub co: Histogram,
    /// `end - actual start`: what a closed loop would report.
    pub naive: Histogram,
}

/// Issues `n` calls at fixed `period_ns`, call `i` intended at
/// `t0 + i * period_ns`; a late call is issued at once (no catch-up skipping),
/// and its queueing delay is charged to the CO-aware histogram.
pub fn run_fixed_rate<C: Clock>(
    clock: &mut C,
    n: u64,
    period_ns: u64,
    mut op: impl FnMut(u64, &mut C),
) -> RateRun {
    let mut co = Histogram::new();
    let mut naive = Histogram::new();
    let t0 = clock.now();
    for i in 0..n {
        let intended = t0 + i * period_ns;
        clock.wait_until(intended);
        let s = clock.now();
        op(i, clock);
        let e = clock.now();
        co.record(e - intended);
        naive.record(e - s);
    }
    RateRun { co, naive }
}

/// Measures the cost of an empty timed region `n` times (timer overhead).
pub fn timer_overhead(c: &RealClock, n: u64) -> Histogram {
    let mut h = Histogram::new();
    for _ in 0..n {
        let s = c.start();
        let e = c.end();
        h.record(e.saturating_sub(s));
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_contiguous_and_tight() {
        let mut last = 0usize;
        for v in (0..5_000_000u64).step_by(7).chain([u64::MAX >> 1]) {
            let i = Histogram::index(v);
            assert!(i >= last);
            last = i;
            let up = Histogram::upper(i);
            assert!(up >= v, "upper {up} < value {v}");
            assert!(
                (up - v) as f64 <= (v as f64) / 1024.0 + 1.0,
                "v={v} up={up}"
            );
            if i > 0 {
                assert!(Histogram::upper(i - 1) < v);
            }
        }
    }

    #[test]
    fn percentiles_exact_below_2048() {
        let mut h = Histogram::new();
        for v in 1..=1000 {
            h.record(v);
        }
        assert_eq!(h.percentile(50.0), 500);
        assert_eq!(h.percentile(99.9), 999);
        assert_eq!(h.percentile(100.0), 1000);
        assert_eq!(h.percentile(0.0), 1);
        assert!((h.mean() - 500.5).abs() < 1e-9);
    }

    /// The worked example on a simulated clock. Service time is 50 ns.
    #[test]
    fn stall_shows_at_the_right_rate_sim() {
        let (s, p, k) = (1_000_000u64, 10_000u64, 10_000u64);
        let mut clk = SimClock::default();
        let r = run_fixed_rate(&mut clk, 50 * k, p, |i, c| {
            c.t += 50;
            if i % k == k / 2 {
                c.t += s;
            }
        });
        // CO-aware: q-quantile = S - (1-q) P K (+ service time, bucket width).
        let want_999 = s - (p * k) / 1000 + 50;
        let got = r.co.percentile(99.9);
        assert!(
            got.abs_diff(want_999) <= want_999 / 1000 + 10_000,
            "co p99.9 {got} vs {want_999}"
        );
        let want_99 = s.saturating_sub((p * k) / 100) + 50; // = 50 ns: stalls fill exactly 1%
        assert!(
            r.co.percentile(99.0) <= want_99 + p,
            "co p99 {}",
            r.co.percentile(99.0)
        );
        // Naive: only one call in K is slow.
        assert_eq!(r.naive.percentile(99.9), 50);
        assert!(r.naive.percentile(99.995) >= s);
        assert_eq!(r.co.count(), 50 * k);
    }

    /// Same check on the real clock. Timing-sensitive, so ignored by default
    /// and run locally. A real host adds its own stalls (scheduler
    /// preemption, interrupts) which the CO-aware histogram also charges, so
    /// the assertion is one-sided: the injected stall must lift p99.9 to at
    /// least 85% of the model value, and the naive p99.9 must not see it.
    /// The run without injection is printed as the host baseline.
    #[test]
    #[ignore]
    fn stall_real_clock() {
        let (s, p, k) = (1_000_000u64, 10_000u64, 10_000u64);
        let mut clk = RealClock::calibrate(50);
        let base = run_fixed_rate(&mut clk, 30 * k, p, |_, _| {});
        let r = run_fixed_rate(&mut clk, 30 * k, p, |i, c| {
            if i % k == k / 2 {
                let t = c.now() + s;
                c.wait_until(t);
            }
        });
        let got = r.co.percentile(99.9) as f64;
        let want = (s - p * k / 1000) as f64;
        eprintln!(
            "real clock: baseline co p99.9 = {} ns; stalled co p99.9 = {got} ns (model {want}); naive p99.9 = {} ns; naive max = {} ns",
            base.co.percentile(99.9),
            r.naive.percentile(99.9),
            r.naive.max()
        );
        assert!(got >= 0.85 * want, "co p99.9 {got} vs model {want}");
        assert!(r.naive.percentile(99.9) < 100_000);
    }
}
