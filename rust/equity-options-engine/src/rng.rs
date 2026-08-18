//! Deterministic, zero-dependency random number generation.
//!
//! # Algorithm choice (documented per the engine conventions)
//!
//! * **State generator: xoshiro256++** (Blackman & Vigna, 2019). 256-bit
//!   state, period 2^256 - 1, passes BigCrush, ~1 ns per `u64` — the same
//!   family NumPy uses for its default `Generator` bit stream. Chosen over
//!   `std`-only alternatives because Rust's standard library ships **no**
//!   RNG at all, and over a bare LCG/PCG32 because 64-bit output with a
//!   large state gives comfortable headroom for 10^6-10^8-path Monte
//!   Carlo without detectable correlation artefacts.
//! * **Seeding: SplitMix64** (Steele, Lea & Flood, 2014). A single `u64`
//!   seed is expanded through four SplitMix64 steps into the 256-bit
//!   xoshiro state. This is the seeding procedure recommended by the
//!   xoshiro authors: it guarantees a non-zero, well-mixed state for
//!   *every* seed including 0, and makes nearby seeds statistically
//!   independent.
//! * **Normal sampling: Box-Muller (trigonometric form)**, caching the
//!   second variate. Chosen over Ziggurat (large tables, harder to audit)
//!   and over inverse-CDF (needs a high-degree rational approximation
//!   whose tail accuracy would dominate the error budget). Box-Muller is
//!   exact in distribution, branch-free per pair, and trivially
//!   reproducible: consuming exactly two `u64` draws per pair makes the
//!   stream layout easy to reason about for bit-reproducibility tests.
//!
//! Same seed => bit-identical stream, on every platform (no
//! `std::collections::HashMap` iteration order, no threading, no libm
//! dispatch in the hot path beyond `ln`/`sqrt`/`sin`/`cos`, which are
//! correctly-rounded-enough and deterministic on a given target).

use std::f64::consts::PI;

/// SplitMix64 step: advances `state` and returns the next output.
///
/// Used only for seeding [`Xoshiro256PlusPlus`]; constants from Steele,
/// Lea & Flood (2014), "Fast splittable pseudorandom number generators".
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ pseudorandom generator with SplitMix64 seeding and a
/// cached Box-Muller normal sampler.
///
/// # Examples
///
/// ```
/// use eq_options_engine::rng::Xoshiro256PlusPlus;
/// let mut a = Xoshiro256PlusPlus::new(42);
/// let mut b = Xoshiro256PlusPlus::new(42);
/// // Same seed => bit-identical streams.
/// assert_eq!(a.next_u64(), b.next_u64());
/// assert_eq!(a.standard_normal().to_bits(), b.standard_normal().to_bits());
///
/// let u = Xoshiro256PlusPlus::new(7).next_uniform();
/// assert!(u > 0.0 && u < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Xoshiro256PlusPlus {
    state: [u64; 4],
    /// Cached second Box-Muller variate, if the last call produced a pair.
    cached_normal: Option<f64>,
}

impl Xoshiro256PlusPlus {
    /// Create a generator from a single `u64` seed via SplitMix64 expansion.
    ///
    /// Every seed (including 0) yields a valid, well-mixed non-zero state.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let state = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self {
            state,
            cached_normal: None,
        }
    }

    /// Next raw 64-bit output of xoshiro256++.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0]
            .wrapping_add(s[3])
            .rotate_left(23)
            .wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform draw in the **open** interval (0, 1) with 53-bit resolution.
    ///
    /// Uses the top 53 bits of [`next_u64`](Self::next_u64) mapped to the
    /// midpoints `(m + 0.5) * 2^-53`, so 0 and 1 are unattainable — safe
    /// to pass straight into `ln` for Box-Muller.
    #[inline]
    pub fn next_uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0) // 2^-53
    }

    /// Standard normal draw N(0, 1) via trigonometric Box-Muller.
    ///
    /// Generates variates in pairs `(R cos(theta), R sin(theta))` with
    /// `R = sqrt(-2 ln u1)`, `theta = 2 pi u2`, caching the second — so
    /// two calls consume exactly two `u64` draws.
    #[inline]
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.cached_normal = Some(radius * theta.sin());
        radius * theta.cos()
    }

    /// Fill `out` with i.i.d. standard normal draws.
    ///
    /// Equivalent to calling [`standard_normal`](Self::standard_normal)
    /// `out.len()` times; provided for tight Monte Carlo loops.
    pub fn fill_standard_normal(&mut self, out: &mut [f64]) {
        for z in out.iter_mut() {
            *z = self.standard_normal();
        }
    }
}
