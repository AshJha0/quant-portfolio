//! Deterministic, zero-dependency random number generation.
//!
//! Rust's standard library ships no RNG, so the engines carry their own.
//! This module holds the pieces they share:
//!
//! * [`splitmix64`] and [`SplitMix64`]: Steele, Lea & Flood (2014). Used to
//!   expand one `u64` seed into a 256-bit xoshiro state, as the xoshiro
//!   authors recommend. Every seed, including 0, gives a well-mixed state.
//! * [`Xoshiro256PlusPlus`]: Blackman & Vigna (2019), period 2^256 - 1,
//!   with a cached trigonometric Box-Muller normal sampler. The equity
//!   options engine uses it directly; both VaR engines wrap it.
//! * [`Xoshiro256StarStar`]: Blackman & Vigna (2018), the `**` scrambler.
//!   The FX options engine wraps it and draws normals by inverse CDF.
//! * [`uniform_from_bits`], [`uniform_closed_open_from_bits`] and
//!   [`box_muller`]: the integer-to-float maps the engines use.
//! * [`marsaglia_tsang`]: the gamma sampler loop for `shape >= 1`, generic
//!   over where its normals and uniforms come from.
//!
//! # Streams
//!
//! The four engines draw from two different integer streams for the same
//! seed. The equity options, equity VaR and FX VaR engines use xoshiro256++.
//! The FX options engine uses xoshiro256**. Both are seeded by four
//! SplitMix64 steps. Each engine keeps the stream it had before this crate
//! existed, so no published number moved.
//!
//! Same seed gives the same `u64` stream on every platform. Normals go
//! through `ln`, `sin`, `cos` (or `exp`) from the platform libm, which is
//! deterministic on one target but not correctly rounded, so normals can
//! differ in the last bits between targets.
//!
//! # All-zero state
//!
//! xoshiro must not start from the all-zero state. Four consecutive
//! SplitMix64 outputs cannot all be zero: the output mix is a bijection
//! and the four internal states are distinct, so at most one output is
//! zero. No guard is needed, and none of the engines' old guards could
//! ever fire.

use std::f64::consts::PI;

/// Golden-ratio increment of SplitMix64.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// One SplitMix64 step: advances `state` and returns the next output.
///
/// Constants from Steele, Lea & Flood (2014), "Fast splittable
/// pseudorandom number generators".
///
/// # Examples
///
/// ```
/// let mut s = 0u64;
/// assert_eq!(qp_core::rng::splitmix64(&mut s), 0xE220_A839_7B1D_CDAF);
/// ```
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Expand one seed into a 256-bit xoshiro state with four SplitMix64 steps.
#[inline]
pub fn seed_state(seed: u64) -> [u64; 4] {
    let mut sm = seed;
    [
        splitmix64(&mut sm),
        splitmix64(&mut sm),
        splitmix64(&mut sm),
        splitmix64(&mut sm),
    ]
}

/// SplitMix64 as a generator object.
///
/// # Examples
///
/// ```
/// use qp_core::rng::{splitmix64, SplitMix64};
/// let mut g = SplitMix64::new(7);
/// let mut s = 7u64;
/// assert_eq!(g.next_u64(), splitmix64(&mut s));
/// ```
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from a `u64` seed (any value is valid).
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.state)
    }
}

/// Map a raw `u64` to the open interval (0, 1): the top 53 bits mapped to
/// the midpoints `(m + 0.5) * 2^-53`. Never 0 or 1, so it is safe to pass
/// to `ln` or to an inverse CDF.
#[inline]
pub fn uniform_from_bits(x: u64) -> f64 {
    ((x >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0) // 2^-53
}

/// Map a raw `u64` to `[0, 1)`: the top 53 bits times `2^-53`.
#[inline]
pub fn uniform_closed_open_from_bits(x: u64) -> f64 {
    (x >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0) // 2^-53
}

/// Trigonometric Box-Muller: `(R cos(theta), R sin(theta))` with
/// `R = sqrt(-2 ln u1)` and `theta = 2 pi u2`.
#[inline]
pub fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

/// xoshiro256++ with SplitMix64 seeding and a cached Box-Muller normal.
///
/// Box-Muller is exact in distribution and consumes exactly two `u64`
/// draws per pair of normals, which keeps the stream layout easy to reason
/// about in bit-reproducibility tests.
///
/// # Examples
///
/// ```
/// use qp_core::rng::Xoshiro256PlusPlus;
/// let mut a = Xoshiro256PlusPlus::new(42);
/// let mut b = Xoshiro256PlusPlus::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
/// assert_eq!(a.standard_normal().to_bits(), b.standard_normal().to_bits());
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
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed_state(seed),
            cached_normal: None,
        }
    }

    /// Next raw 64-bit output of xoshiro256++.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform draw in the open interval (0, 1); see [`uniform_from_bits`].
    #[inline]
    pub fn next_uniform(&mut self) -> f64 {
        uniform_from_bits(self.next_u64())
    }

    /// Uniform draw in `[0, 1)`; see [`uniform_closed_open_from_bits`].
    #[inline]
    pub fn next_uniform_closed_open(&mut self) -> f64 {
        uniform_closed_open_from_bits(self.next_u64())
    }

    /// Standard normal draw by trigonometric Box-Muller, caching the
    /// second variate: two calls consume exactly two `u64` draws.
    #[inline]
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let (z0, z1) = box_muller(u1, u2);
        self.cached_normal = Some(z1);
        z0
    }

    /// Fill `out` with standard normals; the same as calling
    /// [`standard_normal`](Self::standard_normal) `out.len()` times.
    pub fn fill_standard_normal(&mut self, out: &mut [f64]) {
        for z in out.iter_mut() {
            *z = self.standard_normal();
        }
    }

    /// Fill `out` with raw [`next_u64`](Self::next_u64) outputs.
    ///
    /// Used by multi-threaded Monte Carlo: the raw stream is drawn
    /// serially, then the float transforms run in parallel on fixed
    /// chunks. The stream is the one `standard_normal` would consume.
    #[inline]
    pub fn fill_u64(&mut self, out: &mut [u64]) {
        for x in out.iter_mut() {
            *x = self.next_u64();
        }
    }
}

/// xoshiro256** 1.0 (Blackman & Vigna 2018), seeded via SplitMix64.
///
/// # Examples
///
/// ```
/// use qp_core::rng::Xoshiro256StarStar;
/// let mut a = Xoshiro256StarStar::new(42);
/// let mut b = Xoshiro256StarStar::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
/// let u = a.next_open01();
/// assert!(u > 0.0 && u < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Create a generator from a `u64` seed via SplitMix64 expansion.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            s: seed_state(seed),
        }
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform draw strictly inside (0, 1); see [`uniform_from_bits`].
    #[inline]
    pub fn next_open01(&mut self) -> f64 {
        uniform_from_bits(self.next_u64())
    }
}

/// Source of the normals and uniforms that [`marsaglia_tsang`] consumes.
///
/// The engines differ in how they draw a normal (Box-Muller or inverse
/// CDF), so the sampler takes both through this trait and the stream
/// layout of each engine is unchanged.
pub trait GammaSource {
    /// One standard normal draw.
    fn gamma_normal(&mut self) -> f64;
    /// One uniform draw on (0, 1).
    fn gamma_uniform(&mut self) -> f64;
}

/// Marsaglia-Tsang (2000) squeeze-rejection draw of `Gamma(shape, 1)` for
/// `shape >= 1`.
///
/// The caller handles `shape < 1` (by the boost
/// `Gamma(a) = Gamma(a + 1) U^{1/a}`) and invalid shapes; the order in
/// which it draws the boost uniform is part of its stream layout.
#[inline]
pub fn marsaglia_tsang<R: GammaSource>(r: &mut R, shape: f64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = r.gamma_normal();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = r.gamma_uniform();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v3;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Outputs for state {1, 2, 3, 4}, computed with an independent
    /// Python transcription of the authors' C code (prng.di.unimi.it).
    #[test]
    fn reference_vectors_from_c_code() {
        let mut pp = Xoshiro256PlusPlus {
            state: [1, 2, 3, 4],
            cached_normal: None,
        };
        let want_pp = [41943041u64, 58720359, 3588806011781223, 3591011842654386];
        for w in want_pp {
            assert_eq!(pp.next_u64(), w);
        }
        let mut ss = Xoshiro256StarStar { s: [1, 2, 3, 4] };
        let want_ss = [11520u64, 0, 1509978240, 1215971899390074240];
        for w in want_ss {
            assert_eq!(ss.next_u64(), w);
        }
    }

    #[test]
    fn splitmix_reference_seed_zero() {
        // First outputs of SplitMix64 from seed 0 (Vigna's splitmix64.c).
        let mut g = SplitMix64::new(0);
        assert_eq!(g.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(g.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(g.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn uniform_maps_hit_documented_endpoints() {
        assert_eq!(uniform_from_bits(0), 0.5 * 2f64.powi(-53));
        assert_eq!(uniform_from_bits(u64::MAX), 1.0 - 0.5 * 2f64.powi(-53));
        assert_eq!(uniform_closed_open_from_bits(0), 0.0);
        assert_eq!(
            uniform_closed_open_from_bits(u64::MAX),
            1.0 - 2f64.powi(-53)
        );
    }

    #[test]
    fn normal_pairs_use_two_draws() {
        let mut a = Xoshiro256PlusPlus::new(3);
        let mut b = Xoshiro256PlusPlus::new(3);
        let z0 = a.standard_normal();
        let z1 = a.standard_normal();
        let (w0, w1) = box_muller(b.next_uniform(), b.next_uniform());
        assert_eq!((z0.to_bits(), z1.to_bits()), (w0.to_bits(), w1.to_bits()));
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
