//! Deterministic pseudo-random number generation (no external dependencies).
//!
//! The engine owns its RNG so that Monte Carlo results repeat exactly for a
//! given seed on the same build and platform (the integer stream is portable;
//! the float transforms use the platform libm), independent of any
//! third-party crate:
//!
//! * core stream: xoshiro256++ (Blackman & Vigna 2019), seeded through
//!   SplitMix64 so that any `u64` seed - including 0 - yields a well-mixed
//!   state;
//! * uniforms: 53-bit mantissa doubles; the open-interval variant never
//!   returns exactly 0 or 1, so inverse-CDF transforms are always finite;
//! * normals: inverse-CDF transform through
//!   [`crate::stats::inv_norm_cdf`] (pure arithmetic - no platform-dependent
//!   rejection loops in the normal path);
//! * gamma / chi-square: Marsaglia-Tsang squeeze (deterministic given the
//!   stream), used by the Student-t mixing variable in Monte Carlo.
//!
//! Every stochastic component in the crate takes an explicit seed
//! (portfolio convention: no hidden global RNG state).

use crate::stats::inv_norm_cdf_unchecked;
use qp_core::rng::{GammaSource, Xoshiro256PlusPlus, marsaglia_tsang};

/// xoshiro256++ deterministic generator with SplitMix64 seeding.
///
/// Two generators built with the same seed produce bitwise-identical
/// streams; this is asserted by the Monte Carlo determinism tests. The
/// integer stream is `qp_core::rng::Xoshiro256PlusPlus`.
#[derive(Clone, Debug)]
pub struct Rng {
    inner: Xoshiro256PlusPlus,
}

impl GammaSource for Rng {
    #[inline]
    fn gamma_normal(&mut self) -> f64 {
        self.normal()
    }
    #[inline]
    fn gamma_uniform(&mut self) -> f64 {
        self.uniform_open()
    }
}

impl Rng {
    /// Create a generator from a 64-bit seed (any value is valid).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Rng {
            inner: Xoshiro256PlusPlus::new(seed),
        }
    }

    /// Next raw 64-bit output of the xoshiro256++ stream.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Uniform double in `[0, 1)` with full 53-bit mantissa resolution.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        self.inner.next_uniform_closed_open()
    }

    /// Uniform double in the *open* interval `(0, 1)` - safe for inverse-CDF
    /// transforms (never returns an endpoint).
    #[inline]
    pub fn uniform_open(&mut self) -> f64 {
        self.inner.next_uniform()
    }

    /// Standard normal draw via the inverse-CDF transform.
    ///
    /// One uniform per normal; |CDF error| of the transform < 1e-9
    /// (see [`crate::stats::inv_norm_cdf`]), far below Monte Carlo noise.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        inv_norm_cdf_unchecked(self.uniform_open())
    }

    /// Gamma(shape, scale=1) draw by Marsaglia-Tsang (2000).
    ///
    /// Valid for any `shape > 0`; shapes below 1 use the standard boosting
    /// identity `Gamma(a) = Gamma(a+1) * U^{1/a}`.
    ///
    /// An invalid `shape` (NaN, infinite, or `<= 0`) returns NaN without
    /// consuming the stream.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if !(shape > 0.0) || !shape.is_finite() {
            return f64::NAN;
        }
        if shape < 1.0 {
            // Stream order: the boost uniform first, then the Gamma(shape + 1) draw.
            let u = self.uniform_open();
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        marsaglia_tsang(self, shape)
    }

    /// Chi-square draw with `df > 0` degrees of freedom (`2 * Gamma(df/2)`).
    #[inline]
    pub fn chi_square(&mut self, df: f64) -> f64 {
        2.0 * self.gamma(0.5 * df)
    }
}
