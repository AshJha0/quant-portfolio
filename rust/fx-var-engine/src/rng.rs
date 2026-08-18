//! Deterministic pseudo-random number generation (no external dependencies).
//!
//! The engine owns its RNG so that Monte Carlo results are **bit-reproducible**
//! for a given seed, independent of any third-party crate's version churn:
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

use crate::stats::inv_norm_cdf;

/// xoshiro256++ deterministic generator with SplitMix64 seeding.
///
/// Two generators built with the same seed produce bitwise-identical
/// streams; this is asserted by the Monte Carlo determinism tests.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Create a generator from a 64-bit seed (any value is valid).
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Rng { s }
    }

    /// Next raw 64-bit output of the xoshiro256++ stream.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform double in `[0, 1)` with full 53-bit mantissa resolution.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform double in the *open* interval `(0, 1)` - safe for inverse-CDF
    /// transforms (never returns an endpoint).
    #[inline]
    pub fn uniform_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal draw via the inverse-CDF transform.
    ///
    /// One uniform per normal; |CDF error| of the transform < 1e-9
    /// (see [`crate::stats::inv_norm_cdf`]), far below Monte Carlo noise.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        inv_norm_cdf(self.uniform_open())
    }

    /// Gamma(shape, scale=1) draw by Marsaglia-Tsang (2000).
    ///
    /// Valid for any `shape > 0`; shapes below 1 use the standard boosting
    /// identity `Gamma(a) = Gamma(a+1) * U^{1/a}`.
    ///
    /// # Panics
    /// Panics if `shape <= 0` (programming error, not a data error).
    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(shape > 0.0, "gamma shape must be > 0");
        if shape < 1.0 {
            let u = self.uniform_open();
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.uniform_open();
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v3;
            }
            if u.ln() < 0.5 * x2 + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }

    /// Chi-square draw with `df > 0` degrees of freedom (`2 * Gamma(df/2)`).
    #[inline]
    pub fn chi_square(&mut self, df: f64) -> f64 {
        2.0 * self.gamma(0.5 * df)
    }
}
