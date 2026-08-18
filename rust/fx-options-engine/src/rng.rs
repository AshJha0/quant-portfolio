//! Deterministic, dependency-free random number generation.
//!
//! * [`SplitMix64`] — Steele/Lea/Flood's 64-bit mixer, used to expand a
//!   single `u64` seed into the xoshiro state (the seeding procedure
//!   recommended by the xoshiro authors).
//! * [`Xoshiro256StarStar`] — Blackman/Vigna's xoshiro256** 1.0, a fast
//!   all-purpose generator with 256 bits of state and a 2^256 - 1 period.
//! * Standard normal draws by **inverse-CDF transform** of a 53-bit
//!   uniform ([`crate::norm_ppf`]).  Unlike Box–Muller or ziggurat
//!   variants, the inverse-CDF map consumes exactly one `u64` per normal
//!   and involves no rejection loops, so streams are **bit-reproducible**
//!   across platforms and releases — the property the Monte Carlo engine
//!   (and its regression tests) rely on.
//!
//! ```
//! use fx_options_engine::rng::Xoshiro256StarStar;
//! let mut a = Xoshiro256StarStar::new(42);
//! let mut b = Xoshiro256StarStar::new(42);
//! assert_eq!(a.next_u64(), b.next_u64());       // same seed, same stream
//! let z = a.next_normal();
//! assert!(z.is_finite());
//! assert_ne!(Xoshiro256StarStar::new(1).next_u64(),
//!            Xoshiro256StarStar::new(2).next_u64());
//! ```

use crate::norm_ppf;

/// SplitMix64 (Steele, Lea & Flood 2014): a tiny 64-bit generator whose
/// single-pass output mixing makes it the standard seed expander for
/// xoshiro/xoroshiro state initialisation.
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
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// xoshiro256** 1.0 (Blackman & Vigna 2018), seeded via [`SplitMix64`].
///
/// Passes BigCrush; period 2^256 - 1.  Used by the Monte Carlo engine
/// with an explicit `u64` seed for bit-reproducible pricing runs.
#[derive(Debug, Clone)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Create a generator from a `u64` seed, expanding it into the
    /// 256-bit state with [`SplitMix64`] (the authors' recommendation —
    /// guarantees a non-zero state for every seed in practice).
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()];
        if s == [0, 0, 0, 0] {
            // The all-zero state is the one fixed point of the generator;
            // unreachable via SplitMix64 in practice, guarded regardless.
            s[0] = 0x9E37_79B9_7F4A_7C15;
        }
        Self { s }
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform draw strictly inside `(0, 1)`: the top 53 bits of a
    /// [`Self::next_u64`] offset by half an ULP, `u = (x >> 11 + 0.5) *
    /// 2^-53` — never exactly 0 or 1, so [`crate::norm_ppf`] is always
    /// finite.
    #[inline]
    pub fn next_open01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal draw by inverse-CDF transform of
    /// [`Self::next_open01`] — exactly one `u64` consumed per draw,
    /// bit-reproducible across platforms.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        norm_ppf(self.next_open01())
    }
}
