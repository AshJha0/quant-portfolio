//! Deterministic, dependency-free random number generation.
//!
//! * [`SplitMix64`]: Steele/Lea/Flood's 64-bit mixer, used to expand a
//!   single `u64` seed into the xoshiro state (the seeding procedure
//!   recommended by the xoshiro authors).
//! * [`Xoshiro256StarStar`]: Blackman/Vigna's xoshiro256** 1.0, a fast
//!   all-purpose generator with 256 bits of state and a 2^256 - 1 period.
//! * Standard normal draws by **inverse-CDF transform** of a 53-bit
//!   uniform ([`crate::norm_ppf`]).  Unlike Box–Muller or ziggurat
//!   variants, the inverse-CDF map consumes exactly one `u64` per normal
//!   and involves no rejection loops, so the `u64` stream is identical on
//!   every platform; the normals go through `ln`/`exp` from the platform
//!   libm and repeat exactly on a given target.
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

pub use qp_core::rng::SplitMix64;

/// xoshiro256** 1.0 (Blackman & Vigna 2018), seeded via [`SplitMix64`].
///
/// Passes BigCrush; period 2^256 - 1. Used by the Monte Carlo engine with
/// an explicit `u64` seed for reproducible pricing runs. The integer
/// generator is `qp_core::rng::Xoshiro256StarStar`; this wrapper adds the
/// inverse-CDF normal draw.
#[derive(Debug, Clone)]
pub struct Xoshiro256StarStar(qp_core::rng::Xoshiro256StarStar);

impl Xoshiro256StarStar {
    /// Create a generator from a `u64` seed, expanding it into the
    /// 256-bit state with [`SplitMix64`]. Every seed gives a non-zero
    /// state (see `qp_core::rng`).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self(qp_core::rng::Xoshiro256StarStar::new(seed))
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// Uniform draw strictly inside `(0, 1)`: the top 53 bits of a
    /// [`Self::next_u64`] offset by half an ULP, `u = (x >> 11 + 0.5) *
    /// 2^-53`: never exactly 0 or 1, so [`crate::norm_ppf`] is always
    /// finite.
    #[inline]
    pub fn next_open01(&mut self) -> f64 {
        self.0.next_open01()
    }

    /// Standard normal draw by inverse-CDF transform of
    /// [`Self::next_open01`]: exactly one `u64` consumed per draw,
    /// reproducible on a given target.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        norm_ppf(self.next_open01())
    }
}
