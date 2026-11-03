//! Deterministic pseudo-random number generation (zero-dependency).
//!
//! [`Rng`] is a xoshiro256++ generator (Blackman & Vigna, 2019) seeded via
//! SplitMix64 from a single explicit `u64` seed. On a given platform a
//! fixed seed reproduces the **exact bit pattern** of every draw: the
//! property the Monte Carlo module relies on for regression-testable VaR
//! (`tests/test_monte_carlo.rs` asserts bitwise equality across runs).
//!
//! Variate generation:
//!
//! * uniforms on the open interval `(0, 1)` with 53-bit resolution;
//! * standard normals by the Box–Muller transform (pair-cached);
//! * gamma variates by Marsaglia–Tsang squeeze rejection, which yields
//!   chi-squared draws for the multivariate Student-t mixing variable.
//!
//! Why not a `rand` crate dependency: the engine's contract is that risk
//! numbers are reproducible across crate-version upgrades. Owning the ~80
//! lines of generator code freezes the stream definition forever and keeps
//! the crate std-only (see docs/METHODOLOGY.md).

use qp_core::rng::{GammaSource, Xoshiro256PlusPlus, marsaglia_tsang};

/// Deterministic xoshiro256++ generator with Box–Muller normal caching.
///
/// The stream and the normal sampler are `qp_core::rng::Xoshiro256PlusPlus`;
/// this type adds the gamma and chi-squared draws.
///
/// # Examples
///
/// ```
/// use eq_var_engine::rng::Rng;
/// let mut a = Rng::new(42);
/// let mut b = Rng::new(42);
/// assert_eq!(a.next_u64(), b.next_u64()); // bit-reproducible
/// let u = a.uniform();
/// assert!(u > 0.0 && u < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Rng {
    inner: Xoshiro256PlusPlus,
}

impl GammaSource for Rng {
    #[inline]
    fn gamma_normal(&mut self) -> f64 {
        self.standard_normal()
    }
    #[inline]
    fn gamma_uniform(&mut self) -> f64 {
        self.uniform()
    }
}

impl Rng {
    /// Create a generator from an explicit seed.
    ///
    /// The 256-bit xoshiro state is expanded from `seed` with SplitMix64,
    /// the recommended seeding procedure (avoids the all-zero state and
    /// decorrelates nearby seeds).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Rng {
            inner: Xoshiro256PlusPlus::new(seed),
        }
    }

    /// Next raw 64-bit output of xoshiro256++.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Uniform draw on the **open** interval `(0, 1)` with 53-bit
    /// resolution: `(top53bits + 0.5) * 2^-53`. Strictly positive so it is
    /// always safe under `ln`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        self.inner.next_uniform()
    }

    /// Standard normal draw via the Box–Muller transform.
    ///
    /// Each transform produces a `(cos, sin)` pair from two uniforms; the
    /// second variate is cached, so consumption of the underlying uniform
    /// stream is fixed and deterministic.
    #[inline]
    pub fn standard_normal(&mut self) -> f64 {
        self.inner.standard_normal()
    }

    /// Gamma(shape, scale = 1) draw, `shape > 0`.
    ///
    /// Marsaglia–Tsang (2000) squeeze-rejection for `shape >= 1`; the
    /// standard boost `Gamma(a) = Gamma(a + 1) * U^{1/a}` for `shape < 1`.
    ///
    /// An invalid `shape` (NaN, infinite, or `<= 0`) returns NaN without
    /// consuming the stream. It never panics and never loops.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if !(shape > 0.0) || !shape.is_finite() {
            return f64::NAN;
        }
        if shape < 1.0 {
            // Stream order: the Gamma(shape + 1) draw first, then the boost uniform.
            let g = self.gamma(shape + 1.0);
            let u = self.uniform();
            return g * u.powf(1.0 / shape);
        }
        marsaglia_tsang(self, shape)
    }

    /// Chi-squared draw with `df > 0` degrees of freedom:
    /// `chi2(df) = 2 * Gamma(df / 2)`.
    #[inline]
    pub fn chi_square(&mut self, df: f64) -> f64 {
        2.0 * self.gamma(0.5 * df)
    }
}
