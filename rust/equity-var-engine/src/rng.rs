//! Deterministic pseudo-random number generation (zero-dependency).
//!
//! [`Rng`] is a xoshiro256++ generator (Blackman & Vigna, 2019) seeded via
//! SplitMix64 from a single explicit `u64` seed. On a given platform a
//! fixed seed reproduces the **exact bit pattern** of every draw — the
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

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Deterministic xoshiro256++ generator with Box–Muller normal caching.
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
    state: [u64; 4],
    cached_normal: Option<f64>,
}

#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

impl Rng {
    /// Create a generator from an explicit seed.
    ///
    /// The 256-bit xoshiro state is expanded from `seed` with SplitMix64,
    /// the recommended seeding procedure (avoids the all-zero state and
    /// decorrelates nearby seeds).
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in &mut state {
            *s = splitmix64(&mut sm);
        }
        if state == [0, 0, 0, 0] {
            state[0] = 1; // unreachable in practice; xoshiro forbids all-zero
        }
        Rng { state, cached_normal: None }
    }

    /// Next raw 64-bit output of xoshiro256++.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = rotl(s[0].wrapping_add(s[3]), 23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        result
    }

    /// Uniform draw on the **open** interval `(0, 1)` with 53-bit
    /// resolution: `(top53bits + 0.5) * 2^-53`. Strictly positive so it is
    /// always safe under `ln`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal draw via the Box–Muller transform.
    ///
    /// Each transform produces a `(cos, sin)` pair from two uniforms; the
    /// second variate is cached, so consumption of the underlying uniform
    /// stream is fixed and deterministic.
    #[inline]
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TWO_PI * u2;
        self.cached_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Gamma(shape, scale = 1) draw, `shape > 0`.
    ///
    /// Marsaglia–Tsang (2000) squeeze-rejection for `shape >= 1`; the
    /// standard boost `Gamma(a) = Gamma(a + 1) * U^{1/a}` for `shape < 1`.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        debug_assert!(shape > 0.0, "gamma shape must be > 0");
        if shape < 1.0 {
            let g = self.gamma(shape + 1.0);
            let u = self.uniform();
            return g * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.standard_normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.uniform();
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v3;
            }
            if u.ln() < 0.5 * x2 + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }

    /// Chi-squared draw with `df > 0` degrees of freedom:
    /// `chi2(df) = 2 * Gamma(df / 2)`.
    #[inline]
    pub fn chi_square(&mut self, df: f64) -> f64 {
        debug_assert!(df > 0.0, "chi-square df must be > 0");
        2.0 * self.gamma(0.5 * df)
    }
}
