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

/// Per-path counter-based stream for the Monte Carlo kernels (W4(b)).
///
/// Path `p` of a run with seed `s` draws from Philox4x32-10 stream
/// `(key = s, stream = p)` ([`qp_core::philox::PhiloxStream`]). Each 64-bit
/// word gives one uniform `((w >> 11) + 0.5) 2^-53` on (0, 1), and a normal
/// is [`norm_ppf_as241`] of that uniform. The path's `n` factor normals come
/// first, then, for Student-t, the chi-square mixing draw (Marsaglia-Tsang on
/// the same stream). A path's draws depend only on `(seed, p)`, so a run
/// gives the same bits on any number of threads. The C++ twin is
/// `eqvar::PathRng` with the same stream definition.
#[derive(Debug, Clone)]
pub struct PathRng {
    inner: qp_core::philox::PhiloxStream,
    seed: u64,
    path: u64,
    words: u64,
}

impl PathRng {
    /// Stream for path `path` under `seed`.
    #[inline]
    pub fn new(seed: u64, path: u64) -> Self {
        PathRng {
            inner: qp_core::philox::PhiloxStream::new(seed, path),
            seed,
            path,
            words: 0,
        }
    }

    /// Uniform on the open interval (0, 1), 53-bit resolution.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        self.words += 1;
        self.inner.uniform()
    }

    /// Standard normal by inversion: `norm_ppf_as241(uniform())`.
    #[inline]
    pub fn standard_normal(&mut self) -> f64 {
        norm_ppf_as241(self.uniform())
    }

    /// Fill `z` with standard normals: the same values, bit for bit, as
    /// calling [`PathRng::standard_normal`] `z.len()` times, computed in
    /// batches (Philox blocks side by side, then the central AS 241 branch
    /// for every lane and the tail branch only where `|p - 0.5| > 0.425`).
    pub fn fill_normals(&mut self, z: &mut [f64]) {
        if !self.words.is_multiple_of(2) {
            // Not block-aligned (only after an odd number of single draws):
            // take the plain path.
            for zi in z.iter_mut() {
                *zi = self.standard_normal();
            }
            return;
        }
        let mut block = self.words / 2;
        for chunk in z.chunks_mut(2 * LANES) {
            philox_uniforms(self.seed, self.path, block, chunk);
            block += LANES as u64;
            ppf_as241_batch(chunk);
        }
        self.words += z.len() as u64;
        self.inner.seek(self.words);
    }

    /// Chi-squared draw, `df > 2` in the callers (shape `df / 2 >= 1`).
    /// An invalid `df` returns NaN without consuming the stream.
    #[inline]
    pub fn chi_square(&mut self, df: f64) -> f64 {
        let shape = 0.5 * df;
        if !(shape >= 1.0) || !shape.is_finite() {
            return f64::NAN;
        }
        2.0 * marsaglia_tsang(self, shape)
    }
}

/// Philox blocks computed side by side in [`PathRng::fill_normals`].
const LANES: usize = 16;

/// Uniforms for words `2 * block ..` of stream `(seed, path)` into `out`
/// (at most `2 * LANES` values): Philox4x32-10 on `LANES` counters at once,
/// structure-of-arrays so the rounds vectorise. Same words as
/// [`qp_core::philox::PhiloxStream`] (a test checks it).
#[inline]
fn philox_uniforms(seed: u64, path: u64, block: u64, out: &mut [f64]) {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    let mut c0 = [path as u32; LANES];
    let mut c1 = [(path >> 32) as u32; LANES];
    let mut c2 = [0u32; LANES];
    let mut c3 = [0u32; LANES];
    for l in 0..LANES {
        let b = block.wrapping_add(l as u64);
        c2[l] = b as u32;
        c3[l] = (b >> 32) as u32;
    }
    let (mut k0, mut k1) = (seed as u32, (seed >> 32) as u32);
    for r in 0..10 {
        if r > 0 {
            k0 = k0.wrapping_add(W0);
            k1 = k1.wrapping_add(W1);
        }
        for l in 0..LANES {
            let p0 = M0 * u64::from(c0[l]);
            let p1 = M1 * u64::from(c2[l]);
            let n0 = ((p1 >> 32) as u32) ^ c1[l] ^ k0;
            let n2 = ((p0 >> 32) as u32) ^ c3[l] ^ k1;
            c0[l] = n0;
            c1[l] = p1 as u32;
            c2[l] = n2;
            c3[l] = p0 as u32;
        }
    }
    const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
    for (k, o) in out.iter_mut().enumerate() {
        let l = k / 2;
        let w = if k.is_multiple_of(2) {
            (u64::from(c1[l]) << 32) | u64::from(c0[l])
        } else {
            (u64::from(c3[l]) << 32) | u64::from(c2[l])
        };
        *o = ((w >> 11) as f64 + 0.5) * SCALE;
    }
}

/// In-place [`norm_ppf_as241`] of uniforms: the central rational for every
/// element (branch-free, so it vectorises), then the tail formula where
/// `|p - 0.5| > 0.425`. Bit-identical to the scalar function.
#[inline]
fn ppf_as241_batch(x: &mut [f64]) {
    let mut tail = false;
    let mut buf = [0.0f64; 2 * LANES];
    let p = &mut buf[..x.len()];
    p.copy_from_slice(x);
    for (xi, &pi) in x.iter_mut().zip(p.iter()) {
        let q = pi - 0.5;
        *xi = q * central_num(0.180625 - q * q) / central_den(0.180625 - q * q);
        tail |= q.abs() > 0.425;
    }
    if tail {
        for (xi, &pi) in x.iter_mut().zip(p.iter()) {
            if (pi - 0.5).abs() > 0.425 {
                *xi = norm_ppf_as241(pi);
            }
        }
    }
}

impl GammaSource for PathRng {
    #[inline]
    fn gamma_normal(&mut self) -> f64 {
        self.standard_normal()
    }
    #[inline]
    fn gamma_uniform(&mut self) -> f64 {
        self.uniform()
    }
}

/// Inverse standard normal CDF, Wichura (1988) Algorithm AS 241 (`PPND16`).
///
/// Relative accuracy about 1e-16 on (0, 1) with no refinement step; the
/// tests hold it to 3e-15 relative against an mpmath reference. `p` must be
/// in (0, 1) (the generator guarantees this; other inputs give NaN or
/// infinities). Operation order matches the C++ twin `eqvar::norm_ppf_as241`.
#[inline]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
pub fn norm_ppf_as241(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * central_num(r) / central_den(r);
    }
    tail_ppf(p, q)
}

#[inline(always)]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
fn central_num(r: f64) -> f64 {
    ((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r + 6.7265770927008700853e4)
        * r
        + 4.5921953931549871457e4)
        * r
        + 1.3731693765509461125e4)
        * r
        + 1.9715909503065514427e3)
        * r
        + 1.3314166789178437745e2)
        * r
        + 3.3871328727963666080e0
}

#[inline(always)]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
fn central_den(r: f64) -> f64 {
    ((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r + 3.9307895800092710610e4)
        * r
        + 2.1213794301586595867e4)
        * r
        + 5.3941960214247511077e3)
        * r
        + 6.8718700749205790830e2)
        * r
        + 4.2313330701600911252e1)
        * r
        + 1.0
}

/// The two tail branches of AS 241 (`|q| > 0.425`).
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
fn tail_ppf(p: f64, q: f64) -> f64 {
    let mut r = if q < 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        r -= 1.6;
        let num = ((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r
            + 2.41780725177450611770e-1)
            * r
            + 1.27045825245236838258e0)
            * r
            + 3.64784832476320460504e0)
            * r
            + 5.76949722146069140550e0)
            * r
            + 4.63033784615654529590e0)
            * r
            + 1.42343711074968357734e0;
        let den = ((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r
            + 1.51986665636164571966e-2)
            * r
            + 1.48103976427480074590e-1)
            * r
            + 6.89767334985100004550e-1)
            * r
            + 1.67638483018380384940e0)
            * r
            + 2.05319162663775882187e0)
            * r
            + 1.0;
        num / den
    } else {
        r -= 5.0;
        let num = ((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r
            + 1.24266094738807843860e-3)
            * r
            + 2.65321895265761230930e-2)
            * r
            + 2.96560571828504891230e-1)
            * r
            + 1.78482653991729133580e0)
            * r
            + 5.46378491116411436990e0)
            * r
            + 6.65790464350110377720e0;
        let den = ((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r
            + 1.84631831751005468180e-5)
            * r
            + 7.86869131145613259100e-4)
            * r
            + 1.48753612908506148525e-2)
            * r
            + 1.36929880922735805310e-1)
            * r
            + 5.99832206555887937690e-1)
            * r
            + 1.0;
        num / den
    };
    if q < 0.0 { -val } else { val }
}

#[cfg(test)]
mod path_rng_tests {
    use super::*;

    /// AS 241 against mpmath (bisection on mp.ncdf at 50 digits).
    #[test]
    #[allow(clippy::excessive_precision)]
    fn as241_matches_mpmath() {
        let refs: [(f64, f64); 31] = [
            (1e-300, -37.047096299361199237),
            (1e-100, -21.273453560965324294),
            (1e-20, -9.2623400897984075796),
            (1e-12, -7.0344838253011319326),
            (1e-08, -5.6120012441747887279),
            (1e-05, -4.2648907939228246102),
            (0.001, -3.0902323061678135354),
            (0.0125, -2.2414027276049453536),
            (0.025, -1.9599639845400542118),
            (0.05, -1.644853626951472688),
            (0.075, -1.4395314709384559349),
            (0.1, -1.2815515655446004353),
            (0.2, -0.84162123357291416552),
            (0.3, -0.52440051270804081597),
            (0.4, -0.25334710313579974132),
            (0.45, -0.12566134685507400616),
            (0.49, -0.025068908258711058033),
            (0.5, 0.0),
            (0.51, 0.025068908258711058033),
            (0.6, 0.25334710313579974132),
            (0.7, 0.52440051270804065631),
            (0.75, 0.6744897501960817432),
            (0.8, 0.8416212335729143638),
            (0.9, 1.2815515655446005935),
            (0.95, 1.6448536269514722843),
            (0.975, 1.9599639845400538556),
            (0.99, 2.3263478740408407676),
            (0.999, 3.0902323061678132778),
            (0.99999, 4.2648907939238407699),
            (0.9999999999990905, 7.0477002566644087254),
            (0.9999999999999999, 8.2095361516013868556),
        ];
        for (p, x) in refs {
            let got = norm_ppf_as241(p);
            assert!(
                (got - x).abs() <= 3e-15 * x.abs().max(1e-300),
                "p={p}: {got} vs {x}"
            );
        }
        // Symmetry is exact through the 1 - p branch only to rounding.
        assert_eq!(norm_ppf_as241(0.5), 0.0);
    }

    #[test]
    fn batched_fill_equals_scalar_draws() {
        for &len in &[1usize, 2, 3, 31, 32, 33, 50, 97] {
            for path in [0u64, 5, u64::MAX] {
                let mut z = vec![0.0; len];
                let mut a = PathRng::new(0xDEAD_BEEF_1234, path);
                a.fill_normals(&mut z);
                let mut s = qp_core::philox::PhiloxStream::new(0xDEAD_BEEF_1234, path);
                for zi in &z {
                    assert_eq!(norm_ppf_as241(s.uniform()).to_bits(), zi.to_bits());
                }
                // The stream continues where the batch stopped.
                assert_eq!(a.uniform().to_bits(), s.uniform().to_bits());
                // An unaligned start (odd number of single draws) agrees too.
                let mut b = PathRng::new(9, path);
                let _ = b.uniform();
                let mut zb = vec![0.0; len];
                b.fill_normals(&mut zb);
                let mut t = qp_core::philox::PhiloxStream::new(9, path);
                let _ = t.uniform();
                for zi in &zb {
                    assert_eq!(norm_ppf_as241(t.uniform()).to_bits(), zi.to_bits());
                }
            }
        }
    }

    #[test]
    fn path_streams_are_independent_of_order() {
        let mut a = PathRng::new(7, 3);
        let mut z = [0.0; 9];
        a.fill_normals(&mut z);
        let mut b = PathRng::new(7, 3);
        for zi in z {
            assert_eq!(b.standard_normal().to_bits(), zi.to_bits());
        }
        assert_ne!(PathRng::new(7, 4).standard_normal(), z[0]);
        assert!(PathRng::new(7, 1).chi_square(f64::NAN).is_nan());
        assert!(PathRng::new(7, 1).chi_square(1.0).is_nan());
    }
}
