//! Per-path counter-based random numbers for Monte Carlo (W4(b)), shared
//! by the equity and FX VaR engines.
//!
//! Path `p` of a run with seed `s` reads Philox4x32-10 stream
//! `(key = s, stream = p)` ([`crate::philox::PhiloxStream`]); a uniform is
//! `((w >> 11) + 0.5) 2^-53` of one 64-bit word and a normal is Wichura's
//! AS 241 inverse CDF of one uniform (Applied Statistics 37 (1988) 477-484).
//! A path's draws depend only on `(s, p)`, so a run gives the same bits on
//! any thread count. The C++ twin is `qp/path_rng.hpp`.
//!
//! Moved here from `eq_var_engine::rng` (Round 8 perf-fix lane) so the FX
//! engine draws from the identical stream definition.

#![allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail these checks

use crate::philox::PhiloxStream;
use crate::rng::{GammaSource, marsaglia_tsang};

/// Per-path counter-based stream for the Monte Carlo kernels (W4(b)).
///
/// Path `p` of a run with seed `s` draws from Philox4x32-10 stream
/// `(key = s, stream = p)` ([`PhiloxStream`]). Each 64-bit
/// word gives one uniform `((w >> 11) + 0.5) 2^-53` on (0, 1), and a normal
/// is [`norm_ppf_as241`] of that uniform. The path's `n` factor normals come
/// first, then, for Student-t, the chi-square mixing draw (Marsaglia-Tsang on
/// the same stream). A path's draws depend only on `(seed, p)`, so a run
/// gives the same bits on any number of threads. The C++ twin is
/// `eqvar::PathRng` with the same stream definition.
#[derive(Debug, Clone)]
pub struct PathRng {
    inner: PhiloxStream,
    seed: u64,
    path: u64,
    words: u64,
}

impl PathRng {
    /// Stream for path `path` under `seed`.
    #[inline]
    pub fn new(seed: u64, path: u64) -> Self {
        PathRng {
            inner: PhiloxStream::new(seed, path),
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
pub const LANES: usize = 16;

/// Uniforms for words `2 * block ..` of stream `(seed, path)` into `out`
/// (at most `2 * LANES` values): Philox4x32-10 on `LANES` counters at once,
/// structure-of-arrays so the rounds vectorise. Same words as
/// [`PhiloxStream`] (a test checks it).
#[inline]
pub fn philox_uniforms(seed: u64, path: u64, block: u64, out: &mut [f64]) {
    let w = philox_words(seed, path, block);
    for (o, &wk) in out.iter_mut().zip(w.iter()) {
        *o = word_to_uniform(wk);
    }
}

/// Resampling indices `min(n - 1, floor(u n))` for the `2 * LANES` words
/// `2 * block ..` of stream `(seed, path)`, with `u` from
/// [`word_to_uniform`]: the same indices, bit for bit, as
/// `((u * n as f64) as usize).min(n - 1)` (a test checks it), but without
/// the saturating float-to-integer cast, so the loop vectorises. `n >= 1`.
#[inline]
pub fn philox_indices(seed: u64, path: u64, block: u64, n: usize, out: &mut [usize; 2 * LANES]) {
    let nf = n as f64;
    let last = n.saturating_sub(1);
    if nf < MAGIC_LIMIT {
        let [c0, c1, c2, c3] = philox_lanes(seed, path, block);
        let mut even = [0usize; LANES];
        let mut odd = [0usize; LANES];
        for l in 0..LANES {
            even[l] = (floor_small(uniform_from_halves(c1[l], c0[l]) * nf) as usize).min(last);
            odd[l] = (floor_small(uniform_from_halves(c3[l], c2[l]) * nf) as usize).min(last);
        }
        for (l, pair) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair[0] = even[l];
            pair[1] = odd[l];
        }
    } else {
        let w = philox_words(seed, path, block);
        for (o, &wk) in out.iter_mut().zip(w.iter()) {
            *o = ((word_to_uniform(wk) * nf) as usize).min(last);
        }
    }
}

/// [`word_to_uniform`] of the word `(hi << 32) | lo`, without a 64-bit
/// integer-to-float conversion (SSE2 has none for packed lanes). With
/// `x = w >> 11 = hi 2^21 + (lo >> 11)`, both parts are made exactly by the
/// 2^52 exponent trick, their sum is `x` exactly (`x < 2^53`), and the
/// final `+ 0.5` and scaling are the same operations as in
/// `word_to_uniform`, so the result has the same bits.
#[inline(always)]
fn uniform_from_halves(hi: u32, lo: u32) -> f64 {
    const BIG: f64 = 4_503_599_627_370_496.0; // 2^52
    const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
    const TWO21: f64 = 2_097_152.0;
    let h = (f64::from_bits(BIG.to_bits() | u64::from(hi)) - BIG) * TWO21;
    let l = f64::from_bits(BIG.to_bits() | u64::from(lo >> 11)) - BIG;
    ((h + l) + 0.5) * SCALE
}

/// Inputs below this go through [`floor_small`].
const MAGIC_LIMIT: f64 = 2_251_799_813_685_248.0; // 2^51

/// `floor(y)` as an integer for `0 <= y < 2^51`, equal to `y as u64` there.
/// Adding and removing 2^52 rounds `y` to the nearest integer `r` exactly;
/// one is taken off when that rounded up. The bits of `r + 2^52` minus the
/// bits of 2^52 are `r`.
#[inline(always)]
fn floor_small(y: f64) -> u64 {
    const BIG: f64 = 4_503_599_627_370_496.0; // 2^52
    let r = (y + BIG) - BIG;
    let r = if r > y { r - 1.0 } else { r };
    (r + BIG).to_bits() - BIG.to_bits()
}

/// Uniform on (0, 1) from one 64-bit Philox word: `((w >> 11) + 0.5) 2^-53`,
/// the conversion [`PhiloxStream`] uses.
#[inline(always)]
pub fn word_to_uniform(w: u64) -> f64 {
    const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
    ((w >> 11) as f64 + 0.5) * SCALE
}

/// Words `2 * block .. 2 * block + 2 * LANES` of stream `(seed, path)`:
/// Philox4x32-10 on `LANES` counters at once, structure-of-arrays so the
/// rounds vectorise. Word `2 l` is `(x1 << 32) | x0` and word `2 l + 1` is
/// `(x3 << 32) | x2` of block `block + l`, as in [`PhiloxStream`].
#[inline]
pub fn philox_words(seed: u64, path: u64, block: u64) -> [u64; 2 * LANES] {
    let [c0, c1, c2, c3] = philox_lanes(seed, path, block);
    let mut w = [0u64; 2 * LANES];
    for (l, pair) in w.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        pair[0] = (u64::from(c1[l]) << 32) | u64::from(c0[l]);
        pair[1] = (u64::from(c3[l]) << 32) | u64::from(c2[l]);
    }
    w
}

/// The four 32-bit outputs `[x0, x1, x2, x3]` of blocks `block ..
/// block + LANES` of stream `(seed, path)`, one array per output word,
/// lane `l` holding block `block + l`.
#[inline(always)]
fn philox_lanes(seed: u64, path: u64, block: u64) -> [[u32; LANES]; 4] {
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
    [c0, c1, c2, c3]
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
                let mut s = PhiloxStream::new(0xDEAD_BEEF_1234, path);
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
                let mut t = PhiloxStream::new(9, path);
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

#[cfg(test)]
mod index_tests {
    use super::*;

    #[test]
    fn floor_small_equals_cast_near_integers() {
        let mut ys = vec![0.0, f64::MIN_POSITIVE, 0.5, 0.999_999_999_999_999_9];
        for k in [
            1.0f64,
            2.0,
            3.0,
            249.0,
            250.0,
            99_999.0,
            1e9,
            4e15,
            2_251_799_813_685_247.0,
        ] {
            let (mut lo, mut hi) = (k, k);
            for _ in 0..4 {
                lo = f64::from_bits(lo.to_bits() - 1);
                hi = f64::from_bits(hi.to_bits() + 1);
                ys.extend([lo, hi, k - 0.5, k + 0.5]);
            }
        }
        for y in ys.into_iter().filter(|y| *y < MAGIC_LIMIT) {
            assert_eq!(floor_small(y), y as u64, "y={y:e}");
        }
    }

    #[test]
    fn uniform_from_halves_equals_word_to_uniform() {
        let mut x = 0x0123_4567_89AB_CDEFu64;
        let mut ws = vec![
            0,
            1,
            2047,
            2048,
            u64::MAX,
            u64::MAX - 2047,
            1 << 63,
            (1 << 63) - 1,
        ];
        for _ in 0..10_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ws.push(x);
        }
        for w in ws {
            let got = uniform_from_halves((w >> 32) as u32, w as u32);
            assert_eq!(got.to_bits(), word_to_uniform(w).to_bits(), "w={w:#x}");
        }
    }
}
