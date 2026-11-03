//! Error function, normal distribution and the Acklam inverse normal.
//!
//! Rust's `std` has no `erf` or `erfc`. This module implements W. J. Cody's
//! rational Chebyshev approximation (SIAM 1969; the netlib `CALERF`
//! algorithm) in three regimes, `|x| <= 0.46875`, `0.46875 < |x| <= 4` and
//! `|x| > 4`, with the `exp(-x^2)` factor split as
//! `exp(-y16^2) exp(-(y - y16)(y + y16))` for tail accuracy. Relative error
//! is below `2e-15` over `[-27, 27]` against mpmath (unit tests in the
//! engines).
//!
//! # Two coefficient tables
//!
//! Cody publishes the second coefficient of the middle regime as
//! `8.88314979438837594`. The equity options engine writes it as
//! `8.883149794388376`, which rounds to the same double. The FX options
//! engine wrote `8.8831497943883759`, which rounds to the double one ulp
//! below. Both engines pin golden values computed with their own table, so
//! both tables stay reachable:
//!
//! * [`erfc`], [`erf`], [`norm_cdf`]: Cody's coefficient, correctly rounded.
//! * [`erfc_c1_truncated`], [`erf_c1_truncated`], [`norm_cdf_c1_truncated`]:
//!   the FX options table.
//!
//! The two differ by at most a few ulps in the middle regime and agree
//! elsewhere.
//!
//! # Two normal densities
//!
//! [`norm_pdf`] multiplies by `1/sqrt(2 pi)`; [`norm_pdf_div`] divides by
//! `sqrt(2 pi)`. They can differ in the last bit. The equity options
//! engine prices with the second; everything else uses the first.

// Published coefficients are kept digit for digit as Cody and Acklam print them.
#![allow(clippy::excessive_precision)]

const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// `1 / sqrt(2 pi)`.
pub const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// `sqrt(2 pi)`, as the equity options engine writes it.
pub const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Cody's `C[1]`, correctly rounded.
const C1_CODY: f64 = 8.883_149_794_388_376e0;
/// Cody's `C[1]` truncated to 17 significant digits: one ulp below.
const C1_TRUNCATED: f64 = 8.883_149_794_388_375_9e0;

// Regime |x| <= 0.46875: erf(x) = x P(x^2)/Q(x^2).
const CODY_A: [f64; 5] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_6e2,
    3.774_852_376_853_020_2e2,
    3.209_377_589_138_469_5e3,
    1.857_777_061_846_031_5e-1,
];
const CODY_B: [f64; 4] = [
    2.360_129_095_234_412_1e1,
    2.440_246_379_344_441_7e2,
    1.282_616_526_077_372_3e3,
    2.844_236_833_439_170_6e3,
];

/// Cody's direct `erf(x) = x P(x^2)/Q(x^2)` for `|x| <= 0.46875`.
#[inline]
fn erf_small(x: f64) -> f64 {
    let y = x.abs();
    let ysq = if y > 1.11e-16 { y * y } else { 0.0 };
    let mut xnum = CODY_A[4] * ysq;
    let mut xden = ysq;
    for i in 0..3 {
        xnum = (xnum + CODY_A[i]) * ysq;
        xden = (xden + CODY_B[i]) * ysq;
    }
    x * (xnum + CODY_A[3]) / (xden + CODY_B[3])
}

/// Cody's erfc with the middle-regime `C[1]` passed in. Both public tables
/// are this function with a constant argument.
#[inline(always)]
fn erfc_with_c1(x: f64, c1: f64) -> f64 {
    // Regime 0.46875 < |x| <= 4: erfc(x) = exp(-x^2) P(x)/Q(x).
    let c: [f64; 9] = [
        5.641_884_969_886_700_9e-1,
        c1,
        6.611_919_063_714_163e1,
        2.986_351_381_974_001_3e2,
        8.819_522_212_417_691e2,
        1.712_047_612_634_070_6e3,
        2.051_078_377_826_071_5e3,
        1.230_339_354_797_997_2e3,
        2.153_115_354_744_038_5e-8,
    ];
    const D: [f64; 8] = [
        1.574_492_611_070_983_5e1,
        1.176_939_508_913_125e2,
        5.371_811_018_620_098_5e2,
        1.621_389_574_566_690_2e3,
        3.290_799_235_733_459_7e3,
        4.362_619_090_143_247e3,
        3.439_367_674_143_721_6e3,
        1.230_339_354_803_749_4e3,
    ];
    // Regime |x| > 4: erfc(x) = exp(-x^2)/x [1/sqrt(pi) - P(1/x^2)/Q(1/x^2)/x^2].
    const P: [f64; 6] = [
        3.053_266_349_612_323_4e-1,
        3.603_448_999_498_044_4e-1,
        1.257_817_261_112_292_5e-1,
        1.608_378_514_874_227_7e-2,
        6.587_491_615_298_378e-4,
        1.631_538_713_730_209_8e-2,
    ];
    const Q: [f64; 5] = [
        2.568_520_192_289_822_4e0,
        1.872_952_849_923_460_4e0,
        5.279_051_029_514_284e-1,
        6.051_834_131_244_132e-2,
        2.335_204_976_268_691_8e-3,
    ];
    const SQRPI: f64 = 5.641_895_835_477_562_9e-1; // 1/sqrt(pi)
    const THRESH: f64 = 0.46875;
    const XBIG: f64 = 26.543; // erfc underflows to 0 beyond this

    let y = x.abs();
    if y <= THRESH {
        return 1.0 - erf_small(x);
    }

    // y > 0.46875 here, so floor and trunc agree.
    let result = if y <= 4.0 {
        let mut xnum = c[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + c[i]) * y;
            xden = (xden + D[i]) * y;
        }
        let r = (xnum + c[7]) / (xden + D[7]);
        let y16 = (y * 16.0).floor() / 16.0;
        let del = (y - y16) * (y + y16);
        (-y16 * y16).exp() * (-del).exp() * r
    } else if y >= XBIG {
        0.0
    } else {
        let ysq = 1.0 / (y * y);
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let mut r = ysq * (xnum + P[4]) / (xden + Q[4]);
        r = (SQRPI - r) / y;
        let y16 = (y * 16.0).floor() / 16.0;
        let del = (y - y16) * (y + y16);
        (-y16 * y16).exp() * (-del).exp() * r
    };

    if x < 0.0 { 2.0 - result } else { result }
}

/// Complementary error function `erfc(x) = 1 - erf(x)`, Cody's table.
///
/// # Examples
///
/// ```
/// use qp_core::special::erfc;
/// assert_eq!(erfc(0.0), 1.0);
/// assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-16);
/// assert!((erfc(-1.0) - 1.8427007929497148).abs() < 1e-15);
/// ```
#[inline]
pub fn erfc(x: f64) -> f64 {
    erfc_with_c1(x, C1_CODY)
}

/// [`erfc`] with the FX options engine's truncated `C[1]`.
#[inline]
pub fn erfc_c1_truncated(x: f64) -> f64 {
    erfc_with_c1(x, C1_TRUNCATED)
}

/// Error function. Cody's direct form for `|x| <= 0.46875` (full relative
/// precision near zero), `1 - erfc(x)` beyond, where `erfc(x) <= 0.51` so
/// nothing cancels.
///
/// # Examples
///
/// ```
/// use qp_core::special::erf;
/// assert!((erf(1e-10) - 1.1283791670955126e-10).abs() < 1e-25);
/// assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-15);
/// ```
#[inline]
pub fn erf(x: f64) -> f64 {
    if x.abs() <= 0.46875 {
        erf_small(x)
    } else {
        1.0 - erfc(x)
    }
}

/// [`erf`] with the FX options engine's truncated `C[1]`.
#[inline]
pub fn erf_c1_truncated(x: f64) -> f64 {
    if x.abs() <= 0.46875 {
        erf_small(x)
    } else {
        1.0 - erfc_c1_truncated(x)
    }
}

/// Standard normal CDF `0.5 erfc(-x / sqrt(2))`, written with a multiply
/// by `1/sqrt(2)`. The erfc route keeps full relative accuracy in the
/// lower tail.
///
/// # Examples
///
/// ```
/// use qp_core::special::norm_cdf;
/// assert_eq!(norm_cdf(0.0), 0.5);
/// assert!((norm_cdf(1.959963984540054) - 0.975).abs() < 1e-12);
/// assert!(norm_cdf(-37.0) > 0.0);
/// ```
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// [`norm_cdf`] with the FX options engine's truncated `C[1]`.
#[inline]
pub fn norm_cdf_c1_truncated(x: f64) -> f64 {
    0.5 * erfc_c1_truncated(-x * FRAC_1_SQRT_2)
}

/// Standard normal density, `exp(-x^2/2) / sqrt(2 pi)` as a multiply.
///
/// # Examples
///
/// ```
/// use qp_core::special::norm_pdf;
/// assert!((norm_pdf(0.0) - 0.3989422804014327).abs() < 1e-16);
/// ```
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal density as a divide by [`SQRT_2PI`].
#[inline]
pub fn norm_pdf_div(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

/// Acklam's rational approximation to the inverse normal CDF, with no
/// refinement and no domain check (relative error about `1.15e-9`).
///
/// For `p` outside (0, 1) the result is NaN or infinite; callers check
/// the domain first. All four engines start from this same rational.
///
/// # Examples
///
/// ```
/// use qp_core::special::acklam;
/// assert_eq!(acklam(0.5), 0.0);
/// assert!((acklam(0.975) - 1.959963984540054).abs() < 1e-8);
/// ```
#[inline]
pub fn acklam(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// One Halley step for `Phi(x) = p`, given `cdf_x = Phi(x)` and
/// `pdf_x = phi(x)`: `x - u / (1 + x u / 2)` with `u = (cdf_x - p) / pdf_x`.
///
/// The engines refine [`acklam`] with their own CDF, so the CDF value is
/// passed in rather than computed here.
#[inline]
pub fn halley_step(x: f64, p: f64, cdf_x: f64, pdf_x: f64) -> f64 {
    let e = cdf_x - p;
    let u = e / pdf_x;
    x - u / (1.0 + 0.5 * x * u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_differ_only_in_c1() {
        assert_eq!(C1_CODY.to_bits() - C1_TRUNCATED.to_bits(), 1);
        // Outside the middle regime the two tables are the same function.
        for x in [-30.0, -5.0, -0.3, 0.0, 0.2, 0.46875, 4.5, 26.0, 30.0] {
            assert_eq!(erfc(x).to_bits(), erfc_c1_truncated(x).to_bits(), "{x}");
        }
    }

    #[test]
    fn erfc_reference_values() {
        // mpmath, 30 digits, rounded to double.
        let cases = [
            (0.5, 0.479_500_122_186_953_5),
            (1.0, 0.157_299_207_050_285_13),
            (2.0, 4.677_734_981_047_266e-3),
            (3.5, 7.430_983_723_414_128e-7),
            (5.0, 1.537_459_794_428_034_8e-12),
            (10.0, 2.088_487_583_762_544_7e-45),
            (-1.0, 1.842_700_792_949_714_9),
        ];
        for (x, want) in cases {
            for got in [erfc(x), erfc_c1_truncated(x)] {
                assert!(
                    ((got - want) / want).abs() < 2e-15,
                    "erfc({x}) = {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn acklam_plus_two_halley_steps_inverts_the_cdf() {
        for p in [1e-12, 1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-6] {
            let mut x = acklam(p);
            for _ in 0..2 {
                x = halley_step(x, p, norm_cdf(x), norm_pdf(x));
            }
            assert!(((norm_cdf(x) - p) / p).abs() < 1e-13, "p = {p}");
        }
    }
}
