//! Special functions and sample moments (std-only implementations).
//!
//! Everything the estimators need is implemented and validated here:
//!
//! * normal pdf / CDF (CDF via the regularised incomplete gamma, accurate
//!   to ~1e-14);
//! * inverse normal CDF: Acklam's rational approximation polished by one
//!   Halley step (|error| < 1e-13, unit-tested < 1e-9 against reference
//!   values);
//! * regularised incomplete gamma `P`/`Q` (series + Lentz continued
//!   fraction) - gives chi-square survival probabilities for the Kupiec /
//!   Christoffersen tests;
//! * regularised incomplete beta (Lentz continued fraction) - gives the
//!   Student-t CDF, the exact binomial CDF for the Basel traffic light,
//!   and (via safeguarded Newton) the Student-t quantile;
//! * sample moments (mean, unbiased variance, skewness, excess kurtosis).
//!
//! Conventions: quantile levels are probabilities in (0, 1); Student-t
//! quantiles are for the *standard* t (unit scale), and the estimators
//! rescale by `sqrt((df-2)/df)` where a variance-matched t is needed.

use crate::{FxVarError, Result};

/// 1/sqrt(2*pi).
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Standard normal probability density `phi(x)`.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF `Phi(x)`, accurate to ~1e-14.
///
/// Implemented as `Phi(x) = (1 ± P(1/2, x^2/2)) / 2` through the
/// regularised incomplete gamma function (no libm `erf` dependency).
pub fn norm_cdf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.5;
    }
    let p = reg_lower_gamma(0.5, 0.5 * x * x);
    if x > 0.0 {
        0.5 * (1.0 + p)
    } else {
        0.5 * (1.0 - p)
    }
}

/// Inverse standard normal CDF `Phi^{-1}(p)`, |error| < 1e-13.
///
/// Acklam (2003) rational approximation refined by one Halley step using
/// the exact [`norm_cdf`]/[`norm_pdf`]. Unit-tested to better than 1e-9
/// against SciPy reference values.
///
/// # Panics
/// Panics if `p` is outside `(0, 1)` (guard with
/// [`crate::validate_alpha`] on user input).
pub fn inv_norm_cdf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "inv_norm_cdf requires p in (0, 1)");
    // Acklam coefficients.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_690e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    const P_LOW: f64 = 0.024_25;
    let x = if p < P_LOW {
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
    };
    // One Halley refinement against the exact CDF.
    let e = norm_cdf(x) - p;
    let u = e / norm_pdf(x);
    x - u / (1.0 + 0.5 * x * u)
}

/// Natural log of the Gamma function (Lanczos, g = 7, 9 coefficients).
///
/// Accurate to ~1e-14 relative for `x > 0`.
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularised lower incomplete gamma `P(a, x)`.
///
/// Series expansion for `x < a + 1`, Lentz continued fraction otherwise
/// (Numerical Recipes scheme); ~1e-14 accuracy over the tested range.
pub fn reg_lower_gamma(a: f64, x: f64) -> f64 {
    assert!(a > 0.0 && x >= 0.0, "reg_lower_gamma requires a > 0, x >= 0");
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // series
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..500 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        1.0 - reg_upper_gamma_cf(a, x)
    }
}

/// Regularised upper incomplete gamma `Q(a, x) = 1 - P(a, x)`.
pub fn reg_upper_gamma(a: f64, x: f64) -> f64 {
    assert!(a > 0.0 && x >= 0.0, "reg_upper_gamma requires a > 0, x >= 0");
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - reg_lower_gamma(a, x)
    } else {
        reg_upper_gamma_cf(a, x)
    }
}

/// Continued-fraction evaluation of `Q(a, x)` for `x >= a + 1` (Lentz).
fn reg_upper_gamma_cf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Chi-square survival function `P(X > x)` with `df` degrees of freedom.
///
/// `chi2_sf(x, df) = Q(df/2, x/2)`; used for Kupiec / Christoffersen
/// p-values. Validated against SciPy to ~1e-12.
pub fn chi2_sf(x: f64, df: f64) -> Result<f64> {
    if df <= 0.0 {
        return Err(FxVarError::invalid("chi-square df must be > 0"));
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    Ok(reg_upper_gamma(0.5 * df, 0.5 * x))
}

/// Regularised incomplete beta `I_x(a, b)` (Lentz continued fraction with
/// the symmetry `I_x(a,b) = 1 - I_{1-x}(b,a)` for fast convergence).
pub fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "reg_inc_beta requires a, b > 0");
    assert!((0.0..=1.0).contains(&x), "reg_inc_beta requires x in [0, 1]");
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    let ln_front =
        ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Lentz continued fraction for the incomplete beta.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        // even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        // odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Standard Student-t probability density with `df` degrees of freedom.
pub fn student_t_pdf(x: f64, df: f64) -> f64 {
    let ln_c = ln_gamma(0.5 * (df + 1.0))
        - ln_gamma(0.5 * df)
        - 0.5 * (df * std::f64::consts::PI).ln();
    (ln_c - 0.5 * (df + 1.0) * (1.0 + x * x / df).ln()).exp()
}

/// Standard Student-t CDF via the regularised incomplete beta.
pub fn student_t_cdf(x: f64, df: f64) -> f64 {
    if x == 0.0 {
        return 0.5;
    }
    let ib = reg_inc_beta(0.5 * df, 0.5, df / (df + x * x));
    if x > 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// Standard Student-t quantile (inverse CDF) with `df > 0`.
///
/// Safeguarded Newton on [`student_t_cdf`] from a normal-quantile start,
/// with bisection fallback; converges to ~1e-13 relative and is validated
/// against SciPy `t.ppf` to 1e-10.
///
/// # Errors
/// [`FxVarError::Invalid`] if `p` is outside (0, 1) or `df <= 0`.
pub fn student_t_quantile(p: f64, df: f64) -> Result<f64> {
    if !(p > 0.0 && p < 1.0) {
        return Err(FxVarError::invalid("t quantile requires p in (0, 1)"));
    }
    if df <= 0.0 {
        return Err(FxVarError::invalid("t quantile requires df > 0"));
    }
    if p == 0.5 {
        return Ok(0.0);
    }
    // Symmetry: solve in the upper tail.
    if p < 0.5 {
        return Ok(-student_t_quantile(1.0 - p, df)?);
    }
    // Start from the normal quantile, inflated for fat tails.
    let z = inv_norm_cdf(p);
    let mut x = if df > 6.0 { z } else { z * (df / (df - 2.0).max(0.5)).sqrt() };
    if x <= 0.0 {
        x = 0.5;
    }
    let (mut lo, mut hi) = (0.0_f64, f64::INFINITY);
    for _ in 0..100 {
        let f = student_t_cdf(x, df) - p;
        if f > 0.0 {
            hi = hi.min(x);
        } else {
            lo = lo.max(x);
        }
        let d = student_t_pdf(x, df);
        let mut step = f / d;
        // clamp the Newton step inside the bracket
        let mut xn = x - step;
        if !(xn.is_finite()) || xn <= lo || (hi.is_finite() && xn >= hi) {
            xn = if hi.is_finite() { 0.5 * (lo + hi) } else { 2.0 * x.max(1.0) };
            step = x - xn;
        }
        x = xn;
        if step.abs() <= 1e-14 * (1.0 + x.abs()) {
            break;
        }
    }
    Ok(x)
}

/// Exact binomial CDF `P(X <= k)` for `X ~ Binomial(n, p)`.
///
/// Computed through the incomplete-beta identity
/// `P(X <= k) = I_{1-p}(n-k, k+1)` (what SciPy's `binom.cdf` uses), so the
/// Basel traffic-light boundaries are exact, not normal-approximated.
pub fn binom_cdf(k: i64, n: i64, p: f64) -> Result<f64> {
    if n < 0 || !(0.0..=1.0).contains(&p) {
        return Err(FxVarError::invalid("binom_cdf requires n >= 0, p in [0,1]"));
    }
    if k < 0 {
        return Ok(0.0);
    }
    if k >= n {
        return Ok(1.0);
    }
    if p == 0.0 {
        return Ok(1.0);
    }
    if p == 1.0 {
        return Ok(0.0);
    }
    Ok(reg_inc_beta((n - k) as f64, (k + 1) as f64, 1.0 - p))
}

/// Sample moments of a data slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    /// Sample mean.
    pub mean: f64,
    /// Unbiased (ddof=1) sample standard deviation.
    pub std: f64,
    /// Skewness `m3 / m2^{3/2}` (population/biased normalisation, as used
    /// by the Cornish-Fisher inputs).
    pub skewness: f64,
    /// Excess kurtosis `m4 / m2^2 - 3` (population normalisation).
    pub excess_kurtosis: f64,
}

/// Compute [`Moments`] of `x` (requires at least 2 observations).
///
/// # Errors
/// [`FxVarError::Invalid`] for fewer than 2 points or NaNs (NaN policy:
/// refuse, never impute).
pub fn moments(x: &[f64]) -> Result<Moments> {
    if x.len() < 2 {
        return Err(FxVarError::invalid("moments requires at least 2 observations"));
    }
    if x.iter().any(|v| v.is_nan()) {
        return Err(FxVarError::invalid("moments input contains NaNs (NaN policy: refuse)"));
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for v in x {
        let d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    let var_unbiased = m2 / (n - 1.0);
    m2 /= n;
    m3 /= n;
    m4 /= n;
    let (skewness, excess_kurtosis) = if m2 > 0.0 {
        (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0)
    } else {
        (0.0, 0.0)
    };
    Ok(Moments {
        mean,
        std: var_unbiased.sqrt(),
        skewness,
        excess_kurtosis,
    })
}

/// Validate a VaR/ES confidence level, returning it unchanged.
///
/// `alpha` is a **confidence level** (`0.99` = 99 % VaR), matching the
/// C++/Python FX engines.
///
/// # Errors
/// [`FxVarError::Invalid`] unless `alpha` is strictly inside `(0, 1)`.
pub fn validate_alpha(alpha: f64) -> Result<f64> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(FxVarError::invalid(format!("alpha must be in (0, 1), got {alpha}")));
    }
    Ok(alpha)
}

/// Validate a VaR horizon in trading days, returning it unchanged.
///
/// # Errors
/// [`FxVarError::Invalid`] unless `horizon_days > 0`.
pub fn validate_horizon(horizon_days: f64) -> Result<f64> {
    if !(horizon_days > 0.0) {
        return Err(FxVarError::invalid(format!(
            "horizon_days must be > 0, got {horizon_days}"
        )));
    }
    Ok(horizon_days)
}

/// Unbiased (ddof=1) standard deviation of a slice (0 for < 2 points).
pub fn sample_std(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let ss: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    (ss / (n - 1.0)).sqrt()
}
