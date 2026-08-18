//! Special functions and moment statistics (zero-dependency).
//!
//! Self-contained implementations of the distribution machinery the VaR
//! engine needs, with validated accuracy:
//!
//! * [`erfc`] — via the regularized incomplete gamma function
//!   `erfc(x) = Q(1/2, x^2)`; validated against `math.erfc` reference
//!   values to better than 1e-12 relative;
//! * [`normal_pdf`] / [`normal_cdf`] — erfc-based,
//!   `Phi(x) = erfc(-x / sqrt(2)) / 2`;
//! * [`normal_ppf`] — Acklam's rational approximation (|rel err| < 1.15e-9)
//!   refined by one Halley step with the high-accuracy CDF, giving
//!   near-machine-precision quantiles (tested to 1e-9 absolute and better);
//! * [`regularized_beta`] — continued fraction (modified Lentz), powering
//!   the Student-t CDF and the exact binomial CDF for the Basel zones;
//! * [`regularized_gamma_p`] / [`regularized_gamma_q`] — series +
//!   continued fraction (Numerical Recipes `gser`/`gcf` scheme), powering
//!   `erfc` and the chi-squared p-values of the backtests;
//! * [`student_t_ppf`] — bisection on the incomplete-beta CDF;
//! * moments — mean, sample stdev (`ddof = 1`), skewness and **excess**
//!   kurtosis as biased moment ratios (matching `scipy.stats` with
//!   `bias=True`, which the Python reference uses).
//!
//! All functions are pure; invalid inputs return
//! [`EqVarError::InvalidInput`](crate::EqVarError::InvalidInput).

use crate::{EqVarError, Result};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2 pi)
const EPS: f64 = 1e-16;
const MAX_ITER: usize = 500;

// ---------------------------------------------------------------------------
// Log-gamma (Lanczos, g = 7, 9 coefficients; |rel err| ~ 1e-15)
// ---------------------------------------------------------------------------

const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural log of the Gamma function for `x > 0` (Lanczos approximation,
/// ~1e-15 relative accuracy; reflection formula for `x < 0.5`).
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut acc = LANCZOS[0];
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        acc += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

// ---------------------------------------------------------------------------
// Regularized incomplete gamma: P(a, x), Q(a, x)
// ---------------------------------------------------------------------------

/// Lower regularized incomplete gamma `P(a, x)`, `a > 0`, `x >= 0`.
///
/// Series expansion for `x < a + 1`, continued fraction otherwise
/// (each converges to ~1e-15 relative in its region).
pub fn regularized_gamma_p(a: f64, x: f64) -> Result<f64> {
    check_gamma_args(a, x)?;
    if x == 0.0 {
        return Ok(0.0);
    }
    if x < a + 1.0 {
        Ok(gamma_series(a, x))
    } else {
        Ok(1.0 - gamma_cf(a, x))
    }
}

/// Upper regularized incomplete gamma `Q(a, x) = 1 - P(a, x)`.
///
/// Evaluated directly by continued fraction for `x >= a + 1`, so the tiny
/// tail values (e.g. `erfc(5) ~ 1.5e-12`) keep full **relative** accuracy.
pub fn regularized_gamma_q(a: f64, x: f64) -> Result<f64> {
    check_gamma_args(a, x)?;
    if x == 0.0 {
        return Ok(1.0);
    }
    if x < a + 1.0 {
        Ok(1.0 - gamma_series(a, x))
    } else {
        Ok(gamma_cf(a, x))
    }
}

fn check_gamma_args(a: f64, x: f64) -> Result<()> {
    if !(a > 0.0) || !a.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "incomplete gamma needs a > 0, got {a}"
        )));
    }
    if !(x >= 0.0) || !x.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "incomplete gamma needs x >= 0, got {x}"
        )));
    }
    Ok(())
}

/// Series representation of P(a, x) (converges fast for x < a + 1).
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut term = 1.0 / a;
    let mut sum = term;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued-fraction representation of Q(a, x) (modified Lentz),
/// converges fast for x >= a + 1.
fn gamma_cf(a: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

// ---------------------------------------------------------------------------
// erfc / normal distribution
// ---------------------------------------------------------------------------

/// Complementary error function `erfc(x) = 2/sqrt(pi) * int_x^inf e^{-t^2} dt`.
///
/// Evaluated through the regularized incomplete gamma identity
/// `erfc(x) = Q(1/2, x^2)` for `x >= 0` and the reflection
/// `erfc(-x) = 2 - erfc(x)`. Validated against `math.erfc` reference values
/// over `[-2.5, 7]` to better than 1e-12 relative (see `tests/test_stats.rs`).
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        if x == 0.0 {
            return 1.0;
        }
        // a = 0.5 > 0 and x^2 > 0: cannot fail.
        regularized_gamma_q(0.5, x * x).expect("erfc: internal gamma args valid")
    } else {
        2.0 - erfc(-x)
    }
}

/// Error function `erf(x) = 1 - erfc(x)`.
pub fn erf(x: f64) -> f64 {
    if x.abs() < 0.5 {
        // For small |x| compute P(1/2, x^2) directly to preserve relative
        // accuracy of the *erf* value (1 - erfc would lose digits).
        let p = regularized_gamma_p(0.5, x * x).expect("erf: internal gamma args valid");
        if x >= 0.0 {
            p
        } else {
            -p
        }
    } else {
        1.0 - erfc(x)
    }
}

/// Standard normal density `phi(x) = e^{-x^2/2} / sqrt(2 pi)`.
pub fn normal_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF `Phi(x) = erfc(-x / sqrt(2)) / 2`.
///
/// Full relative accuracy in the lower tail (the tail VaR cares about).
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

/// Inverse standard normal CDF `Phi^{-1}(p)`, `p` in `(0, 1)`.
///
/// Acklam's piecewise rational approximation (|rel err| < 1.15e-9)
/// followed by one Halley refinement step using the erfc-based CDF, which
/// pushes the error to near machine precision. Unit-tested against known
/// quantiles, e.g. `Phi^{-1}(0.975) = 1.959963984540054`.
pub fn normal_ppf(p: f64) -> Result<f64> {
    if !(p > 0.0 && p < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "normal_ppf needs p in (0, 1), got {p}"
        )));
    }
    // Acklam coefficients.
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

    let mut x = if p < P_LOW {
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

    // One Halley step: x <- x - u / (1 + x u / 2), u = (Phi(x) - p) / phi(x).
    let e = normal_cdf(x) - p;
    let u = e / normal_pdf(x);
    x -= u / (1.0 + 0.5 * x * u);
    Ok(x)
}

// ---------------------------------------------------------------------------
// Regularized incomplete beta and derived CDFs
// ---------------------------------------------------------------------------

/// Regularized incomplete beta `I_x(a, b)`, `a, b > 0`, `x` in `[0, 1]`.
///
/// Continued fraction (modified Lentz) with the symmetry
/// `I_x(a, b) = 1 - I_{1-x}(b, a)` applied so the fraction is always in its
/// fast-convergence region.
pub fn regularized_beta(a: f64, b: f64, x: f64) -> Result<f64> {
    if !(a > 0.0) || !(b > 0.0) {
        return Err(EqVarError::InvalidInput(format!(
            "incomplete beta needs a, b > 0, got a={a}, b={b}"
        )));
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(EqVarError::InvalidInput(format!(
            "incomplete beta needs x in [0, 1], got {x}"
        )));
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    let ln_front =
        ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(front * beta_cf(a, b, x) / a)
    } else {
        Ok(1.0 - front * beta_cf(b, a, 1.0 - x) / b)
    }
}

/// Continued fraction for the incomplete beta (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Chi-squared survival function `P(X > x)` for `df` degrees of freedom:
/// `Q(df/2, x/2)`. Used for the Kupiec / Christoffersen LR p-values.
pub fn chi2_sf(x: f64, df: f64) -> Result<f64> {
    if !(df > 0.0) {
        return Err(EqVarError::InvalidInput(format!(
            "chi2_sf needs df > 0, got {df}"
        )));
    }
    if x < 0.0 {
        return Ok(1.0);
    }
    regularized_gamma_q(0.5 * df, 0.5 * x)
}

/// Exact Binomial(n, p) CDF `P(X <= k)` via the incomplete-beta identity
/// `P(X <= k) = I_{1-p}(n - k, k + 1)`.
///
/// `k >= n` returns 1. Used for the Basel traffic-light zone probabilities.
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(EqVarError::InvalidInput(format!(
            "binomial_cdf needs p in [0, 1], got {p}"
        )));
    }
    if n == 0 || k >= n {
        return Ok(1.0);
    }
    regularized_beta((n - k) as f64, k as f64 + 1.0, 1.0 - p)
}

// ---------------------------------------------------------------------------
// Student-t distribution
// ---------------------------------------------------------------------------

fn check_df(df: f64) -> Result<()> {
    if !(df > 0.0) || !df.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "Student-t df must be > 0, got {df}"
        )));
    }
    Ok(())
}

/// Student-t density with `df > 0` degrees of freedom.
pub fn student_t_pdf(x: f64, df: f64) -> Result<f64> {
    check_df(df)?;
    let ln_norm = ln_gamma(0.5 * (df + 1.0))
        - ln_gamma(0.5 * df)
        - 0.5 * (df * std::f64::consts::PI).ln();
    Ok((ln_norm - 0.5 * (df + 1.0) * (1.0 + x * x / df).ln()).exp())
}

/// Student-t CDF via the regularized incomplete beta:
/// for `x <= 0`, `F(x) = I_{df/(df+x^2)}(df/2, 1/2) / 2` (symmetric above).
pub fn student_t_cdf(x: f64, df: f64) -> Result<f64> {
    check_df(df)?;
    if x == 0.0 {
        return Ok(0.5);
    }
    let ib = regularized_beta(0.5 * df, 0.5, df / (df + x * x))?;
    Ok(if x < 0.0 { 0.5 * ib } else { 1.0 - 0.5 * ib })
}

/// Student-t quantile `F^{-1}(p)` by bisection on the CDF.
///
/// Bracket doubling followed by bisection to a width of `1e-13 * max(1, |x|)`;
/// agrees with `scipy.stats.t.ppf` to better than 1e-9 over the tested
/// range (`tests/test_stats.rs`). `p` must be in `(0, 1)`.
pub fn student_t_ppf(p: f64, df: f64) -> Result<f64> {
    check_df(df)?;
    if !(p > 0.0 && p < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "student_t_ppf needs p in (0, 1), got {p}"
        )));
    }
    if (p - 0.5).abs() < 1e-16 {
        return Ok(0.0);
    }
    // Bracket the root: expand [lo, hi] until F(lo) < p < F(hi).
    let mut lo = -1.0;
    let mut hi = 1.0;
    while student_t_cdf(lo, df)? > p {
        lo *= 2.0;
        if lo < -1e12 {
            break;
        }
    }
    while student_t_cdf(hi, df)? < p {
        hi *= 2.0;
        if hi > 1e12 {
            break;
        }
    }
    // Bisection.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, df)? < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() <= 1e-13 * hi.abs().max(1.0) {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// Sample moments
// ---------------------------------------------------------------------------

/// Arithmetic mean; errors on empty input.
pub fn mean(x: &[f64]) -> Result<f64> {
    if x.is_empty() {
        return Err(EqVarError::InvalidInput(
            "mean of an empty slice is undefined".to_string(),
        ));
    }
    Ok(x.iter().sum::<f64>() / x.len() as f64)
}

/// Central moment of order `k` about the sample mean (biased, divides by n).
fn central_moment(x: &[f64], mu: f64, k: u32) -> f64 {
    x.iter().map(|v| (v - mu).powi(k as i32)).sum::<f64>() / x.len() as f64
}

/// Sample standard deviation with `ddof = 1`; errors if fewer than 2 points.
pub fn stdev(x: &[f64]) -> Result<f64> {
    if x.len() < 2 {
        return Err(EqVarError::InvalidInput(format!(
            "stdev (ddof=1) needs at least 2 observations, got {}",
            x.len()
        )));
    }
    let mu = mean(x)?;
    let ss = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>();
    Ok((ss / (x.len() - 1) as f64).sqrt())
}

/// Population variance (`ddof = 0`), the seed used by EWMA volatility
/// (matches `numpy.var` default); errors on empty input.
pub fn population_variance(x: &[f64]) -> Result<f64> {
    let mu = mean(x)?;
    Ok(central_moment(x, mu, 2))
}

/// Sample skewness `m3 / m2^{3/2}` (biased moment ratio, scipy `bias=True`).
///
/// Errors if fewer than 3 points; returns 0 for zero-variance input.
pub fn skewness(x: &[f64]) -> Result<f64> {
    if x.len() < 3 {
        return Err(EqVarError::InvalidInput(format!(
            "skewness needs at least 3 observations, got {}",
            x.len()
        )));
    }
    let mu = mean(x)?;
    let m2 = central_moment(x, mu, 2);
    if m2 <= 0.0 {
        return Ok(0.0);
    }
    Ok(central_moment(x, mu, 3) / m2.powf(1.5))
}

/// Sample **excess** kurtosis `m4 / m2^2 - 3` (biased, scipy `bias=True`).
///
/// Errors if fewer than 4 points; returns 0 for zero-variance input.
pub fn excess_kurtosis(x: &[f64]) -> Result<f64> {
    if x.len() < 4 {
        return Err(EqVarError::InvalidInput(format!(
            "excess kurtosis needs at least 4 observations, got {}",
            x.len()
        )));
    }
    let mu = mean(x)?;
    let m2 = central_moment(x, mu, 2);
    if m2 <= 0.0 {
        return Ok(0.0);
    }
    Ok(central_moment(x, mu, 4) / (m2 * m2) - 3.0)
}
