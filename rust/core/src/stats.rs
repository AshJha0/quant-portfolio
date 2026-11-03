//! Special functions and sample moments for the two VaR engines.
//!
//! Plan W20 (Round 8): `eq-var-engine` and `fx-var-engine` each carried
//! their own copy of this file. The two used the same Numerical Recipes
//! schemes but differed where it mattered: FX evaluated the normal CDF as
//! `(1 - P(1/2, x^2/2)) / 2`, which cancels in the lower tail, and solved
//! the Student-t quantile in the upper tail. This module keeps the equity
//! versions, which were the more accurate in every row of the comparison
//! (see `ORACLE_KERNELS.md` and the W20 handback for the numbers), and both
//! engines' `stats` modules are now wrappers over it.
//!
//! * [`ln_gamma`]: Lanczos, g = 7, 9 coefficients (~1e-15 relative).
//! * [`regularized_gamma_p`] / [`regularized_gamma_q`]: series for
//!   `x < a + 1`, modified-Lentz continued fraction otherwise (Press et al.,
//!   *Numerical Recipes*, 3rd ed., s.6.2). `Q` is evaluated directly in its
//!   region, so small tails keep their relative accuracy.
//! * [`erfc`] and [`normal_cdf`]: `erfc(x) = Q(1/2, x^2)` for `x >= 0`, so
//!   `Phi(x) = erfc(-x/sqrt 2)/2` keeps full relative accuracy in the lower
//!   tail, the tail VaR uses. This is a different algorithm from the Cody
//!   `crate::special::erfc` used by the options engines; the two agree to
//!   about 1e-15 relative and are kept apart so neither family's goldens
//!   move.
//! * [`normal_ppf`]: Acklam's guess plus one Halley step on [`normal_cdf`].
//! * [`regularized_beta`]: Lentz continued fraction with the symmetry
//!   `I_x(a, b) = 1 - I_{1-x}(b, a)`; gives the Student-t CDF and the exact
//!   binomial CDF.
//! * [`student_t_ppf`]: bracket doubling then bisection on the CDF to a
//!   width of `1e-13 max(1, |x|)`.
//! * Moments: mean, `ddof = 1` standard deviation, and skewness and excess
//!   kurtosis as biased moment ratios (`scipy.stats` with `bias=True`).
//!
//! Worked example: the 1% normal quantile.
//!
//! ```
//! let z = qp_core::stats::normal_ppf(0.01).unwrap();
//! assert!((z + 2.326_347_874_040_841).abs() < 1e-15);
//! ```

// `!(x > 0.0)` is the NaN-rejecting guard used throughout; `x <= 0.0`
// would let NaN through.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::Error;

/// Result of the fallible functions in this module.
pub type Result<T> = std::result::Result<T, Error>;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2 pi)
/// Relative convergence threshold for the series and continued fractions.
///
/// One machine epsilon (`2.22e-16`). A term smaller than this relative to
/// the running value cannot change it in `f64`, so iterating further buys
/// nothing. The previous value `1e-16` sat below half an ulp at 1.0, so the
/// test `|del - 1| < EPS` passed only on exact equality and the loops often
/// ran to `MAX_ITER`.
const EPS: f64 = f64::EPSILON;
const MAX_ITER: usize = 500;

// ---------------------------------------------------------------------------
// Log-gamma (Lanczos, g = 7, 9 coefficients; |rel err| ~ 1e-15)
// ---------------------------------------------------------------------------

// Published coefficients, kept digit for digit; rustc rounds them to f64.
#[allow(clippy::excessive_precision)]
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

/// Natural log of the Gamma function for finite `x > 0` (Lanczos
/// approximation, ~1e-15 relative accuracy; reflection for `x < 0.5`).
///
/// # Errors
/// [`Error::InvalidInput`] unless `x` is finite and `> 0`, the domain
/// on which every caller in this crate needs it.
pub fn ln_gamma(x: f64) -> Result<f64> {
    if !(x > 0.0) || !x.is_finite() {
        return Err(Error::InvalidInput(format!(
            "ln_gamma needs a finite x > 0, got {x}"
        )));
    }
    Ok(ln_gamma_pos(x))
}

/// Lanczos `ln Gamma(x)` for `x > 0` (caller guarantees the domain).
pub fn ln_gamma_pos(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma_pos(1.0 - x);
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
        return Err(Error::InvalidInput(format!(
            "incomplete gamma needs a > 0, got {a}"
        )));
    }
    if !(x >= 0.0) || !x.is_finite() {
        return Err(Error::InvalidInput(format!(
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
    sum * (-x + a * x.ln() - ln_gamma_pos(a)).exp()
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
    (-x + a * x.ln() - ln_gamma_pos(a)).exp() * h
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
///
/// Total function: `erfc(NaN) = NaN`, `erfc(+inf) = 0`, `erfc(-inf) = 2`.
/// The previous version recursed without end on NaN and panicked on
/// `+inf`, because `x * x = inf` failed the gamma argument check.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        if x == 0.0 {
            return 1.0;
        }
        // erfc(27) ~ 5e-319: below this the value is 0 in f64, and the
        // argument x^2 stays far from overflow.
        if x > 27.0 {
            return 0.0;
        }
        let x2 = x * x;
        if x2 < 1.5 {
            1.0 - gamma_series(0.5, x2)
        } else {
            gamma_cf(0.5, x2)
        }
    } else {
        2.0 - erfc(-x)
    }
}

/// Error function `erf(x) = 1 - erfc(x)`.
///
/// Total function: `erf(NaN) = NaN`, `erf(+/-inf) = +/-1`.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() < 0.5 {
        // For small |x| compute P(1/2, x^2) directly to preserve relative
        // accuracy of the *erf* value (1 - erfc would lose digits).
        let p = if x == 0.0 {
            0.0
        } else {
            gamma_series(0.5, x * x)
        };
        if x >= 0.0 { p } else { -p }
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
        return Err(Error::InvalidInput(format!(
            "normal_ppf needs p in (0, 1), got {p}"
        )));
    }
    Ok(normal_ppf_unchecked(p))
}

/// [`normal_ppf`] for `p` the caller already knows is in `(0, 1)`.
///
/// Acklam's guess, then one Halley step,
/// `x <- x - u / (1 + x u / 2)` with `u = (Phi(x) - p) / phi(x)`.
pub fn normal_ppf_unchecked(p: f64) -> f64 {
    let x = crate::special::acklam(p);
    crate::special::halley_step(x, p, normal_cdf(x), normal_pdf(x))
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
    if !(a > 0.0) || !(b > 0.0) || !a.is_finite() || !b.is_finite() {
        return Err(Error::InvalidInput(format!(
            "incomplete beta needs a, b > 0, got a={a}, b={b}"
        )));
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(Error::InvalidInput(format!(
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
        ln_gamma_pos(a + b) - ln_gamma_pos(a) - ln_gamma_pos(b) + a * x.ln() + b * (1.0 - x).ln();
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
    if !(df > 0.0) || !df.is_finite() {
        return Err(Error::InvalidInput(format!(
            "chi2_sf needs a finite df > 0, got {df}"
        )));
    }
    if x.is_nan() {
        return Err(Error::InvalidInput("chi2_sf: x is NaN".to_string()));
    }
    if x == f64::INFINITY {
        return Ok(0.0);
    }
    if x < 0.0 {
        return Ok(1.0);
    }
    regularized_gamma_q(0.5 * df, 0.5 * x)
}

/// Exact Binomial(n, p) CDF `P(X <= k)` via the incomplete-beta identity
/// `P(X <= k) = I_{1-p}(n - k, k + 1)` (what SciPy's `binom.cdf` uses).
///
/// `k < 0` gives 0 and `k >= n` gives 1. Used for the Basel traffic-light
/// zone probabilities.
///
/// # Errors
/// [`Error::InvalidInput`] unless `n >= 0` and `p` is in `[0, 1]`.
pub fn binomial_cdf(k: i64, n: i64, p: f64) -> Result<f64> {
    if n < 0 || !(0.0..=1.0).contains(&p) {
        return Err(Error::InvalidInput(format!(
            "binomial_cdf needs n >= 0 and p in [0, 1], got n={n}, p={p}"
        )));
    }
    if k < 0 {
        return Ok(0.0);
    }
    if k >= n {
        return Ok(1.0);
    }
    regularized_beta((n - k) as f64, k as f64 + 1.0, 1.0 - p)
}

// ---------------------------------------------------------------------------
// Student-t distribution
// ---------------------------------------------------------------------------

fn check_df(df: f64) -> Result<()> {
    if !(df > 0.0) || !df.is_finite() {
        return Err(Error::InvalidInput(format!(
            "Student-t df must be > 0, got {df}"
        )));
    }
    Ok(())
}

/// Student-t density with `df > 0` degrees of freedom.
pub fn student_t_pdf(x: f64, df: f64) -> Result<f64> {
    check_df(df)?;
    if x.is_nan() {
        return Err(Error::InvalidInput("student_t_pdf: x is NaN".to_string()));
    }
    let ln_norm = ln_gamma_pos(0.5 * (df + 1.0))
        - ln_gamma_pos(0.5 * df)
        - 0.5 * (df * std::f64::consts::PI).ln();
    Ok((ln_norm - 0.5 * (df + 1.0) * (1.0 + x * x / df).ln()).exp())
}

/// Student-t CDF via the regularized incomplete beta:
/// for `x <= 0`, `F(x) = I_{df/(df+x^2)}(df/2, 1/2) / 2` (symmetric above).
pub fn student_t_cdf(x: f64, df: f64) -> Result<f64> {
    check_df(df)?;
    if x.is_nan() {
        return Err(Error::InvalidInput("student_t_cdf: x is NaN".to_string()));
    }
    if x.is_infinite() {
        return Ok(if x > 0.0 { 1.0 } else { 0.0 });
    }
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
        return Err(Error::InvalidInput(format!(
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
        return Err(Error::InvalidInput(
            "mean of an empty slice is undefined".to_string(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "sample contains NaN or infinite values".to_string(),
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
        return Err(Error::InvalidInput(format!(
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
        return Err(Error::InvalidInput(format!(
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
        return Err(Error::InvalidInput(format!(
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

/// Sample moments of a data slice, as one value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    /// Sample mean.
    pub mean: f64,
    /// Sample standard deviation, `ddof = 1`.
    pub std: f64,
    /// Skewness `m3 / m2^{3/2}` (biased moment ratio).
    pub skewness: f64,
    /// Excess kurtosis `m4 / m2^2 - 3` (biased moment ratio).
    pub excess_kurtosis: f64,
}

/// [`Moments`] of `x`; needs at least 2 finite observations.
///
/// Skewness and excess kurtosis are 0 for a zero-variance sample, and are
/// computed for 2 or 3 points too (the stand-alone [`skewness`] and
/// [`excess_kurtosis`] refuse samples that short).
pub fn moments(x: &[f64]) -> Result<Moments> {
    let std = stdev(x)?;
    let mu = mean(x)?;
    let m2 = central_moment(x, mu, 2);
    let (skewness, excess_kurtosis) = if m2 > 0.0 {
        (
            central_moment(x, mu, 3) / m2.powf(1.5),
            central_moment(x, mu, 4) / (m2 * m2) - 3.0,
        )
    } else {
        (0.0, 0.0)
    };
    Ok(Moments {
        mean: mu,
        std,
        skewness,
        excess_kurtosis,
    })
}
