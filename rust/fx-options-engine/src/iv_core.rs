//! Shared implied-volatility core: normalised Black, solved on the OTM side.
//!
//! The same algorithm, constants and branch thresholds are implemented in
//! the Python (`_iv_core.py`), C++ and Rust engines. Inputs to
//! [`total_vol`] are the normalised OTM time value `beta`, its complement
//! `u = b_max - beta` and `x = -|ln(F/K)| <= 0`; the result is the total
//! volatility `s = sigma sqrt(T)`. One evaluation is one pass of the
//! iteration loop.

// `!(a < b)` is used on purpose: it is also true when a value is NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::{erf, erfc_cody as erfc};

const SQRT2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
const SQRT_2PI: f64 = 2.506_628_274_631_000_2;
const EPS: f64 = f64::EPSILON;
const MAX_ITER: u32 = 100;

#[inline]
fn norm_cdf(d: f64) -> f64 {
    0.5 * erfc(-d / SQRT2)
}

#[inline]
fn norm_pdf(d: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * d * d).exp()
}

/// `b(x,s) = e^{x/2} N(x/s+s/2) - e^{-x/2} N(x/s-s/2)` for `x <= 0`, `s > 0`,
/// written as `cosh(x/2) A + sinh(x/2) (N(d1)+N(d2))`, `A = N(d1) - N(d2)`.
pub fn normalised_otm_call(x: f64, s: f64) -> f64 {
    let h = x / s;
    let t = 0.5 * s;
    let d1 = h + t;
    let d2 = h - t;
    let a = if -x < 0.1 && s < 0.2 {
        // A = 2 n(h) sum_k He_{2k}(h) t^{2k+1} / (2k+1)!
        let (mut he_prev, mut he_cur) = (1.0_f64, h); // He_0, He_1
        let mut n = 1.0_f64;
        let tt = t * t;
        let mut term = t;
        let mut acc = t;
        let mut small = 0;
        for k in 1..40 {
            let he_next = h * he_cur - n * he_prev;
            n += 1.0;
            he_prev = he_cur;
            he_cur = he_next;
            let he_next = h * he_cur - n * he_prev;
            n += 1.0;
            he_prev = he_cur;
            he_cur = he_next;
            // he_prev is now He_{2k}
            let kf = k as f64;
            term *= tt / ((2.0 * kf) * (2.0 * kf + 1.0));
            let c = he_prev * term;
            acc += c;
            // two consecutive negligible terms: He_{2k}(h) can vanish at isolated h
            small = if c.abs() <= 1e-17 * acc.abs() {
                small + 1
            } else {
                0
            };
            if small == 2 {
                break;
            }
        }
        2.0 * norm_pdf(h) * acc
    } else if d1.abs() < 1.0 && d2.abs() < 1.0 {
        0.5 * (erf(d1 / SQRT2) - erf(d2 / SQRT2))
    } else {
        norm_cdf(d1) - norm_cdf(d2)
    };
    (0.5 * x).cosh() * a + (0.5 * x).sinh() * (norm_cdf(d1) + norm_cdf(d2))
}

/// `b_max - b = e^{x/2} N(-d1) + e^{-x/2} N(d2)`: a sum, no cancellation.
pub fn normalised_otm_call_complement(x: f64, s: f64) -> f64 {
    let d1 = x / s + 0.5 * s;
    let d2 = x / s - 0.5 * s;
    (0.5 * x).exp() * norm_cdf(-d1) + (-0.5 * x).exp() * norm_cdf(d2)
}

/// Acklam inverse normal, no refinement: used for the initial guess only.
fn acklam_guess(p: f64) -> f64 {
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
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > 1.0 - 0.02425 {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

/// Solve `b(x, s) = beta` for `s`. Requires `beta > 0`, `u > 0`, `x <= 0`.
/// `evals` is incremented once per iteration.
pub fn total_vol(beta: f64, u: f64, x: f64, evals: &mut u32) -> Result<f64, &'static str> {
    let b_max = (0.5 * x).exp();
    let s_c = (-2.0 * x).sqrt(); // inflection point of b in s
    let b_c = if s_c > 0.0 {
        normalised_otm_call(x, s_c)
    } else {
        0.0
    };
    let mode;
    let ln_target;
    let (mut lo, mut hi);
    let mut s;
    if beta < b_c {
        mode = 0; // lower: f = 1/ln(beta) - 1/ln b(s)
        ln_target = beta.ln();
        lo = 0.0;
        hi = s_c;
        // Mills-ratio asymptotic, y = s^2:
        // ln b ~ -x^2/(2y) - y/8 - ln sqrt(2pi) + 1.5 ln y - ln(x^2 - y^2/4)
        let mut y = x * x / (-2.0 * ln_target);
        for _ in 0..3 {
            if !(y < -2.0 * x) {
                break;
            }
            let rhs =
                -ln_target - 0.125 * y - LN_SQRT_2PI + 1.5 * y.ln() - (x * x - 0.25 * y * y).ln();
            if !(rhs > 0.0) {
                break;
            }
            y = x * x / (2.0 * rhs);
        }
        s = y.sqrt().min(s_c);
    } else {
        lo = s_c;
        hi = f64::INFINITY;
        if beta <= 0.5 * b_max {
            mode = 1; // middle: f = ln b(s) - ln(beta)
            ln_target = beta.ln();
        } else {
            mode = 2; // upper: f = ln(u) - ln(b_max - b(s))
            ln_target = u.ln();
        }
        let p = u / (b_max + 1.0 / b_max); // < 0.5 always
        s = (-2.0 * acklam_guess(p)).max(s_c);
        if !(s > 0.0) {
            s = SQRT_2PI * beta;
        }
    }
    // The bracket ends start as the a-priori bounds (0 or s_c, s_c or inf);
    // an end is "probed" once the objective has been evaluated there.
    let mut lo_probed = false;
    let mut hi_probed = false;
    for _ in 0..MAX_ITER {
        *evals += 1;
        let d1 = x / s + 0.5 * s;
        let bp = b_max * norm_pdf(d1); // db/ds
        let r2 = x * x / (s * s * s) - 0.25 * s; // b''/b'
        let (f, fp, fpp);
        if mode == 0 {
            let b = normalised_otm_call(x, s);
            if !(b > 0.0) {
                // underflow: s is far too small
                lo = s;
                s = 0.5 * (lo + hi);
                continue;
            }
            let l = b.ln();
            let lp = bp / b;
            let lpp = lp * r2 - lp * lp;
            f = 1.0 / ln_target - 1.0 / l;
            fp = lp / (l * l);
            fpp = lpp / (l * l) - 2.0 * lp * lp / (l * l * l);
        } else if mode == 1 {
            let b = normalised_otm_call(x, s);
            let lp = bp / b;
            f = b.ln() - ln_target;
            fp = lp;
            fpp = lp * r2 - lp * lp;
        } else {
            let w = normalised_otm_call_complement(x, s);
            let gp = bp / w;
            f = ln_target - w.ln();
            fp = gp;
            fpp = gp * r2 + gp * gp;
        }
        if f > 0.0 {
            hi = hi.min(s);
            hi_probed = hi == s;
        } else if f < 0.0 {
            lo = lo.max(s);
            lo_probed = lo == s;
        } else {
            return Ok(s);
        }
        let newton = f / fp;
        let den = 1.0 - 0.5 * newton * fpp / fp;
        let mut s_new = if den > 0.5 {
            s - newton / den
        } else {
            f64::NAN
        };
        let tol: f64;
        if lo < s_new && s_new < hi {
            tol = 1e-7; // Halley: next error ~ step^3
        } else {
            s_new = s - newton;
            if lo < s_new && s_new < hi {
                tol = 1e-10; // Newton: next error ~ step^2
            } else if s_new >= hi && hi.is_finite() && !hi_probed {
                s_new = hi; // root lies next to s_c: probe it
                tol = 0.0;
            } else if s_new <= lo && lo > 0.0 && !lo_probed {
                s_new = lo;
                tol = 0.0;
            } else {
                s_new = if hi.is_finite() {
                    0.5 * (lo + hi)
                } else {
                    2.0 * s
                };
                tol = 0.0;
            }
        }
        if (s_new - s).abs() <= tol.max(4.0 * EPS) * s || (hi - lo) <= 4.0 * EPS * s {
            return Ok(s_new);
        }
        s = s_new;
    }
    Err("implied vol: no convergence in 100 iterations")
}

/// Black implied vol from a validated quote with `lower < price < upper`.
pub fn black_implied_vol(
    price: f64,
    f: f64,
    k: f64,
    t: f64,
    df: f64,
    is_call: bool,
    evals: &mut u32,
) -> Result<f64, &'static str> {
    let lower = df * (if is_call { f - k } else { k - f }).max(0.0);
    let upper = if is_call { df * f } else { df * k };
    let norm = df * (f * k).sqrt();
    let beta = (price - lower) / norm;
    let u = (upper - price) / norm;
    if !(beta > 0.0 && u > 0.0) {
        return Err("price is within rounding of a no-arbitrage bound");
    }
    let x = -((f - k) / k).ln_1p().abs(); // F-K is exact near the money (Sterbenz)
    Ok(total_vol(beta, u, x, evals)? / t.sqrt())
}
