//! Brent's root finder.

use crate::Error;

/// Brent's method mirroring `scipy.optimize.brentq` semantics: given
/// `f(a) f(b) <= 0`, returns a root to absolute x-tolerance `xtol`.
///
/// Returns [`Error::Numerical`] if the initial bracket has no sign
/// change, or if `max_iter` iterations elapse without reaching `xtol`
/// (scipy raises in both cases; silently returning the last iterate
/// would let a non-converged strike or vol escape into a risk number).
///
/// # Examples
///
/// ```
/// use qp_core::roots::brentq;
/// let root = brentq(|x| x * x - 2.0, 0.0, 2.0, 1e-14, 100).unwrap();
/// assert!((root - std::f64::consts::SQRT_2).abs() < 1e-14);
/// assert!(brentq(|x| x * x + 1.0, 0.0, 2.0, 1e-14, 100).is_err());
/// ```
pub fn brentq<F>(f: F, a: f64, b: f64, xtol: f64, max_iter: usize) -> Result<f64, Error>
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = (a, b);
    let mut fa = f(a);
    let mut fb = f(b);
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa * fb > 0.0 {
        return Err(Error::Numerical(format!(
            "brentq: root not bracketed on [{a}, {b}]"
        )));
    }
    let mut c = a;
    let mut fc = fa;
    let mut e = b - a;
    let mut dstep = e;
    for _ in 0..max_iter {
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * xtol;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 {
            return Ok(b);
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // Attempt inverse quadratic interpolation / secant.
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                let qq = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            let min1 = 3.0 * xm * q - (tol1 * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = dstep;
                dstep = p / q;
            } else {
                dstep = xm;
                e = dstep;
            }
        } else {
            dstep = xm;
            e = dstep;
        }
        a = b;
        fa = fb;
        if dstep.abs() > tol1 {
            b += dstep;
        } else {
            b += if xm > 0.0 { tol1 } else { -tol1 };
        }
        fb = f(b);
        if (fb > 0.0) == (fc > 0.0) {
            c = a;
            fc = fa;
            e = b - a;
            dstep = e;
        }
    }
    Err(Error::Numerical(format!(
        "brentq: no convergence to xtol {xtol} in {max_iter} iterations \
         (last iterate {b})"
    )))
}
