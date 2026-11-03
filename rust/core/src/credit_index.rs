//! ISDA CDS Standard Model legs, the numeric kernel under the credit
//! indices (CDX, iTraxx).
//!
//! Rust port of `isda_legs_kernel` in
//! `python/core/src/qp_core/credit_index/isda.py`; the C++ port is
//! `cpp/core/include/qp/credit_index.hpp`. Conventions, sources and the
//! worked example (Markit's published upfronts for 21 May 2009) are in the
//! Python docstring. In short: piecewise-flat hazard and piecewise-flat
//! forwards, both legs integrated exactly on the union of the knots
//! (O'Kane 2008 eqs. 6.6 and 6.11), accrued-on-default with the ISDA
//! half-day bias `1/730`, Taylor expansion when `|k D| < 1e-8`.
//!
//! Dates are turned into times on the Python side; this kernel takes only
//! numbers, in the same order of operations, and
//! `tests/credit_bondfut_parity.rs` checks it against the Python values.

use crate::Error;

const BIAS: f64 = 1.0 / 730.0;

fn cum_hazard(times: &[f64], rates: &[f64], t: f64) -> f64 {
    let mut out = 0.0;
    let mut prev = 0.0;
    let n = times.len();
    for k in 0..n {
        let hi = if k + 1 < n { times[k] } else { f64::INFINITY };
        if t <= prev {
            break;
        }
        out += rates[k] * (t.min(hi) - prev);
        prev = hi;
    }
    out
}

fn df(times: &[f64], zeros: &[f64], t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    let n = times.len();
    if t <= times[0] {
        return (-zeros[0] * t).exp();
    }
    for k in 1..n {
        if t <= times[k] {
            let a = zeros[k - 1] * times[k - 1];
            let b = zeros[k] * times[k];
            let w = (t - times[k - 1]) / (times[k] - times[k - 1]);
            return (-(a + w * (b - a))).exp();
        }
    }
    if n == 1 {
        return (-zeros[0] * t).exp();
    }
    let f =
        (zeros[n - 1] * times[n - 1] - zeros[n - 2] * times[n - 2]) / (times[n - 1] - times[n - 2]);
    (-(zeros[n - 1] * times[n - 1] + f * (t - times[n - 1]))).exp()
}

fn grid(a: f64, b: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut pts = vec![a, b];
    pts.extend(
        x.iter()
            .chain(y.iter())
            .copied()
            .filter(|&v| a < v && v < b),
    );
    pts.sort_by(f64::total_cmp);
    pts.dedup();
    pts
}

fn segment(lam: f64, f: f64, d: f64) -> (f64, f64) {
    let k = lam + f;
    let x = k * d;
    if x.abs() < 1e-8 {
        return (d * (1.0 - x / 2.0), d * d * (0.5 - x / 3.0));
    }
    let e = (-x).exp();
    ((1.0 - e) / k, (1.0 - e * (1.0 + x)) / (k * k))
}

fn check_curve(t: &[f64], v: &[f64], what: &str) -> Result<(), Error> {
    if t.is_empty() || t.len() != v.len() {
        return Err(Error::InvalidInput(format!(
            "{what}: times and values must be non-empty and equal length"
        )));
    }
    if t[0] <= 0.0 || t.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::InvalidInput(format!(
            "{what}: times must be positive and strictly increasing"
        )));
    }
    if t.iter().chain(v.iter()).any(|x| !x.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "{what}: values must be finite"
        )));
    }
    Ok(())
}

/// Coupon schedule of one contract, as times (see the Python docstring).
#[derive(Clone, Debug)]
pub struct Schedule<'a> {
    /// Accrual starts, hazard time (from the start of the step-in day).
    pub acc_start: &'a [f64],
    /// Accrual ends, hazard time.
    pub acc_end: &'a [f64],
    /// Payment dates, discount time (from the trade date).
    pub pay_time: &'a [f64],
    /// ACT/360 accrual fractions.
    pub acc_frac: &'a [f64],
}

/// `(protection PV, risky annuity incl. accrued-on-default, df to cash settlement)`.
///
/// # Errors
/// `InvalidInput` for malformed curves, a recovery outside `[0, 1)` or
/// schedule arrays of different lengths.
#[allow(clippy::too_many_arguments)]
pub fn isda_legs(
    hz_times: &[f64],
    hz_rates: &[f64],
    df_times: &[f64],
    df_zeros: &[f64],
    recovery: f64,
    t_end: f64,
    sched: &Schedule<'_>,
    t_cash: f64,
) -> Result<(f64, f64, f64), Error> {
    check_curve(hz_times, hz_rates, "hazard")?;
    check_curve(df_times, df_zeros, "discount")?;
    if !(0.0..1.0).contains(&recovery) {
        return Err(Error::InvalidInput("recovery must lie in [0, 1)".into()));
    }
    let n = sched.acc_start.len();
    if sched.acc_end.len() != n || sched.pay_time.len() != n || sched.acc_frac.len() != n {
        return Err(Error::InvalidInput(
            "schedule arrays must have equal length".into(),
        ));
    }
    let disc = |u: f64| df(df_times, df_zeros, u);
    let qp = |u: f64| (-cum_hazard(hz_times, hz_rates, u)).exp() * disc(u.max(0.0));
    let lam_f = |u0: f64, u1: f64| {
        let lam =
            (cum_hazard(hz_times, hz_rates, u1) - cum_hazard(hz_times, hz_rates, u0)) / (u1 - u0);
        let f = (disc(u0.max(0.0)) / disc(u1)).ln() / (u1 - u0);
        (lam, f)
    };

    let mut prot = 0.0;
    let g = grid(0.0, t_end, hz_times, df_times);
    for w in g.windows(2) {
        let (u0, u1) = (w[0], w[1]);
        let (lam, f) = lam_f(u0, u1);
        let (i0, _) = segment(lam, f, u1 - u0);
        prot += lam * qp(u0) * i0;
    }
    prot *= 1.0 - recovery;

    let mut prem = 0.0;
    for j in 0..n {
        let (s, e) = (sched.acc_start[j], sched.acc_end[j]);
        prem += sched.acc_frac[j]
            * (-cum_hazard(hz_times, hz_rates, e)).exp()
            * disc(sched.pay_time[j]);
        let g = grid(s.max(0.0), e, hz_times, df_times);
        let mut aod = 0.0;
        for w in g.windows(2) {
            let (u0, u1) = (w[0], w[1]);
            let (lam, f) = lam_f(u0, u1);
            let (i0, i1) = segment(lam, f, u1 - u0);
            aod += lam * qp(u0) * ((u0 - s + BIAS) * i0 + i1);
        }
        prem += 365.0 / 360.0 * aod;
    }
    Ok((prot, prem, disc(t_cash)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_hazard_has_no_protection() {
        let s = Schedule {
            acc_start: &[-0.1],
            acc_end: &[0.9],
            pay_time: &[0.9],
            acc_frac: &[1.0],
        };
        let (p, a, d) = isda_legs(&[50.0], &[0.0], &[50.0], &[0.03], 0.4, 0.9, &s, 0.01).unwrap();
        assert_eq!(p, 0.0);
        assert!((a - (-0.03f64 * 0.9).exp()).abs() < 1e-15);
        assert!((d - (-0.03f64 * 0.01).exp()).abs() < 1e-16);
    }

    #[test]
    fn rejects_bad_input() {
        let s = Schedule {
            acc_start: &[0.0],
            acc_end: &[1.0],
            pay_time: &[1.0],
            acc_frac: &[],
        };
        assert!(isda_legs(&[1.0], &[0.01], &[1.0], &[0.01], 0.4, 1.0, &s, 0.0).is_err());
        let s = Schedule {
            acc_start: &[0.0],
            acc_end: &[1.0],
            pay_time: &[1.0],
            acc_frac: &[1.0],
        };
        assert!(isda_legs(&[1.0], &[0.01], &[1.0], &[0.01], 1.0, 1.0, &s, 0.0).is_err());
        assert!(
            isda_legs(
                &[2.0, 1.0],
                &[0.01, 0.01],
                &[1.0],
                &[0.01],
                0.4,
                1.0,
                &s,
                0.0
            )
            .is_err()
        );
        assert!(isda_legs(&[1.0], &[0.01], &[1.0], &[f64::NAN], 0.4, 1.0, &s, 0.0).is_err());
    }
}
