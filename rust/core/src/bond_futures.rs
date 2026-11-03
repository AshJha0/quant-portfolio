//! Bond futures kernels: CME conversion factor, implied repo, net basis
//! and the basic delivery (switch) option.
//!
//! Rust port of the `*_kernel` functions in
//! `python/equity/04-fixed-income/src/fi_rates/bond_futures.py`; the C++
//! port is `cpp/core/include/qp/bond_futures.hpp`. Formulas and sources
//! (CME IR232, Burghardt *The Treasury Bond Basis*, Hull s. 6.2) are in
//! the Python docstring, which also lists the 20 published conversion
//! factors the Python side reproduces. Same order of operations as the
//! Python; `tests/credit_bondfut_parity.rs` checks the Python values.

use crate::Error;

const NODES: usize = 801;
const WIDTH: f64 = 8.0;

/// Unrounded CME conversion factor from the coupon (already rounded to
/// 1/8%), whole years `n` and months `z`; `quarterly` truncates `z` to a
/// quarter (TY, TN, US, UB).
pub fn cme_cf(coupon: f64, n: u32, z: u32, quarterly: bool) -> f64 {
    let (z, v) = if quarterly {
        let z = 3 * (z / 3);
        (z, if z < 7 { z } else { 3 })
    } else {
        (z, if z < 7 { z } else { z - 6 })
    };
    let v = f64::from(v);
    let a = 1.0 / 1.03f64.powf(v / 6.0);
    let b = coupon / 2.0 * (6.0 - v) / 6.0;
    let k = if z < 7 {
        1.0 / 1.03f64.powf(f64::from(2 * n))
    } else {
        1.0 / 1.03f64.powf(f64::from(2 * n + 1))
    };
    let d = coupon / 0.06 * (1.0 - k);
    a * (coupon / 2.0 + k + d) - b
}

/// Implied repo rate (simple interest) with at most one interim coupon.
///
/// # Errors
/// `InvalidInput` when the financing denominator is not positive.
pub fn implied_repo(
    dirty_now: f64,
    invoice: f64,
    coupon_cash: f64,
    tau: f64,
    tau_c: f64,
) -> Result<f64, Error> {
    let den = dirty_now * tau - coupon_cash * tau_c;
    if den.is_nan() || den <= 0.0 {
        return Err(Error::InvalidInput(
            "delivery must be after settlement".into(),
        ));
    }
    Ok((invoice + coupon_cash - dirty_now) / den)
}

/// Net basis: forward clean price at `repo` minus futures times CF.
pub fn net_basis(
    dirty_now: f64,
    coupon_cash: f64,
    tau: f64,
    tau_c: f64,
    repo: f64,
    accrued_delivery: f64,
    futures_x_cf: f64,
) -> f64 {
    let fwd_dirty = dirty_now * (1.0 + repo * tau) - coupon_cash * (1.0 + repo * tau_c);
    fwd_dirty - accrued_delivery - futures_x_cf
}

/// One deliverable as seen on the delivery date.
#[derive(Clone, Debug)]
pub struct Bond<'a> {
    /// Street-convention discount exponents (periods).
    pub exps: &'a [f64],
    /// Cash flows per 100.
    pub flows: &'a [f64],
    /// Accrued interest at delivery.
    pub accrued: f64,
    /// Coupons per year.
    pub freq: u32,
    /// Forward yield.
    pub fwd_yield: f64,
    /// Conversion factor.
    pub cf: f64,
}

fn ratio(b: &Bond<'_>, y: f64) -> f64 {
    let m = f64::from(b.freq);
    let mut p = 0.0;
    for (e, f) in b.exps.iter().zip(b.flows) {
        p += f * (1.0 + y / m).powf(-e);
    }
    (p - b.accrued) / b.cf
}

/// `(fair futures price, switch-option value)` under one parallel normal
/// yield shift with standard deviation `sd` (801-point trapezoid on +-8 sd).
///
/// # Errors
/// `InvalidInput` for a negative or NaN `sd` or an empty basket.
pub fn delivery_option(bonds: &[Bond<'_>], sd: f64) -> Result<(f64, f64), Error> {
    if sd.is_nan() || sd < 0.0 {
        return Err(Error::InvalidInput("sd must be non-negative".into()));
    }
    if bonds.is_empty() {
        return Err(Error::InvalidInput("empty basket".into()));
    }
    if sd == 0.0 {
        let best = bonds
            .iter()
            .map(|b| ratio(b, b.fwd_yield))
            .fold(f64::INFINITY, f64::min);
        return Ok((best, 0.0));
    }
    let h = 2.0 * WIDTH / (NODES - 1) as f64;
    let mut den = 0.0;
    let mut num = 0.0;
    let mut each = vec![0.0; bonds.len()];
    for j in 0..NODES {
        let x = -WIDTH + j as f64 * h;
        let w = (-0.5 * x * x).exp() * if j == 0 || j == NODES - 1 { 0.5 } else { 1.0 };
        let mut best = f64::INFINITY;
        for (i, b) in bonds.iter().enumerate() {
            let r = ratio(b, b.fwd_yield + sd * x);
            each[i] += w * r;
            best = best.min(r);
        }
        num += w * best;
        den += w;
    }
    let fair = num / den;
    let min_each = each.iter().map(|e| e / den).fold(f64::INFINITY, f64::min);
    Ok((fair, min_each - fair))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cme_brochure_ty_example() {
        // CME IR232 example 4: 3-3/4s of Nov 2018 into Dec 2008 TY -> 0.8357
        let cf = cme_cf(0.0375, 9, 11, true);
        assert_eq!((cf * 1e4 + 0.5).floor() / 1e4, 0.8357);
    }

    #[test]
    fn identical_bonds_have_no_option() {
        let b = Bond {
            exps: &[1.0, 2.0],
            flows: &[3.0, 103.0],
            accrued: 0.0,
            freq: 2,
            fwd_yield: 0.05,
            cf: 1.0,
        };
        let (_, opt) = delivery_option(&[b.clone(), b], 0.01).unwrap();
        assert!(opt.abs() < 1e-13);
        assert!(delivery_option(&[], 0.01).is_err());
        assert!(implied_repo(100.0, 100.0, 0.0, 0.0, 0.0).is_err());
    }
}
