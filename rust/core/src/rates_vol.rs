//! Rates volatility: SABR implied vol and the Hull-White trinomial Bermudan.
//!
//! Rust port of `python/core/src/qp_core/vol/sabr.py` (`sabr_vol`) and of
//! `bermudan_tree_core` in `python/equity/04-fixed-income/src/fi_rates/bermudan.py`.
//! The C++ port is `cpp/core/include/qp/rates_vol.hpp`. Formulas, sources
//! and worked examples are in the Python docstrings; in short:
//!
//! * SABR: Hagan, Kumar, Lesniewski and Woodward (2002), eq. (2.17a)
//!   (`Variant::Hagan`), with the Obloj (2008) leading term
//!   (`Variant::Obloj`); `z / x(z)` by its series near `z = 0` (West 2005);
//!   shifted SABR on `F + shift`, `K + shift`.
//! * Bermudan: one-factor Hull-White trinomial tree with per-step spacing
//!   `sqrt(3 V_i)`, nearest-node branching, `alpha_i` fitted to the
//!   discount factors by forward induction (Brigo and Mercurio 2006,
//!   section 3.3.3; the construction of QuantLib's `TrinomialTree`), and a
//!   coupon bond rolled back next to the option.
//!
//! Every expression is in the same order as the Python, so the parity test
//! (`tests/rates_vol_parity.rs`, Python reference values) holds to a few
//! ulps; the Python side matches QuantLib's `TreeSwaptionEngine` to 1e-12.

use crate::Error;

const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Which SABR leading term to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Hagan et al. (2002) eq. (2.17a) as printed.
    Hagan,
    /// Obloj (2008) eq. (3.4) leading term (the Python default).
    Obloj,
}

fn z_over_x(z: f64, rho: f64) -> f64 {
    if z.abs() < 1e-6 {
        return 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0;
    }
    let root = (1.0 - 2.0 * rho * z + z * z).sqrt();
    let x = ((z + (z * z - 2.0 * rho * z) / (root + 1.0)) / (1.0 - rho)).ln_1p();
    z / x
}

/// Black (lognormal) implied vol of the SABR model at strike `k`.
///
/// # Errors
/// `InvalidInput` for non-finite inputs, `f + shift <= 0`, `k + shift <= 0`,
/// `t < 0`, `alpha <= 0`, `nu < 0`, `beta` outside `[0, 1]` or `|rho| >= 1`.
#[allow(clippy::too_many_arguments)]
pub fn sabr_vol(
    f: f64,
    k: f64,
    t: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    shift: f64,
    variant: Variant,
) -> Result<f64, Error> {
    for (name, v) in [
        ("F", f),
        ("K", k),
        ("T", t),
        ("alpha", alpha),
        ("beta", beta),
        ("rho", rho),
        ("nu", nu),
        ("shift", shift),
    ] {
        if !v.is_finite() {
            return Err(Error::InvalidInput(format!(
                "{name} must be finite, got {v}"
            )));
        }
    }
    if f + shift <= 0.0 || k + shift <= 0.0 {
        return Err(Error::InvalidInput(
            "F + shift and K + shift must be > 0".into(),
        ));
    }
    if t < 0.0
        || alpha <= 0.0
        || nu < 0.0
        || !(0.0..=1.0).contains(&beta)
        || !(-1.0 < rho && rho < 1.0)
    {
        return Err(Error::InvalidInput(
            "need T >= 0, alpha > 0, nu >= 0, 0 <= beta <= 1, -1 < rho < 1".into(),
        ));
    }
    let (fs, ks) = (f + shift, k + shift);
    let omb = 1.0 - beta;
    let log_fk = (fs / ks).ln();
    let fk_mid = (fs * ks).powf(0.5 * omb);
    let correction = 1.0
        + t * (omb * omb * alpha * alpha / (24.0 * fk_mid * fk_mid)
            + rho * beta * nu * alpha / (4.0 * fk_mid)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0);
    if variant == Variant::Hagan {
        let z = nu / alpha * fk_mid * log_fk;
        let denom =
            1.0 + omb.powi(2) / 24.0 * log_fk.powi(2) + omb.powi(4) / 1920.0 * log_fk.powi(4);
        return Ok(alpha / (fk_mid * denom) * z_over_x(z, rho) * correction);
    }
    let (z, lead) = if omb == 0.0 {
        (nu * log_fk / alpha, alpha)
    } else {
        let diff = ks.powf(omb) * (omb * log_fk).exp_m1();
        let z = nu * diff / (alpha * omb);
        let lead = if log_fk == 0.0 {
            alpha / ks.powf(omb)
        } else {
            alpha * omb * log_fk / diff
        };
        (z, lead)
    };
    Ok(lead * z_over_x(z, rho) * correction)
}

/// Bermudan on a Hull-White trinomial tree; all inputs on one time grid.
///
/// `times[0] = 0`, `dfs[i] = P(0, times[i])`, `cash[i]` the coupon-bond cash
/// flow at `times[i]`, `exercise[i]` an exercise date where a payer gets
/// `1 - B` (a receiver `B - 1`), `B` the coupon bond of later cash flows.
///
/// # Errors
/// `InvalidInput` on mismatched lengths, a grid that does not start at 0 or
/// increase strictly, or `a <= 0`, `sigma <= 0`.
// Index loops mirror the Python and C++ node arithmetic one for one.
#[allow(clippy::needless_range_loop)]
pub fn bermudan_tree(
    a: f64,
    sigma: f64,
    times: &[f64],
    dfs: &[f64],
    cash: &[f64],
    exercise: &[bool],
    payer: bool,
) -> Result<f64, Error> {
    if times.len() < 2
        || dfs.len() != times.len()
        || cash.len() != times.len()
        || exercise.len() != times.len()
    {
        return Err(Error::InvalidInput(
            "times, dfs, cash and exercise must have equal length >= 2".into(),
        ));
    }
    let n = times.len() - 1;
    if times[0] != 0.0 || times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::InvalidInput(
            "times must start at 0 and increase strictly".into(),
        ));
    }
    if !(a > 0.0 && sigma > 0.0) {
        return Err(Error::InvalidInput("need a > 0 and sigma > 0".into()));
    }
    let mut dx = vec![0.0; n + 1];
    let mut jmin = vec![0_i64; n + 1];
    let mut width = vec![1_usize; n + 1];
    let mut kk: Vec<Vec<i64>> = Vec::with_capacity(n);
    let mut pp: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n);
    for i in 0..n {
        let dt = times[i + 1] - times[i];
        let v2 = sigma * sigma * (1.0 - (-2.0 * a * dt).exp()) / (2.0 * a);
        let v = v2.sqrt();
        dx[i + 1] = v * SQRT3;
        let decay = (-a * dt).exp();
        let mut ks = Vec::with_capacity(width[i]);
        let mut ps = Vec::with_capacity(width[i]);
        for jj in 0..width[i] {
            let x = (jmin[i] + jj as i64) as f64 * dx[i];
            let m = x * decay;
            let k = (m / dx[i + 1] + 0.5).floor() as i64;
            let e = m - k as f64 * dx[i + 1];
            let e2 = e * e / v2;
            let e3 = e * SQRT3 / v;
            ks.push(k);
            ps.push([
                (1.0 + e2 - e3) / 6.0,
                (2.0 - e2) / 3.0,
                (1.0 + e2 + e3) / 6.0,
            ]);
        }
        let lo = *ks.iter().min().expect("non-empty level");
        let hi = *ks.iter().max().expect("non-empty level");
        jmin[i + 1] = lo - 1;
        width[i + 1] = (hi + 1 - jmin[i + 1] + 1) as usize;
        kk.push(ks);
        pp.push(ps);
    }
    let mut alpha = vec![0.0; n];
    let mut q = vec![1.0];
    for i in 0..n {
        let dt = times[i + 1] - times[i];
        let mut s = 0.0;
        for jj in 0..width[i] {
            let x = (jmin[i] + jj as i64) as f64 * dx[i];
            s += q[jj] * (-x * dt).exp();
        }
        alpha[i] = (s / dfs[i + 1]).ln() / dt;
        let mut qn = vec![0.0; width[i + 1]];
        // Same accumulation order as the Python (all p0 terms, then p1, then p2).
        let mut d = vec![0.0; width[i]];
        for (jj, dj) in d.iter_mut().enumerate() {
            let x = (jmin[i] + jj as i64) as f64 * dx[i];
            *dj = q[jj] * (-(x + alpha[i]) * dt).exp();
        }
        for b in 0..3 {
            for jj in 0..width[i] {
                let c = (kk[i][jj] - jmin[i + 1]) as usize + b - 1;
                qn[c] += d[jj] * pp[i][jj][b];
            }
        }
        q = qn;
    }
    let mut bond = vec![cash[n]; width[n]];
    let mut opt = vec![0.0; width[n]];
    for i in (0..n).rev() {
        let dt = times[i + 1] - times[i];
        let mut nb = vec![0.0; width[i]];
        let mut no = vec![0.0; width[i]];
        for jj in 0..width[i] {
            let x = (jmin[i] + jj as i64) as f64 * dx[i];
            let d = (-(x + alpha[i]) * dt).exp();
            let c = (kk[i][jj] - jmin[i + 1]) as usize;
            let [p0, p1, p2] = pp[i][jj];
            let b = d * (p0 * bond[c - 1] + p1 * bond[c] + p2 * bond[c + 1]);
            let mut o = d * (p0 * opt[c - 1] + p1 * opt[c] + p2 * opt[c + 1]);
            if exercise[i] {
                let pay = if payer { 1.0 - b } else { b - 1.0 };
                if pay > o {
                    o = pay;
                }
            }
            nb[jj] = b + cash[i];
            no[jj] = o;
        }
        bond = nb;
        opt = no;
    }
    Ok(opt[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docstring_worked_example() {
        let v = sabr_vol(0.03, 0.03, 5.0, 0.02, 0.5, -0.3, 0.4, 0.0, Variant::Obloj).unwrap();
        assert!((v - 0.121209).abs() < 5e-7);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(sabr_vol(0.03, -0.01, 1.0, 0.02, 0.5, 0.0, 0.3, 0.0, Variant::Hagan).is_err());
        assert!(sabr_vol(0.03, 0.03, 1.0, 0.02, 0.5, 1.0, 0.3, 0.0, Variant::Hagan).is_err());
        assert!(
            bermudan_tree(
                0.0,
                0.01,
                &[0.0, 1.0],
                &[1.0, 0.97],
                &[0.0, 1.0],
                &[true, false],
                true
            )
            .is_err()
        );
        assert!(
            bermudan_tree(
                0.1,
                0.01,
                &[0.0, 0.0],
                &[1.0, 0.97],
                &[0.0, 1.0],
                &[true, false],
                true
            )
            .is_err()
        );
    }
}
