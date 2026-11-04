//! FRTB sensitivities-based aggregation and the curvature charge (Rust
//! parity D1).
//!
//! Rust port of `aggregate` in `python/core/src/qp_core/capital/frtb_classes.py`
//! (with `scenario_corr`, `within_bucket` and `across_buckets` from
//! `frtb_sbm.py`) and of `curvature_charge` and `net_cvr` in `frtb_curvature.py`.
//! The C++ port is `cpp/core/include/qp/frtb.hpp`. Plain idea, formal statement,
//! worked examples and sources (BCBS d457, MAR21) are in the Python docstrings;
//! in short:
//!
//! * [`aggregate`]: inside a bucket `K_b = sqrt(max(0, WS' R WS))`, or
//!   `sum |WS|` for an "other" bucket; across buckets
//!   `sqrt(sum K_b^2 + sum gamma S_b S_c)` with `S_b` clipped to `[-K_b, K_b]`
//!   when that radicand is negative. All this under the three MAR21.6
//!   correlation scenarios; the charge is the worst.
//! * [`curvature_charge`]: the same shape on CVRs, with squared delta
//!   correlations and `psi` switching off a cross term when both sides are
//!   negative. Each bucket picks its up or down side.
//!
//! Parity: `tests/frtb_collateral_parity.rs` reproduces the Python values in
//! `tests/data/frtb_collateral_parity.txt` to 1e-12 relative.
//!
//! Three differences from the Python, none of which changes a value on valid
//! input. NaN and infinite inputs are rejected here, where Python lets them
//! through. Factor keys are sorted with `Ord` here and by `repr` in Python;
//! that only changes the order of a sum. Python 3.12 `sum` is compensated and
//! the sums here are plain. So if `K+` and `K-` agree to the last bit by
//! accident rather than exactly, the side chosen may differ; exact ties, the
//! ones the rule is written for, agree.

use std::collections::{BTreeMap, BTreeSet};

use crate::{Error, require_finite};

/// A MAR21.6 correlation scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// The correlations as given.
    Medium,
    /// `min(1.25 rho, 1)`.
    High,
    /// `max(2 rho - 1, 0.75 rho)`.
    Low,
}

/// The scenarios, in the order Python runs them.
pub const SCENARIOS: [Scenario; 3] = [Scenario::Medium, Scenario::High, Scenario::Low];

/// Charge under each scenario and `charge`, the largest of the three.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Charges {
    /// Medium-correlation charge.
    pub medium: f64,
    /// High-correlation charge.
    pub high: f64,
    /// Low-correlation charge.
    pub low: f64,
    /// The maximum of the three.
    pub charge: f64,
}

impl Charges {
    fn from_scenarios(r: [f64; 3]) -> Self {
        Self {
            medium: r[0],
            high: r[1],
            low: r[2],
            charge: r[0].max(r[1]).max(r[2]),
        }
    }
}

/// Correlation under a MAR21.6 scenario.
///
/// # Examples
///
/// ```
/// use qp_core::frtb::{Scenario, scenario_corr};
/// assert_eq!(scenario_corr(0.9, Scenario::High), 1.0);
/// assert_eq!(scenario_corr(0.4, Scenario::Low), 0.30000000000000004);
/// ```
#[inline]
pub fn scenario_corr(rho: f64, scenario: Scenario) -> f64 {
    match scenario {
        Scenario::Medium => rho,
        Scenario::High => (1.25 * rho).min(1.0),
        Scenario::Low => (2.0 * rho - 1.0).max(0.75 * rho),
    }
}

/// `(v' M) v` for a row-major `n x n` matrix.
fn quad(v: &[f64], m: &[f64]) -> f64 {
    let n = v.len();
    let mut out = 0.0;
    for (j, vj) in v.iter().enumerate() {
        let mut y = 0.0;
        for (i, vi) in v.iter().enumerate() {
            y += vi * m[i * n + j];
        }
        out += y * vj;
    }
    out
}

fn require_all_finite(values: &[f64], name: &str) -> Result<(), Error> {
    values.iter().try_for_each(|v| require_finite(*v, name))
}

/// `K_b = sqrt(max(0, WS' R WS))` (MAR21.4(4)). `rho` is row-major and its
/// diagonal is used as given.
///
/// # Errors
///
/// [`Error::InvalidInput`] on a shape mismatch or a non-finite entry.
pub fn within_bucket(ws: &[f64], rho: &[f64]) -> Result<f64, Error> {
    if rho.len() != ws.len() * ws.len() {
        return Err(Error::InvalidInput(
            "rho must be square and match ws".into(),
        ));
    }
    require_all_finite(ws, "ws")?;
    require_all_finite(rho, "rho")?;
    Ok(quad(ws, rho).max(0.0).sqrt())
}

/// `sqrt(sum K_b^2 + sum_{b != c} gamma S_b S_c)`, with the clipped `S_b`
/// fallback when the radicand is negative (MAR21.4(5)). `gamma` is row-major;
/// its diagonal is ignored.
///
/// # Errors
///
/// [`Error::InvalidInput`] on a shape mismatch or a non-finite entry.
pub fn across_buckets(k: &[f64], s: &[f64], gamma: &[f64]) -> Result<f64, Error> {
    let n = k.len();
    if s.len() != n || gamma.len() != n * n {
        return Err(Error::InvalidInput(
            "gamma must be square and match k and s".into(),
        ));
    }
    require_all_finite(k, "k")?;
    require_all_finite(s, "s")?;
    require_all_finite(gamma, "gamma")?;
    let mut g = gamma.to_vec();
    for i in 0..n {
        g[i * n + i] = 0.0;
    }
    let kk: f64 = k.iter().fold(0.0, |acc, x| acc + x * x);
    let mut q = kk + quad(s, &g);
    if q < 0.0 {
        // Not f64::clamp: it panics when a caller's k is negative, and
        // numpy's clip does not.
        let clipped: Vec<f64> = s
            .iter()
            .zip(k)
            .map(|(sv, kv)| sv.max(-kv).min(*kv))
            .collect();
        q = kk + quad(&clipped, &g);
    }
    Ok(q.max(0.0).sqrt())
}

/// Weighted sensitivities of one bucket and their medium correlations.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketInput {
    /// Weighted sensitivities.
    pub ws: Vec<f64>,
    /// Row-major `n x n` medium correlations; not read for an `other` bucket.
    pub rho: Vec<f64>,
    /// Aggregate as `sum |WS|` (MAR21.56, 21.79).
    pub other: bool,
}

/// Charge per MAR21.6 scenario plus their maximum. `gamma(b, c)` is the medium
/// cross-bucket correlation, asked for `b != c` in both orders. No buckets
/// gives zero.
///
/// # Errors
///
/// [`Error::InvalidInput`] on a shape mismatch or a non-finite input, including
/// a non-finite `gamma`.
///
/// # Examples
///
/// Two IG financials from the Python docstring: `WS = +50, -50`,
/// `rho = 0.35 x 0.999`.
///
/// ```
/// use qp_core::frtb::{BucketInput, aggregate};
/// use std::collections::BTreeMap;
/// let r = 0.35 * 0.999;
/// let b = BucketInput { ws: vec![50.0, -50.0], rho: vec![1.0, r, r, 1.0], other: false };
/// let c = aggregate(&BTreeMap::from([(3, b)]), |_, _| 0.0).unwrap();
/// assert!((c.medium - 57.02).abs() < 5e-3);
/// ```
pub fn aggregate<G>(buckets: &BTreeMap<i32, BucketInput>, gamma: G) -> Result<Charges, Error>
where
    G: Fn(i32, i32) -> f64,
{
    let n = buckets.len();
    let mut res = [0.0; 3];
    for (slot, sc) in res.iter_mut().zip(SCENARIOS) {
        let mut ks = Vec::with_capacity(n);
        let mut ss = Vec::with_capacity(n);
        for bi in buckets.values() {
            ks.push(if bi.other {
                require_all_finite(&bi.ws, "ws")?;
                bi.ws.iter().fold(0.0, |acc, w| acc + w.abs())
            } else {
                within_bucket(&bi.ws, &scenario_matrix(&bi.rho, bi.ws.len(), sc))?
            });
            ss.push(bi.ws.iter().sum::<f64>());
        }
        let mut g = vec![0.0; n * n];
        for (i, b) in buckets.keys().enumerate() {
            for (j, c) in buckets.keys().enumerate() {
                if i != j {
                    g[i * n + j] = scenario_corr(gamma(*b, *c), sc);
                }
            }
        }
        *slot = across_buckets(&ks, &ss, &g)?;
    }
    Ok(Charges::from_scenarios(res))
}

/// `rho` with the scenario applied off the diagonal. A wrong length is passed
/// through for [`within_bucket`] to refuse.
fn scenario_matrix(rho: &[f64], n: usize, sc: Scenario) -> Vec<f64> {
    rho.iter()
        .enumerate()
        .map(|(idx, r)| {
            if n > 0 && idx / n == idx % n {
                *r
            } else {
                scenario_corr(*r, sc)
            }
        })
        .collect()
}

/// `(CVR+, CVR-)` of one instrument (MAR21.5(2)). `rw * s` is the delta P&L of
/// the shock, so for a linear instrument both are 0.
///
/// # Examples
///
/// ```
/// assert_eq!(qp_core::frtb::net_cvr(10.0, 13.0, 7.0, 0.3, 10.0), (0.0, 0.0));
/// ```
#[inline]
pub fn net_cvr(v0: f64, v_up: f64, v_down: f64, rw: f64, s: f64) -> (f64, f64) {
    (v0 - v_up + rw * s, v0 - v_down - rw * s)
}

/// Bucket id, net keys, net `CVR+` and net `CVR-`.
type NetBucket<'a, K> = (i32, Vec<&'a K>, Vec<f64>, Vec<f64>);

/// One curvature row: a risk factor key with its `CVR+` and `CVR-`.
#[derive(Debug, Clone, PartialEq)]
pub struct CvrRow<K> {
    /// Risk factor key; rows with equal keys are summed.
    pub key: K,
    /// `CVR+`.
    pub up: f64,
    /// `CVR-`.
    pub down: f64,
}

#[inline]
fn psi(a: f64, b: f64) -> f64 {
    if a < 0.0 && b < 0.0 { 0.0 } else { 1.0 }
}

fn k_side<R>(c: &[f64], rho: &R) -> Result<f64, Error>
where
    R: Fn(usize, usize) -> f64,
{
    let mut q = c.iter().fold(0.0, |acc, x| {
        let p = x.max(0.0);
        acc + p * p
    });
    for (i, a) in c.iter().enumerate() {
        for (j, b) in c.iter().enumerate() {
            if i != j {
                let r = rho(i, j);
                require_finite(r, "delta_rho")?;
                q += r * a * b * psi(*a, *b);
            }
        }
    }
    Ok(q.max(0.0).sqrt())
}

/// Curvature charge per scenario and their maximum (MAR21.96-101).
///
/// `delta_rho(b, k, l)` and `delta_gamma(b, c)` are the medium *delta*
/// correlations, squared here. Buckets in `other` take the sum of the positive
/// CVRs on each side.
///
/// # Errors
///
/// [`Error::InvalidInput`] on a non-finite CVR or correlation.
///
/// # Examples
///
/// The Python docstring case: `CVR+ = 12`, `CVR- = 9`, one bucket.
///
/// ```
/// use qp_core::frtb::{CvrRow, curvature_charge};
/// use std::collections::{BTreeMap, BTreeSet};
/// let b = BTreeMap::from([(5, vec![CvrRow { key: "X", up: 12.0, down: 9.0 }])]);
/// let c = curvature_charge(&b, |_, _, _| 0.0, |_, _| 0.0, &BTreeSet::new()).unwrap();
/// assert_eq!(c.charge, 12.0);
/// ```
pub fn curvature_charge<K, R, G>(
    buckets: &BTreeMap<i32, Vec<CvrRow<K>>>,
    delta_rho: R,
    delta_gamma: G,
    other: &BTreeSet<i32>,
) -> Result<Charges, Error>
where
    K: Ord,
    R: Fn(i32, &K, &K) -> f64,
    G: Fn(i32, i32) -> f64,
{
    let mut nets: Vec<NetBucket<'_, K>> = Vec::with_capacity(buckets.len());
    for (b, rows) in buckets {
        let mut acc: BTreeMap<&K, (f64, f64)> = BTreeMap::new();
        for r in rows {
            require_finite(r.up, "CVR+")?;
            require_finite(r.down, "CVR-")?;
            let e = acc.entry(&r.key).or_insert((0.0, 0.0));
            e.0 += r.up;
            e.1 += r.down;
        }
        let keys = acc.keys().copied().collect();
        let ups = acc.values().map(|v| v.0).collect();
        let dns = acc.values().map(|v| v.1).collect();
        nets.push((*b, keys, ups, dns));
    }
    let mut res = [0.0; 3];
    for (slot, sc) in res.iter_mut().zip(SCENARIOS) {
        let mut ks = Vec::with_capacity(nets.len());
        let mut ws = Vec::with_capacity(nets.len());
        for (b, keys, ups, dns) in &nets {
            let (kp, km) = if other.contains(b) {
                let pos = |c: &[f64]| c.iter().fold(0.0, |acc: f64, x| acc + x.max(0.0));
                (pos(ups), pos(dns))
            } else {
                let rho = |i: usize, j: usize| {
                    let d = delta_rho(*b, keys[i], keys[j]);
                    scenario_corr(d * d, sc)
                };
                (k_side(ups, &rho)?, k_side(dns, &rho)?)
            };
            let (su, sd): (f64, f64) = (ups.iter().sum(), dns.iter().sum());
            let up_sel = kp > km || (kp == km && su > sd);
            ks.push(if up_sel { kp } else { km });
            ws.push(if up_sel { su } else { sd });
        }
        let mut q = ks.iter().fold(0.0, |acc, k| acc + k * k);
        for (i, (b, ..)) in nets.iter().enumerate() {
            for (j, (c, ..)) in nets.iter().enumerate() {
                if i != j {
                    let d = delta_gamma(*b, *c);
                    require_finite(d, "delta_gamma")?;
                    q += scenario_corr(d * d, sc) * ws[i] * ws[j] * psi(ws[i], ws[j]);
                }
            }
        }
        *slot = q.max(0.0).sqrt();
    }
    Ok(Charges::from_scenarios(res))
}
