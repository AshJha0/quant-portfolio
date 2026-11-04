//! Collateralised exposure under a CSA: the margin balance, the margin period
//! of risk lag and pathwise exposure (Rust parity D1).
//!
//! Rust port of `python/core/src/qp_core/xva/collateral.py`. The C++ port is
//! `cpp/core/include/qp/collateral.hpp`. Plain idea, formal statement, worked
//! examples and sources (Gregory, *The xVA Challenge*, ch. 8; BCBS CRE53) are
//! in the Python docstring; in short:
//!
//! * The target collateral is `max(V - H_c, 0) - max(-V - H_b, 0)`. The balance
//!   moves to the target only when the move is at least the MTA ([`collateral_balance`]).
//! * Collateral held at `t` is the balance at `t - MPOR`, or `initial` before the
//!   first date ([`lag_index`], [`lagged_grid`]).
//! * `E = max(V - C - IM_r, 0)` and `N = min(V - C + IM_p, 0)`
//!   ([`collateralised_exposure`]).
//!
//! Matrices are row-major `paths x times` slices. Parity:
//! `tests/frtb_collateral_parity.rs` reproduces the Python values in
//! `tests/data/frtb_collateral_parity.txt` to 1e-12 relative.
//!
//! Differences from the Python, none of which changes a value on valid input:
//! NaN is rejected here, as is an infinite value, time or `initial`; infinite
//! thresholds, MTA, MPOR and margin stay legal, because `H = infinity` is the
//! documented uncollateralised limit. `Csa` is checked when it is used, not when
//! it is built. Python's broadcastable margin arrays are the explicit
//! [`Margin`] shapes.

use crate::{Error, require_finite};

const EPS: f64 = 1e-12;

/// Initial margin, in the array shapes Python's broadcast is used with.
#[derive(Debug, Clone, PartialEq)]
pub enum Margin {
    /// One amount everywhere.
    Scalar(f64),
    /// One amount per time.
    ByTime(Vec<f64>),
    /// One amount per path.
    ByPath(Vec<f64>),
    /// Row-major `paths x times`.
    Full(Vec<f64>),
}

impl Default for Margin {
    fn default() -> Self {
        Margin::Scalar(0.0)
    }
}

impl Margin {
    fn check(&self, paths: usize, times: usize) -> Result<(), Error> {
        let (values, want): (&[f64], usize) = match self {
            Margin::Scalar(x) => (std::slice::from_ref(x), 1),
            Margin::ByTime(x) => (x, times),
            Margin::ByPath(x) => (x, paths),
            Margin::Full(x) => (x, paths * times),
        };
        if values.len() != want {
            return Err(Error::InvalidInput(
                "initial margin does not match (paths, times)".into(),
            ));
        }
        if values.iter().any(|x| x.is_nan() || *x < 0.0) {
            return Err(Error::InvalidInput(
                "initial margin must be non-negative".into(),
            ));
        }
        Ok(())
    }

    #[inline]
    fn at(&self, path: usize, time: usize, times: usize) -> f64 {
        match self {
            Margin::Scalar(x) => *x,
            Margin::ByTime(x) => x[time],
            Margin::ByPath(x) => x[path],
            Margin::Full(x) => x[path * times + time],
        }
    }
}

/// CSA terms. `mpor` is in the grid's time unit (years: 10 business days is
/// about 10/250). `initial` is the balance at `t_0`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Csa {
    /// Counterparty threshold `H_c`.
    pub threshold_cpty: f64,
    /// Bank threshold `H_b`.
    pub threshold_bank: f64,
    /// Minimum transfer amount.
    pub mta: f64,
    /// Margin period of risk.
    pub mpor: f64,
    /// Initial margin received.
    pub im_received: Margin,
    /// Initial margin posted.
    pub im_posted: Margin,
    /// Balance at `t_0`.
    pub initial: f64,
}

impl Csa {
    fn validate(&self) -> Result<(), Error> {
        let terms = [
            self.threshold_cpty,
            self.threshold_bank,
            self.mta,
            self.mpor,
        ];
        if terms.iter().any(|x| x.is_nan() || *x < 0.0) {
            return Err(Error::InvalidInput(
                "thresholds, MTA and MPOR must be non-negative".into(),
            ));
        }
        require_finite(self.initial, "initial")
    }

    /// Two-way target collateral `T(V)` (positive: the bank holds it).
    ///
    /// # Examples
    ///
    /// ```
    /// use qp_core::collateral::Csa;
    /// let csa = Csa { threshold_cpty: 3.0, threshold_bank: 1.0, ..Csa::default() };
    /// assert_eq!((csa.target_vm(10.0), csa.target_vm(-4.0)), (7.0, -3.0));
    /// ```
    #[inline]
    pub fn target_vm(&self, v: f64) -> f64 {
        (v - self.threshold_cpty).max(0.0) - (-v - self.threshold_bank).max(0.0)
    }
}

fn check_values(values: &[f64], paths: usize, times: usize) -> Result<(), Error> {
    if values.len() != paths * times {
        return Err(Error::InvalidInput(
            "values must be (paths, len(times))".into(),
        ));
    }
    values.iter().try_for_each(|v| require_finite(*v, "values"))
}

fn check_times(times: &[f64]) -> Result<(), Error> {
    if times.is_empty() {
        return Err(Error::InvalidInput("times must not be empty".into()));
    }
    times.iter().try_for_each(|t| require_finite(*t, "times"))?;
    if times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::InvalidInput(
            "times must be strictly increasing".into(),
        ));
    }
    Ok(())
}

/// Balance `B(t_k)` per path after each margin call, with the MTA rule.
/// `values` is row-major `paths x times`.
///
/// # Errors
///
/// [`Error::InvalidInput`] on bad CSA terms, a shape mismatch or a non-finite
/// value.
pub fn collateral_balance(
    values: &[f64],
    paths: usize,
    times: usize,
    csa: &Csa,
) -> Result<Vec<f64>, Error> {
    csa.validate()?;
    check_values(values, paths, times)?;
    let mut out = vec![0.0; values.len()];
    for p in 0..paths {
        let mut cur = csa.initial;
        for k in 0..times {
            let tgt = csa.target_vm(values[p * times + k]);
            if csa.mta <= 0.0 || (tgt - cur).abs() >= csa.mta {
                cur = tgt;
            }
            out[p * times + k] = cur;
        }
    }
    Ok(out)
}

/// `times` with every `t - mpor >= t_0` added, rounded to 12 decimals, sorted
/// and de-duplicated, so the MPOR lag lands exactly on grid dates.
///
/// # Errors
///
/// [`Error::InvalidInput`] when `times` is empty or holds a non-finite value,
/// or `mpor` is NaN.
///
/// # Examples
///
/// ```
/// let g = qp_core::collateral::lagged_grid(&[0.0, 1.0, 2.0], 0.25).unwrap();
/// assert_eq!(g, vec![0.0, 0.75, 1.0, 1.75, 2.0]);
/// ```
pub fn lagged_grid(times: &[f64], mpor: f64) -> Result<Vec<f64>, Error> {
    let Some(first) = times.first() else {
        return Err(Error::InvalidInput("times must not be empty".into()));
    };
    times.iter().try_for_each(|t| require_finite(*t, "times"))?;
    if mpor.is_nan() {
        return Err(Error::InvalidInput("mpor must not be NaN".into()));
    }
    let lagged = times.iter().map(|t| t - mpor).filter(|l| *l >= first - EPS);
    // numpy's round(x, 12): scale, round half to even, unscale.
    let mut all: Vec<f64> = times
        .iter()
        .copied()
        .chain(lagged)
        .map(|x| (x * 1e12).round_ties_even() / 1e12)
        .collect();
    all.sort_by(f64::total_cmp);
    all.dedup();
    Ok(all)
}

/// For each date, the index of the latest grid date `<= t - mpor`, or 0 if none.
///
/// # Errors
///
/// [`Error::InvalidInput`] when `times` is empty, non-finite or not strictly
/// increasing, or `mpor` is NaN.
///
/// # Examples
///
/// ```
/// let i = qp_core::collateral::lag_index(&[0.0, 1.0, 2.0], 1.5).unwrap();
/// assert_eq!(i, vec![0, 0, 0]);
/// ```
pub fn lag_index(times: &[f64], mpor: f64) -> Result<Vec<usize>, Error> {
    check_times(times)?;
    if mpor.is_nan() {
        return Err(Error::InvalidInput("mpor must not be NaN".into()));
    }
    Ok(times
        .iter()
        .map(|t| {
            let target = t - mpor + EPS;
            times.partition_point(|x| *x <= target).saturating_sub(1)
        })
        .collect())
}

/// Pathwise collateral, exposure and negative exposure, row-major
/// `paths x times`.
#[derive(Debug, Clone, PartialEq)]
pub struct CollateralisedExposure {
    /// Collateral held `C(t)`.
    pub collateral: Vec<f64>,
    /// Exposure `E(t)`.
    pub exposure: Vec<f64>,
    /// Negative exposure `N(t)`.
    pub negative_exposure: Vec<f64>,
}

/// Pathwise collateral `C(t)`, exposure `E(t)` and negative exposure `N(t)`.
///
/// # Errors
///
/// [`Error::InvalidInput`] on bad CSA terms, a shape mismatch, times that are
/// not strictly increasing, negative margin or a non-finite value.
///
/// # Examples
///
/// The Python docstring path: exposures are capped at the threshold.
///
/// ```
/// use qp_core::collateral::{Csa, collateralised_exposure};
/// let csa = Csa { threshold_cpty: 3.0, ..Csa::default() };
/// let e = collateralised_exposure(&[0.0, 1.0, 2.0], &[0.0, 10.0, 4.0], 1, &csa).unwrap();
/// assert_eq!(e.exposure, vec![0.0, 3.0, 3.0]);
/// ```
pub fn collateralised_exposure(
    times: &[f64],
    values: &[f64],
    paths: usize,
    csa: &Csa,
) -> Result<CollateralisedExposure, Error> {
    check_times(times)?;
    let n = times.len();
    let bal = collateral_balance(values, paths, n, csa)?;
    csa.im_received.check(paths, n)?;
    csa.im_posted.check(paths, n)?;
    let mut c = bal.clone();
    if csa.mpor > 0.0 {
        let lagged = lag_index(times, csa.mpor)?;
        for p in 0..paths {
            for (k, t) in times.iter().enumerate() {
                c[p * n + k] = if t - csa.mpor < times[0] - EPS {
                    csa.initial
                } else {
                    bal[p * n + lagged[k]]
                };
            }
        }
    }
    let mut exposure = vec![0.0; values.len()];
    let mut negative_exposure = vec![0.0; values.len()];
    for p in 0..paths {
        for k in 0..n {
            let i = p * n + k;
            exposure[i] = (values[i] - c[i] - csa.im_received.at(p, k, n)).max(0.0);
            negative_exposure[i] = (values[i] - c[i] + csa.im_posted.at(p, k, n)).min(0.0);
        }
    }
    Ok(CollateralisedExposure {
        collateral: c,
        exposure,
        negative_exposure,
    })
}
