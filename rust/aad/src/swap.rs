//! Swap PV off the dated bootstrap with adjoint curve and quote Greeks; the
//! port of `qp_core.aad.swap`.
//!
//! `x_j = ln P(days[j])`, log-linear against ACT/365F time with an implicit
//! node `ln P(0) = 0` and flat forward beyond the last pillar (the
//! interpolation of `qp_rates_curves::DatedCurve`). Quote Greeks use the
//! implicit function theorem on the bootstrap equations `R(x, q) = 0`:
//! `dPV/dq = g^T J^{-1}` with `J = dR/dx` and `g = dPV/dx` (Henrard 2014,
//! *Interest Rate Modelling in the Multi-Curve Framework*, ch. 4).

use crate::{Real, Tape, Var, linalg};
use qp_rates_curves::{FixedPeriod, FloatPeriod, Quote};

const DAYS_PER_YEAR: f64 = 365.0;

/// Fixed-against-float swap.
#[derive(Debug, Clone, PartialEq)]
pub struct Swap {
    /// Fixed coupons.
    pub fixed: Vec<FixedPeriod>,
    /// Float coupons.
    pub floating: Vec<FloatPeriod>,
    /// Fixed rate.
    pub rate: f64,
    /// Notional.
    pub notional: f64,
    /// Pay fixed when true.
    pub payer: bool,
}

/// `ln P(day)` from pillar log discount factors `xs` on `days`.
pub fn log_df<T: Real>(days: &[i64], xs: &[T], day: f64) -> T {
    let t = day / DAYS_PER_YEAR;
    let tt = |j: usize| {
        if j == 0 {
            0.0
        } else {
            days[j - 1] as f64 / DAYS_PER_YEAR
        }
    };
    let n = days.len();
    if t >= tt(n) {
        let w = (t - tt(n)) / (tt(n) - tt(n - 1));
        return if n == 1 {
            xs[0] + xs[0] * w
        } else {
            xs[n - 1] + (xs[n - 1] - xs[n - 2]) * w
        };
    }
    let (mut lo, mut hi) = (0usize, n);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if tt(mid) <= t { lo = mid } else { hi = mid }
    }
    let w = (t - tt(lo)) / (tt(hi) - tt(lo));
    if lo == 0 {
        xs[hi - 1] * w
    } else {
        xs[lo - 1] + (xs[hi - 1] - xs[lo - 1]) * w
    }
}

fn df<T: Real>(days: &[i64], xs: &[T], day: i64) -> T {
    log_df(days, xs, day as f64).exp()
}

fn legs<T: Real>(
    days: &[i64],
    xs: &[T],
    fixed: &[FixedPeriod],
    floating: &[FloatPeriod],
) -> (T, T) {
    let mut flt: Option<T> = None;
    for p in floating {
        let c = df(days, xs, p.pay) * ((df(days, xs, p.start) / df(days, xs, p.end)) - 1.0);
        flt = Some(flt.map_or(c, |a| a + c));
    }
    let mut ann: Option<T> = None;
    for f in fixed {
        let c = df(days, xs, f.pay) * f.accrual;
        ann = Some(ann.map_or(c, |a| a + c));
    }
    (
        flt.expect("float leg is empty"),
        ann.expect("fixed leg is empty"),
    )
}

/// Generic PV.
pub fn pv<T: Real>(days: &[i64], xs: &[T], swap: &Swap) -> T {
    let (flt, ann) = legs(days, xs, &swap.fixed, &swap.floating);
    let v = (flt - ann * swap.rate) * swap.notional;
    if swap.payer { v } else { -v }
}

/// Model minus quoted rate on a self-discounted, self-projected curve.
pub fn quote_residual<T: Real>(q: &Quote, days: &[i64], xs: &[T]) -> T {
    match q {
        Quote::Forward {
            start,
            end,
            accrual,
            rate,
        } => ((df(days, xs, *start) / df(days, xs, *end)) - 1.0) / *accrual - *rate,
        Quote::Swap {
            rate,
            fixed,
            floating,
        } => {
            let (flt, ann) = legs(days, xs, fixed, floating);
            flt / ann - *rate
        }
    }
}

/// Plain-float PV from discount factors.
pub fn pv_f64(days: &[i64], dfs: &[f64], swap: &Swap) -> f64 {
    let xs: Vec<f64> = dfs.iter().map(|d| d.ln()).collect();
    pv(days, &xs, swap)
}

/// PV and `dPV / d ln P(days[j])` from one backward sweep.
pub fn pillar_greeks(days: &[i64], dfs: &[f64], swap: &Swap) -> (f64, Vec<f64>) {
    let tape = Tape::with_capacity(dfs.len() + 40 * (swap.fixed.len() + swap.floating.len()));
    let xs: Vec<Var<'_>> = dfs.iter().map(|d| tape.var(d.ln())).collect();
    let v = pv(days, &xs, swap);
    (v.value(), tape.gradient(v, &xs))
}

/// `J[k][j] = d R_k / d ln P(days[j])`, row-major, one sweep per quote.
pub fn residual_jacobian(quotes: &[Quote], days: &[i64], dfs: &[f64]) -> Vec<f64> {
    let tape = Tape::new();
    let xs: Vec<Var<'_>> = dfs.iter().map(|d| tape.var(d.ln())).collect();
    let mark = tape.mark();
    let mut jac = Vec::with_capacity(quotes.len() * xs.len());
    for q in quotes {
        let r = quote_residual(q, days, &xs);
        jac.extend(tape.gradient(r, &xs));
        tape.rewind(mark);
    }
    jac
}

/// Quote Greeks of [`quote_greeks`].
#[derive(Debug, Clone, PartialEq)]
pub struct QuoteGreeks {
    /// PV.
    pub pv: f64,
    /// `dPV / d quote_k`, in quote order.
    pub d_quote: Vec<f64>,
    /// `dPV / d ln P(days[j])`.
    pub d_log_df: Vec<f64>,
}

/// `days`/`dfs` must be the curve bootstrapped from `quotes`, one pillar per
/// quote. `None` when the counts differ or the Jacobian is singular.
pub fn quote_greeks(
    quotes: &[Quote],
    days: &[i64],
    dfs: &[f64],
    swap: &Swap,
) -> Option<QuoteGreeks> {
    if quotes.len() != days.len() {
        return None;
    }
    let (v, g) = pillar_greeks(days, dfs, swap);
    let n = days.len();
    let jac = residual_jacobian(quotes, days, dfs);
    let mut jt = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            jt[j * n + i] = jac[i * n + j];
        }
    }
    let lam = linalg::solve(jt, g.clone())?;
    Some(QuoteGreeks {
        pv: v,
        d_quote: lam,
        d_log_df: g,
    })
}
