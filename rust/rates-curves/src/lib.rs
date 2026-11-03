//! Dated OIS, multi-curve and cross-currency curve bootstrap (W13b).
//!
//! The Rust port of `fi_rates.bootstrap.bootstrap_dated`,
//! `fi_rates.multicurve.bootstrap_projection_dated` and
//! `fx_rates.xccy.bootstrap_xccy_discount`. Same algorithm, same loop order:
//!
//! * curve time `t = days / 365` (ACT/365F); `ln P` linear in `t` between
//!   pillars, anchored at `P(0) = 1`; beyond the last pillar the last
//!   segment's forward is held flat;
//! * forward quote: `(P(s)/P(e) - 1)/tau = R` (deposits, FRAs, futures with
//!   `R = (100 - price)/100 - convexity`);
//! * swap: `sum_j D(p_j)(P(s_j)/P(e_j) - 1) = K sum_i tau_i D(p_i)`; a
//!   compounded overnight leg telescopes to the same expression;
//! * cross-currency basis (constant notional, spread on the foreign leg):
//!   `V_f / X(s0) = V_d / D(s0)`;
//! * each pillar's `ln P` is found by bisection on `[ln 1e-8, ln 20]` until
//!   the bracket cannot shrink.
//!
//! The oracle is QuantLib 1.44 through the shared fixture
//! `python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt`
//! (`tests/fixture.rs`); the three languages are each checked against it,
//! not against one another.

#![forbid(unsafe_code)]
// `!(x > 0.0)` is deliberate: it rejects NaN as well as non-positive values.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::fmt;

/// Days per year of curve time (ACT/365F).
pub const DAYS_PER_YEAR: f64 = 365.0;

/// Error from curve construction or bootstrap.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveError(pub String);

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CurveError {}

fn err<T>(msg: impl Into<String>) -> Result<T, CurveError> {
    Err(CurveError(msg.into()))
}

/// Anything that gives a discount factor on a (possibly fractional) day.
pub trait DfSource {
    fn df(&self, day: f64) -> f64;
}

/// Log-linear discount curve on integer day pillars, flat forward beyond.
#[derive(Debug, Clone, PartialEq)]
pub struct DatedCurve {
    days: Vec<i64>,
    dfs: Vec<f64>,
    t: Vec<f64>,
    x: Vec<f64>,
}

impl DatedCurve {
    pub fn new(days: Vec<i64>, dfs: Vec<f64>) -> Result<Self, CurveError> {
        if days.is_empty() || days.len() != dfs.len() {
            return err("days and dfs must be non-empty and of equal length");
        }
        let mut prev = 0;
        for &d in &days {
            if d <= prev {
                return err("pillar days must be positive and strictly increasing");
            }
            prev = d;
        }
        if dfs.iter().any(|&x| !(x.is_finite() && x > 0.0)) {
            return err("discount factors must be finite and positive");
        }
        let mut t = vec![0.0];
        t.extend(days.iter().map(|&d| d as f64 / DAYS_PER_YEAR));
        let mut x = vec![0.0];
        x.extend(dfs.iter().map(|v| v.ln()));
        Ok(Self { days, dfs, t, x })
    }

    pub fn days(&self) -> &[i64] {
        &self.days
    }

    pub fn dfs(&self) -> &[f64] {
        &self.dfs
    }

    /// `ln P(day)`; `day` must be `>= 0` (NaN for a negative day).
    pub fn log_df(&self, day: f64) -> f64 {
        if !(day >= 0.0) {
            return f64::NAN;
        }
        let t = day / DAYS_PER_YEAR;
        let (ts, xs) = (&self.t, &self.x);
        let n = ts.len() - 1;
        if t >= ts[n] {
            let slope = (xs[n] - xs[n - 1]) / (ts[n] - ts[n - 1]);
            return xs[n] + slope * (t - ts[n]);
        }
        let (mut lo, mut hi) = (0usize, n);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if ts[mid] <= t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let w = (t - ts[lo]) / (ts[hi] - ts[lo]);
        xs[lo] + w * (xs[hi] - xs[lo])
    }

    /// Instantaneous forward just after `day` (piecewise flat).
    pub fn forward_rate(&self, day: f64) -> f64 {
        let t = day / DAYS_PER_YEAR;
        let k = (1..self.t.len())
            .find(|&i| t < self.t[i])
            .unwrap_or(self.t.len() - 1);
        -(self.x[k] - self.x[k - 1]) / (self.t[k] - self.t[k - 1])
    }
}

impl DfSource for DatedCurve {
    fn df(&self, day: f64) -> f64 {
        self.log_df(day).exp()
    }
}

/// Fixed coupon paid on `pay` with accrual fraction `accrual`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedPeriod {
    pub pay: i64,
    pub accrual: f64,
}

/// Float coupon projected on `[start, end]`, paid on `pay`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatPeriod {
    pub start: i64,
    pub end: i64,
    pub pay: i64,
    pub accrual: f64,
}

/// A quote for [`bootstrap_dated`].
#[derive(Debug, Clone, PartialEq)]
pub enum Quote {
    /// Simple rate on `[start, end]` with accrual `accrual`.
    Forward {
        start: i64,
        end: i64,
        accrual: f64,
        rate: f64,
    },
    /// Par swap with fixed `rate`.
    Swap {
        rate: f64,
        fixed: Vec<FixedPeriod>,
        floating: Vec<FloatPeriod>,
    },
}

/// `(100 - price)/100 - convexity`.
pub fn futures_rate(price: f64, convexity: f64) -> f64 {
    (100.0 - price) / 100.0 - convexity
}

/// Hull-White futures convexity bias (QuantLib `HullWhite::convexityBias`).
pub fn hull_white_convexity(
    price: f64,
    t1: f64,
    t2: f64,
    sigma: f64,
    a: f64,
) -> Result<f64, CurveError> {
    if !(t2 > t1 && t1 >= 0.0) || sigma < 0.0 || a < 0.0 {
        return err("need 0 <= t1 < t2, sigma >= 0 and a >= 0");
    }
    let dt = t2 - t1;
    let (b_dt, b_t1, var) = if a < 1e-12 {
        (dt, t1, 2.0 * t1)
    } else {
        (
            (1.0 - (-a * dt).exp()) / a,
            (1.0 - (-a * t1).exp()) / a,
            (1.0 - (-2.0 * a * t1).exp()) / a,
        )
    };
    let half_s2 = 0.5 * sigma * sigma;
    let z = half_s2 * var * b_dt * b_dt + half_s2 * b_dt * b_t1 * b_t1;
    Ok((1.0 - (-z).exp()) * ((100.0 - price) / 100.0 + 1.0 / dt))
}

/// Model minus quoted rate.
pub fn quote_residual(q: &Quote, discount: &dyn DfSource, projection: &dyn DfSource) -> f64 {
    match q {
        Quote::Forward {
            start,
            end,
            accrual,
            rate,
        } => (projection.df(*start as f64) / projection.df(*end as f64) - 1.0) / accrual - rate,
        Quote::Swap {
            rate,
            fixed,
            floating,
        } => {
            let mut flt = 0.0;
            for p in floating {
                flt += discount.df(p.pay as f64)
                    * (projection.df(p.start as f64) / projection.df(p.end as f64) - 1.0);
            }
            let mut ann = 0.0;
            for f in fixed {
                ann += f.accrual * discount.df(f.pay as f64);
            }
            flt / ann - rate
        }
    }
}

/// Bisection on `[ln 1e-8, ln 20]` to floating-point resolution.
pub fn bisect_log_df(f: impl Fn(f64) -> f64) -> Result<(f64, u32), CurveError> {
    let (mut lo, mut hi) = (1e-8f64.ln(), 20f64.ln());
    let (mut flo, mut fhi) = (f(lo), f(hi));
    if !(flo.is_finite() && fhi.is_finite()) || (flo > 0.0) == (fhi > 0.0) {
        return err(format!(
            "no sign change on the bracket (f(lo)={flo:.3e}, f(hi)={fhi:.3e})"
        ));
    }
    let mut it = 0u32;
    while it < 200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        it += 1;
        let fm = f(mid);
        if fm == 0.0 {
            return Ok((mid, it));
        }
        if (fm > 0.0) == (flo > 0.0) {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
            fhi = fm;
        }
    }
    Ok((if flo.abs() <= fhi.abs() { lo } else { hi }, it))
}

/// One line of the bootstrap report.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub pillar: i64,
    pub df: f64,
    pub residual: f64,
    pub iterations: u32,
}

/// Bootstrapped curve plus quote-level diagnostics in pillar order.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    pub curve: DatedCurve,
    pub diagnostics: Vec<Diagnostic>,
}

impl BootstrapResult {
    pub fn max_abs_residual(&self) -> f64 {
        self.diagnostics
            .iter()
            .map(|d| d.residual.abs())
            .fold(0.0, f64::max)
    }
}

fn validate(q: &Quote) -> Result<(), CurveError> {
    match q {
        Quote::Forward {
            start,
            end,
            accrual,
            rate,
        } => {
            if !(0 <= *start && start < end) || !(*accrual > 0.0) || !rate.is_finite() {
                return err("bad forward quote");
            }
        }
        Quote::Swap {
            rate,
            fixed,
            floating,
        } => {
            if !rate.is_finite() || fixed.is_empty() || floating.is_empty() {
                return err("bad swap quote: need a finite rate and both legs");
            }
            if fixed.iter().any(|f| f.pay < 0 || !(f.accrual > 0.0)) {
                return err("bad fixed period");
            }
            if floating
                .iter()
                .any(|p| !(0 <= p.start && p.start < p.end) || p.pay < 0 || !(p.accrual > 0.0))
            {
                return err("bad float period");
            }
        }
    }
    Ok(())
}

fn pillar(q: &Quote, self_proj: bool, self_disc: bool) -> i64 {
    match q {
        Quote::Forward { end, .. } => *end,
        Quote::Swap {
            fixed, floating, ..
        } => {
            let mut m = i64::MIN;
            if self_proj {
                m = floating.iter().map(|p| p.end).fold(m, i64::max);
            }
            if self_disc {
                m = floating.iter().map(|p| p.pay).fold(m, i64::max);
                m = fixed.iter().map(|p| p.pay).fold(m, i64::max);
            }
            m
        }
    }
}

/// Bootstrap a dated curve. `None` for `discount`/`projection` means the
/// curve being built (an OIS curve passes both as `None`).
pub fn bootstrap_dated(
    quotes: &[Quote],
    discount: Option<&dyn DfSource>,
    projection: Option<&dyn DfSource>,
) -> Result<BootstrapResult, CurveError> {
    if quotes.is_empty() {
        return err("no quotes supplied to bootstrap");
    }
    if discount.is_some() && projection.is_some() {
        return err("at least one of discount/projection must be the curve being built");
    }
    for q in quotes {
        validate(q)?;
    }
    let mut keyed: Vec<(i64, usize)> = quotes
        .iter()
        .enumerate()
        .map(|(i, q)| (pillar(q, projection.is_none(), discount.is_none()), i))
        .collect();
    keyed.sort();
    for w in keyed.windows(2) {
        if w[0].0 == w[1].0 {
            return err(format!("two quotes share pillar day {}", w[0].0));
        }
    }
    if keyed[0].0 <= 0 {
        return err("pillars must be after the valuation date");
    }
    let mut days: Vec<i64> = Vec::new();
    let mut dfs: Vec<f64> = Vec::new();
    let mut iters = Vec::new();
    for &(p, i) in &keyed {
        let q = &quotes[i];
        let f = |x: f64| {
            let mut d = days.clone();
            d.push(p);
            let mut v = dfs.clone();
            v.push(x.exp());
            let trial = DatedCurve::new(d, v).expect("valid trial curve");
            quote_residual(q, discount.unwrap_or(&trial), projection.unwrap_or(&trial))
        };
        let (x, n) = bisect_log_df(f)
            .map_err(|e| CurveError(format!("bootstrap failed at pillar day {p}: {e}")))?;
        days.push(p);
        dfs.push(x.exp());
        iters.push(n);
    }
    let curve = DatedCurve::new(days, dfs)?;
    let diagnostics = keyed
        .iter()
        .enumerate()
        .map(|(k, &(p, i))| Diagnostic {
            pillar: p,
            df: curve.dfs[k],
            residual: quote_residual(
                &quotes[i],
                discount.unwrap_or(&curve),
                projection.unwrap_or(&curve),
            ),
            iterations: iters[k],
        })
        .collect();
    Ok(BootstrapResult { curve, diagnostics })
}

/// Cross-currency constant-notional basis swap quote.
#[derive(Debug, Clone, PartialEq)]
pub struct XccyBasisQuote {
    pub basis: f64,
    pub start: i64,
    pub end: i64,
    pub foreign: Vec<FloatPeriod>,
    pub domestic: Vec<FloatPeriod>,
}

impl XccyBasisQuote {
    pub fn pillar(&self) -> i64 {
        self.foreign.iter().map(|p| p.pay).fold(self.end, i64::max)
    }
}

fn leg_value(
    periods: &[FloatPeriod],
    spread: f64,
    start: i64,
    end: i64,
    disc: &dyn DfSource,
    proj: &dyn DfSource,
) -> (f64, f64) {
    let mut v = -disc.df(start as f64) + disc.df(end as f64);
    let mut ann = 0.0;
    for p in periods {
        let d = disc.df(p.pay as f64);
        v += d * (proj.df(p.start as f64) / proj.df(p.end as f64) - 1.0 + spread * p.accrual);
        ann += p.accrual * d;
    }
    (v, ann)
}

/// Par error in spread units: `(V_f/X(s0) - V_d/D(s0)) / A_f`.
pub fn xccy_basis_residual(
    q: &XccyBasisQuote,
    fdisc: &dyn DfSource,
    fproj: &dyn DfSource,
    dom: &dyn DfSource,
) -> f64 {
    let (vf, af) = leg_value(&q.foreign, q.basis, q.start, q.end, fdisc, fproj);
    let (vd, _) = leg_value(&q.domestic, 0.0, q.start, q.end, dom, dom);
    let (x0, d0) = (fdisc.df(q.start as f64), dom.df(q.start as f64));
    (vf / x0 - vd / d0) / (af / x0)
}

/// Foreign discount curve under domestic collateral from basis swaps.
pub fn bootstrap_xccy_discount(
    quotes: &[XccyBasisQuote],
    foreign_projection: &dyn DfSource,
    domestic: &dyn DfSource,
) -> Result<BootstrapResult, CurveError> {
    if quotes.is_empty() {
        return err("no basis quotes supplied");
    }
    for q in quotes {
        if !q.basis.is_finite()
            || !(0 <= q.start && q.start < q.end)
            || q.foreign.is_empty()
            || q.domestic.is_empty()
        {
            return err("bad basis quote");
        }
    }
    let mut keyed: Vec<(i64, usize)> = quotes
        .iter()
        .enumerate()
        .map(|(i, q)| (q.pillar(), i))
        .collect();
    keyed.sort();
    for w in keyed.windows(2) {
        if w[0].0 == w[1].0 {
            return err(format!("two quotes share pillar day {}", w[0].0));
        }
    }
    let mut days: Vec<i64> = Vec::new();
    let mut dfs: Vec<f64> = Vec::new();
    let mut iters = Vec::new();
    for &(p, i) in &keyed {
        let q = &quotes[i];
        let f = |x: f64| {
            let mut d = days.clone();
            d.push(p);
            let mut v = dfs.clone();
            v.push(x.exp());
            let trial = DatedCurve::new(d, v).expect("valid trial curve");
            xccy_basis_residual(q, &trial, foreign_projection, domestic)
        };
        let (x, n) = bisect_log_df(f)
            .map_err(|e| CurveError(format!("xccy bootstrap failed at pillar day {p}: {e}")))?;
        days.push(p);
        dfs.push(x.exp());
        iters.push(n);
    }
    let curve = DatedCurve::new(days, dfs)?;
    let diagnostics = keyed
        .iter()
        .enumerate()
        .map(|(k, &(p, i))| Diagnostic {
            pillar: p,
            df: curve.dfs[k],
            residual: xccy_basis_residual(&quotes[i], &curve, foreign_projection, domestic),
            iterations: iters[k],
        })
        .collect();
    Ok(BootstrapResult { curve, diagnostics })
}
