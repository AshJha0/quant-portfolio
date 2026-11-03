//! Capital and xVA kernels: SA-CCR (BCBS 279 / CRE52), Hull-White bond and
//! swaption closed forms, and the CVA / DVA / FVA / effective-EPE sums.
//!
//! Ports of `qp_core.capital.saccr`, `qp_core.xva.hull_white`,
//! `qp_core.xva.exposure` and `qp_core.xva.adjustments`. The Python modules
//! hold the derivations and sources; here every function takes plain numbers
//! (discount factors rather than a curve object) so the three languages can
//! be compared on one fixture file (`tests/data/xva_capital_parity.txt`).
//!
//! ```
//! use qp_core::xva::saccr_ir_effective_notional;
//! // BCBS 279 Annex 4a, Example 1, hedging set USD: 59,270 thousand.
//! assert_eq!(saccr_ir_effective_notional(0.0, -36254.0, 78694.0).round(), 59270.0);
//! ```

use crate::Error;
use crate::special::norm_cdf;

type Result<T> = std::result::Result<T, Error>;

const BD_PER_YEAR: f64 = 250.0;

fn bad(msg: &str) -> Error {
    Error::InvalidInput(msg.to_string())
}

// ------------------------------------------------------------------ SA-CCR

/// Supervisory duration (exp(-0.05 S) - exp(-0.05 E)) / 0.05, S and E in
/// years floored at ten business days (S = 0 stays 0).
pub fn saccr_supervisory_duration(start: f64, end: f64) -> Result<f64> {
    if !start.is_finite() || !end.is_finite() {
        return Err(bad("start and end must be finite"));
    }
    let s = if start > 0.0 {
        start.max(10.0 / BD_PER_YEAR)
    } else {
        0.0
    };
    let e = end.max(10.0 / BD_PER_YEAR);
    if e < s {
        return Err(bad("end must not precede start"));
    }
    Ok(((-0.05 * s).exp() - (-0.05 * e).exp()) / 0.05)
}

/// Supervisory option delta (CRE52.40) with shift `lambda`.
pub fn saccr_supervisory_delta(
    price: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    call: bool,
    bought: bool,
    shift: f64,
) -> Result<f64> {
    let (p, k) = (price + shift, strike + shift);
    if !(p > 0.0 && k > 0.0) {
        return Err(bad("shifted price and strike must be positive"));
    }
    if !(vol > 0.0 && expiry > 0.0) {
        return Err(bad("vol and expiry must be positive"));
    }
    let srt = vol * expiry.sqrt();
    let d1 = ((p / k).ln() + 0.5 * vol * vol * expiry) / srt;
    let d = if call { norm_cdf(d1) } else { -norm_cdf(-d1) };
    Ok(if bought { d } else { -d })
}

/// Maturity factor: unmargined sqrt(min(max(M, 10/250), 1)); margined
/// (`mpor_days` = Some) 1.5 sqrt(MPOR / 250).
pub fn saccr_maturity_factor(maturity: f64, mpor_days: Option<f64>) -> Result<f64> {
    match mpor_days {
        Some(m) if m > 0.0 => Ok(1.5 * (m / BD_PER_YEAR).sqrt()),
        Some(_) => Err(bad("mpor must be positive")),
        None if maturity.is_finite() => Ok(maturity.clamp(10.0 / BD_PER_YEAR, 1.0).sqrt()),
        None => Err(bad("maturity must be finite")),
    }
}

/// PFE multiplier min(1, 0.05 + 0.95 exp((V - C) / (2 * 0.95 * AddOn))).
pub fn saccr_multiplier(v: f64, c: f64, addon: f64) -> Result<f64> {
    if addon < 0.0 {
        return Err(bad("addon must be non-negative"));
    }
    if addon == 0.0 {
        return Ok(1.0);
    }
    let f = 0.05;
    Ok((f + (1.0 - f) * ((v - c) / (2.0 * (1.0 - f) * addon)).exp()).min(1.0))
}

/// Replacement cost; `margin` = Some((TH, MTA, NICA)) for a margined set.
pub fn saccr_replacement_cost(v: f64, c: f64, margin: Option<(f64, f64, f64)>) -> f64 {
    let rc = (v - c).max(0.0);
    match margin {
        Some((th, mta, nica)) => rc.max(th + mta - nica),
        None => rc,
    }
}

/// Interest-rate hedging-set effective notional over the three buckets.
pub fn saccr_ir_effective_notional(d1: f64, d2: f64, d3: f64) -> f64 {
    let q = d1 * d1 + d2 * d2 + d3 * d3 + 1.4 * d1 * d2 + 1.4 * d2 * d3 + 0.6 * d1 * d3;
    q.max(0.0).sqrt()
}

/// Single-factor aggregation sqrt((sum rho A)^2 + sum (1 - rho^2) A^2).
pub fn saccr_single_factor_addon(addons: &[f64], rhos: &[f64]) -> Result<f64> {
    if addons.len() != rhos.len() {
        return Err(bad("addons and rhos must have equal length"));
    }
    let (mut sys, mut idio) = (0.0, 0.0);
    for (a, r) in addons.iter().zip(rhos) {
        sys += r * a;
        idio += (1.0 - r * r) * a * a;
    }
    Ok((sys * sys + idio).sqrt())
}

// -------------------------------------------------------------- Hull-White

/// Hull-White one-factor parameters (mean reversion `a`, volatility `sigma`).
#[derive(Debug, Clone, Copy)]
pub struct HullWhite {
    /// Mean reversion, > 0.
    pub a: f64,
    /// Short-rate volatility, > 0.
    pub sigma: f64,
}

impl HullWhite {
    /// Validated constructor.
    pub fn new(a: f64, sigma: f64) -> Result<Self> {
        if !(a > 0.0 && a.is_finite() && sigma > 0.0 && sigma.is_finite()) {
            return Err(bad("a and sigma must be positive and finite"));
        }
        Ok(Self { a, sigma })
    }

    /// B(t, T) = (1 - exp(-a (T - t))) / a.
    pub fn b(&self, t: f64, big_t: f64) -> f64 {
        (1.0 - (-self.a * (big_t - t)).exp()) / self.a
    }

    /// V(t, T), the variance of int_t^T x ds.
    pub fn v(&self, t: f64, big_t: f64) -> f64 {
        let (a, s, tau) = (self.a, self.sigma, big_t - t);
        (s * s / (a * a))
            * (tau + 2.0 * (-a * tau).exp() / a - (-2.0 * a * tau).exp() / (2.0 * a) - 1.5 / a)
    }

    /// ln A(t, T) given P(0, t) = `p_t` and P(0, T) = `p_big_t`.
    pub fn log_a(&self, t: f64, big_t: f64, p_t: f64, p_big_t: f64) -> f64 {
        (p_big_t / p_t).ln() + 0.5 * (self.v(t, big_t) - self.v(0.0, big_t) + self.v(0.0, t))
    }

    fn sigma_p(&self, big_t: f64, s: f64) -> f64 {
        let a = self.a;
        self.sigma * ((1.0 - (-2.0 * a * big_t).exp()) / (2.0 * a)).sqrt() * self.b(big_t, s)
    }

    /// Time-0 option on a zero-coupon bond: expiry T, bond maturity S,
    /// P(0,T) = `p_t`, P(0,S) = `p_s`.
    pub fn zcb_option(
        &self,
        big_t: f64,
        s: f64,
        p_t: f64,
        p_s: f64,
        strike: f64,
        call: bool,
    ) -> Result<f64> {
        if !(0.0 < big_t && big_t < s && strike > 0.0) {
            return Err(bad("need 0 < T < S and strike > 0"));
        }
        let sp = self.sigma_p(big_t, s);
        let h = (p_s / (p_t * strike)).ln() / sp + 0.5 * sp;
        Ok(if call {
            p_s * norm_cdf(h) - strike * p_t * norm_cdf(h - sp)
        } else {
            strike * p_t * norm_cdf(-h + sp) - p_s * norm_cdf(-h)
        })
    }

    /// European swaption by Jamshidian; `p_pay[i]` = P(0, pay_times[i]).
    #[allow(clippy::too_many_arguments)]
    pub fn swaption(
        &self,
        expiry: f64,
        p_exp: f64,
        pay_times: &[f64],
        p_pay: &[f64],
        accruals: &[f64],
        strike: f64,
        payer: bool,
    ) -> Result<f64> {
        let n = pay_times.len();
        if n == 0 || p_pay.len() != n || accruals.len() != n {
            return Err(bad(
                "pay_times, p_pay and accruals must be non-empty and equal length",
            ));
        }
        if expiry.is_nan() || expiry <= 0.0 || pay_times.iter().any(|&t| t <= expiry) {
            return Err(bad("pay times must follow expiry > 0"));
        }
        let mut c: Vec<f64> = accruals.iter().map(|tau| strike * tau).collect();
        c[n - 1] += 1.0;
        if c.iter().any(|&ci| ci <= 0.0) {
            return Err(bad("Jamshidian needs positive coupons"));
        }
        let la: Vec<f64> = (0..n)
            .map(|i| self.log_a(expiry, pay_times[i], p_exp, p_pay[i]))
            .collect();
        let bb: Vec<f64> = (0..n).map(|i| self.b(expiry, pay_times[i])).collect();
        let f = |x: f64| -> f64 {
            let mut acc = 0.0;
            for i in 0..n {
                acc += c[i] * (la[i] - bb[i] * x).exp();
            }
            acc - 1.0
        };
        let (mut lo, mut hi, mut k) = (-1.0_f64, 1.0_f64, 0);
        while f(lo) < 0.0 {
            lo *= 2.0;
            k += 1;
            if k > 60 {
                return Err(Error::Numerical("Jamshidian bracket failed".into()));
            }
        }
        while f(hi) > 0.0 {
            hi *= 2.0;
            k += 1;
            if k > 60 {
                return Err(Error::Numerical("Jamshidian bracket failed".into()));
            }
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi {
                break;
            }
            if f(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let xstar = 0.5 * (lo + hi);
        let mut total = 0.0;
        for i in 0..n {
            let xi = (la[i] - bb[i] * xstar).exp();
            total += c[i] * self.zcb_option(expiry, pay_times[i], p_exp, p_pay[i], xi, !payer)?;
        }
        Ok(total)
    }
}

// --------------------------------------------------------------------- xVA

/// Survival to `t` under a piecewise-flat hazard (`hazards[i]` on
/// (times[i-1], times[i]], the last extended flat).
pub fn survival(times: &[f64], hazards: &[f64], t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    let (mut acc, mut prev) = (0.0, 0.0);
    for (&ti, &hi) in times.iter().zip(hazards) {
        if t <= ti {
            return (-(acc + hi * (t - prev))).exp();
        }
        acc += hi * (ti - prev);
        prev = ti;
    }
    (-(acc + hazards[hazards.len() - 1] * (t - prev))).exp()
}

/// Piecewise-flat hazard curve borrowed as two slices.
#[derive(Debug, Clone, Copy)]
pub struct Hazard<'a> {
    /// Segment end times, increasing.
    pub times: &'a [f64],
    /// Hazard on each segment.
    pub hazards: &'a [f64],
}

impl Hazard<'_> {
    /// Survival probability to `t`.
    pub fn survival(&self, t: f64) -> f64 {
        survival(self.times, self.hazards, t)
    }
}

fn check_grid(times: &[f64], prof: &[f64]) -> Result<()> {
    if times.len() != prof.len() || times.len() < 2 || times[0] != 0.0 {
        return Err(bad("times must start at 0 and match the profile"));
    }
    if times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(bad("times must increase strictly"));
    }
    Ok(())
}

/// (1 - R) sum EE*(t_k) [S(t_{k-1}) - S(t_k)] w(t_k); `other` gives
/// first-to-default weighting. Pass -ENE* as `expo` for DVA.
pub fn default_leg(
    times: &[f64],
    expo: &[f64],
    who: Hazard<'_>,
    recovery: f64,
    other: Option<Hazard<'_>>,
) -> Result<f64> {
    if !(0.0..=1.0).contains(&recovery) {
        return Err(bad("recovery must be in [0, 1]"));
    }
    check_grid(times, expo)?;
    let mut acc = 0.0;
    for k in 1..times.len() {
        let dpd = who.survival(times[k - 1]) - who.survival(times[k]);
        let w = other.map_or(1.0, |o| o.survival(times[k]));
        acc += expo[k] * dpd * w;
    }
    Ok((1.0 - recovery) * acc)
}

/// (FCA, FBA) with symmetric spread funding.
pub fn fva(
    times: &[f64],
    ee_disc: &[f64],
    ene_disc: &[f64],
    borrow: f64,
    lend: f64,
    cpty: Hazard<'_>,
    bank: Hazard<'_>,
) -> Result<(f64, f64)> {
    check_grid(times, ee_disc)?;
    check_grid(times, ene_disc)?;
    let (mut fca, mut fba) = (0.0, 0.0);
    for k in 1..times.len() {
        let w = (times[k] - times[k - 1]) * cpty.survival(times[k]) * bank.survival(times[k]);
        fca += borrow * ee_disc[k] * w;
        fba += lend * -ene_disc[k] * w;
    }
    Ok((fca, fba))
}

/// Basel effective EPE over min(1y, maturity).
pub fn eepe(times: &[f64], ee: &[f64], maturity: f64) -> Result<f64> {
    check_grid(times, ee)?;
    let h = maturity.min(1.0);
    if h.is_nan() || h <= 0.0 {
        return Err(bad("maturity must be positive"));
    }
    let (mut eee, mut acc) = (0.0_f64, 0.0);
    for k in 1..times.len() {
        eee = eee.max(ee[k]);
        if times[k] <= h + 1e-12 {
            acc += eee * (times[k] - times[k - 1]);
        }
    }
    Ok(acc / h)
}

/// Zero-based PFE order-statistic index ceil(q n) - 1.
pub fn pfe_index(n: usize, q: f64) -> Result<usize> {
    if n < 1 || !(q > 0.0 && q < 1.0) {
        return Err(bad("need n >= 1 and 0 < q < 1"));
    }
    Ok(((q * n as f64).ceil() as usize).saturating_sub(1))
}
