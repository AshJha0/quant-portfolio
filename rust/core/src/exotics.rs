//! Closed-form exotic pricers: the Rust twin of `qp_core.exotics` (Python).
//!
//! Plain idea: the same formulas, written a second time in a second
//! language, so a slip in one shows up as a parity failure. The golden
//! vectors in `tests/data/exotics_golden.txt` are generated from the
//! Python pricers and held to 1e-12 relative (floored at 1e-11 absolute on a 100 notional) (see `tests/exotics_parity.rs`).
//!
//! Sources, as in the Python modules:
//! - single barriers: Reiner and Rubinstein (1991), Haug (2007) s.4.17.1;
//! - touches: Reiner and Rubinstein (1991b), Haug (2007) s.4.19;
//! - double knock-out: Ikeda and Kunitomo (1992), Haug (2007) s.4.17.3;
//! - double no-touch: Hui (1996);
//! - geometric Asian (discrete): Kemna and Vorst (1990);
//! - Turnbull and Wakeman (1991), discrete moments and Haug's continuous form;
//! - eSSVI slice (Hendriks and Martini 2019) and raw SVI (Gatheral 2004).
//!
//! The Monte Carlo pricers (barrier bridge MC, LSM, TARF, LSV particles)
//! are not ported: their outputs depend on the random stream, and the
//! closed forms above are what they are checked against.

use crate::special::norm_cdf as n;

/// Call or put.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cp {
    /// Call.
    Call,
    /// Put.
    Put,
}

impl Cp {
    fn phi(self) -> f64 {
        match self {
            Cp::Call => 1.0,
            Cp::Put => -1.0,
        }
    }
}

/// Single-barrier type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Barrier {
    /// Down-and-in.
    DownIn,
    /// Down-and-out.
    DownOut,
    /// Up-and-in.
    UpIn,
    /// Up-and-out.
    UpOut,
}

/// Touch type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Touch {
    /// One-touch, lower barrier.
    OneTouchDown,
    /// One-touch, upper barrier.
    OneTouchUp,
    /// No-touch, lower barrier.
    NoTouchDown,
    /// No-touch, upper barrier.
    NoTouchUp,
}

/// Generalised Black-Scholes with cost of carry `b` (Haug 2007 s.1.1.6).
pub fn vanilla_price(cp: Cp, s: f64, x: f64, t: f64, r: f64, b: f64, sigma: f64) -> f64 {
    let phi = cp.phi();
    let v = sigma * t.sqrt();
    let d1 = ((s / x).ln() + (b + 0.5 * sigma * sigma) * t) / v;
    let d2 = d1 - v;
    phi * (s * ((b - r) * t).exp() * n(phi * d1) - x * (-r * t).exp() * n(phi * d2))
}

#[allow(clippy::too_many_arguments)]
fn rr_terms(
    phi: f64,
    eta: f64,
    s: f64,
    x: f64,
    h: f64,
    k: f64,
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
) -> [f64; 6] {
    let v = sigma * t.sqrt();
    let mu = (b - 0.5 * sigma * sigma) / (sigma * sigma);
    let lam = (mu * mu + 2.0 * r / (sigma * sigma)).sqrt();
    let x1 = (s / x).ln() / v + (1.0 + mu) * v;
    let x2 = (s / h).ln() / v + (1.0 + mu) * v;
    let y1 = (h * h / (s * x)).ln() / v + (1.0 + mu) * v;
    let y2 = (h / s).ln() / v + (1.0 + mu) * v;
    let z = (h / s).ln() / v + lam * v;
    let fs = s * ((b - r) * t).exp();
    let fx = x * (-r * t).exp();
    let hs = h / s;
    let a = phi * fs * n(phi * x1) - phi * fx * n(phi * x1 - phi * v);
    let bb = phi * fs * n(phi * x2) - phi * fx * n(phi * x2 - phi * v);
    let c = phi * fs * hs.powf(2.0 * (mu + 1.0)) * n(eta * y1)
        - phi * fx * hs.powf(2.0 * mu) * n(eta * y1 - eta * v);
    let d = phi * fs * hs.powf(2.0 * (mu + 1.0)) * n(eta * y2)
        - phi * fx * hs.powf(2.0 * mu) * n(eta * y2 - eta * v);
    let e =
        k * (-r * t).exp() * (n(eta * x2 - eta * v) - hs.powf(2.0 * mu) * n(eta * y2 - eta * v));
    let f =
        k * (hs.powf(mu + lam) * n(eta * z) + hs.powf(mu - lam) * n(eta * z - 2.0 * eta * lam * v));
    [a, bb, c, d, e, f]
}

/// Single-barrier option, continuous monitoring (Reiner-Rubinstein 1991).
#[allow(clippy::too_many_arguments)]
pub fn barrier_price(
    kind: Barrier,
    cp: Cp,
    s: f64,
    x: f64,
    h: f64,
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
    rebate: f64,
) -> f64 {
    let down = matches!(kind, Barrier::DownIn | Barrier::DownOut);
    let is_in = matches!(kind, Barrier::DownIn | Barrier::UpIn);
    let knocked = if down { s <= h } else { s >= h };
    if knocked {
        return if is_in {
            vanilla_price(cp, s, x, t, r, b, sigma)
        } else {
            rebate
        };
    }
    let eta = if down { 1.0 } else { -1.0 };
    let [a, bb, c, d, e, f] = rr_terms(cp.phi(), eta, s, x, h, rebate, t, r, b, sigma);
    let call = cp == Cp::Call;
    let (hi, lo) = match (kind, call) {
        (Barrier::DownIn, true) => (c + e, a - bb + d + e),
        (Barrier::UpIn, true) => (a + e, bb - c + d + e),
        (Barrier::DownIn, false) => (bb - c + d + e, a + e),
        (Barrier::UpIn, false) => (a - bb + d + e, c + e),
        (Barrier::DownOut, true) => (a - c + f, bb - d + f),
        (Barrier::UpOut, true) => (f, a - bb + c - d + f),
        (Barrier::DownOut, false) => (a - bb + c - d + f, f),
        (Barrier::UpOut, false) => (bb - d + f, a - c + f),
    };
    if x > h { hi } else { lo }
}

/// One-touch / no-touch paying `cash` (at expiry, or at the hit for one-touches with `pay_at_hit`).
#[allow(clippy::too_many_arguments)]
pub fn touch_price(
    kind: Touch,
    s: f64,
    h: f64,
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
    cash: f64,
    pay_at_hit: bool,
) -> f64 {
    let down = matches!(kind, Touch::OneTouchDown | Touch::NoTouchDown);
    let one = matches!(kind, Touch::OneTouchDown | Touch::OneTouchUp);
    let knocked = if down { s <= h } else { s >= h };
    let df = (-r * t).exp();
    if knocked {
        return if one {
            if pay_at_hit { cash } else { cash * df }
        } else {
            0.0
        };
    }
    let eta = if down { 1.0 } else { -1.0 };
    let v = sigma * t.sqrt();
    let mu = (b - 0.5 * sigma * sigma) / (sigma * sigma);
    let hs = h / s;
    let x2 = (s / h).ln() / v + (1.0 + mu) * v;
    let y2 = (h / s).ln() / v + (1.0 + mu) * v;
    let p_no = n(eta * x2 - eta * v) - hs.powf(2.0 * mu) * n(eta * y2 - eta * v);
    if !one {
        return cash * df * p_no;
    }
    if !pay_at_hit {
        return cash * df * (1.0 - p_no);
    }
    let lam = (mu * mu + 2.0 * r / (sigma * sigma)).sqrt();
    let z = (h / s).ln() / v + lam * v;
    cash * (hs.powf(mu + lam) * n(eta * z) + hs.powf(mu - lam) * n(eta * z - 2.0 * eta * lam * v))
}

/// Double knock-out with flat barriers (Ikeda-Kunitomo 1992), series over `-n_terms..=n_terms`.
#[allow(clippy::too_many_arguments)]
pub fn double_knockout_price(
    cp: Cp,
    s: f64,
    x: f64,
    l: f64,
    u: f64,
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
    n_terms: i32,
) -> f64 {
    if s <= l || s >= u {
        return 0.0;
    }
    let v = sigma * t.sqrt();
    let s2 = sigma * sigma;
    let drift = (b + 0.5 * s2) * t;
    let mu1 = 2.0 * b / s2 + 1.0;
    let mu3 = mu1;
    let (e, f) = (l, u);
    let call = cp == Cp::Call;
    let mut s1 = 0.0;
    let mut s2_ = 0.0;
    for k in -n_terms..=n_terms {
        let u2n = u.powi(2 * k);
        let l2n = l.powi(2 * k);
        let a1 = (u.powi(k) / l.powi(k)).powf(mu1);
        let a1m = (u.powi(k) / l.powi(k)).powf(mu1 - 2.0);
        let r3 = l.powi(k + 1) / (u.powi(k) * s);
        let a3 = r3.powf(mu3);
        let a3m = r3.powf(mu3 - 2.0);
        let l22 = l.powi(2 * k + 2);
        let (d1, d2, d3, d4) = if call {
            (
                ((s * u2n / (x * l2n)).ln() + drift) / v,
                ((s * u2n / (f * l2n)).ln() + drift) / v,
                ((l22 / (x * s * u2n)).ln() + drift) / v,
                ((l22 / (f * s * u2n)).ln() + drift) / v,
            )
        } else {
            (
                ((s * u2n / (e * l2n)).ln() + drift) / v,
                ((s * u2n / (x * l2n)).ln() + drift) / v,
                ((l22 / (e * s * u2n)).ln() + drift) / v,
                ((l22 / (x * s * u2n)).ln() + drift) / v,
            )
        };
        s1 += a1 * (n(d1) - n(d2)) - a3 * (n(d3) - n(d4));
        s2_ += a1m * (n(d1 - v) - n(d2 - v)) - a3m * (n(d3 - v) - n(d4 - v));
    }
    let fs = s * ((b - r) * t).exp();
    let fx = x * (-r * t).exp();
    if call {
        if x >= u {
            return 0.0;
        }
        fs * s1 - fx * s2_
    } else {
        if x <= l {
            return 0.0;
        }
        fx * s2_ - fs * s1
    }
}

/// Double no-touch paying `cash` at expiry (Hui 1996), `n_terms` sine terms.
#[allow(clippy::too_many_arguments)]
pub fn double_no_touch(
    s: f64,
    l: f64,
    u: f64,
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
    cash: f64,
    n_terms: u32,
) -> f64 {
    if s <= l || s >= u {
        return 0.0;
    }
    let s2 = sigma * sigma;
    let z = (u / l).ln();
    let alpha = -0.5 * (2.0 * b / s2 - 1.0);
    let beta = -0.25 * (2.0 * b / s2 - 1.0).powi(2) - 2.0 * r / s2;
    let sl = (s / l).powf(alpha);
    let su = (s / u).powf(alpha);
    let mut total = 0.0;
    for i in 1..=n_terms {
        let fi = f64::from(i);
        let ip = fi * std::f64::consts::PI / z;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        total += 2.0 * std::f64::consts::PI * fi / (z * z) * (sl - sign * su)
            / (alpha * alpha + ip * ip)
            * (ip * (s / l).ln()).sin()
            * (-0.5 * (ip * ip - beta) * s2 * t).exp();
    }
    cash * total
}

fn black(cp: Cp, f: f64, x: f64, v: f64, df: f64) -> f64 {
    let d1 = ((f / x).ln() + 0.5 * v * v) / v;
    let d2 = d1 - v;
    match cp {
        Cp::Call => df * (f * n(d1) - x * n(d2)),
        Cp::Put => df * (x * n(-d2) - f * n(-d1)),
    }
}

/// Discretely monitored geometric-average Asian (exact).
#[allow(clippy::too_many_arguments)]
pub fn geometric_asian_price(
    cp: Cp,
    s: f64,
    x: f64,
    fixings: &[f64],
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
) -> f64 {
    let nf = fixings.len() as f64;
    let tmean = fixings.iter().sum::<f64>() / nf;
    let mut mins = 0.0;
    for &ti in fixings {
        for &tj in fixings {
            mins += ti.min(tj);
        }
    }
    let mean = s.ln() + (b - 0.5 * sigma * sigma) * tmean;
    let var = sigma * sigma * mins / (nf * nf);
    black(cp, (mean + 0.5 * var).exp(), x, var.sqrt(), (-r * t).exp())
}

/// Turnbull-Wakeman with exact discrete moments.
#[allow(clippy::too_many_arguments)]
pub fn turnbull_wakeman_price(
    cp: Cp,
    s: f64,
    x: f64,
    fixings: &[f64],
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
) -> f64 {
    let nf = fixings.len() as f64;
    let e: Vec<f64> = fixings.iter().map(|&ti| (b * ti).exp()).collect();
    let m1 = s * e.iter().sum::<f64>() / nf;
    let mut acc = 0.0;
    for (i, &ti) in fixings.iter().enumerate() {
        for (j, &tj) in fixings.iter().enumerate() {
            acc += e[i] * e[j] * (sigma * sigma * ti.min(tj)).exp();
        }
    }
    let m2 = s * s * acc / (nf * nf);
    let v = (m2 / (m1 * m1)).ln().sqrt();
    black(cp, m1, x, v, (-r * t).exp())
}

/// Continuous-averaging Turnbull-Wakeman over `[tau, T]` (Haug 2007 s.4.20.2).
#[allow(clippy::too_many_arguments)]
pub fn turnbull_wakeman_continuous(
    cp: Cp,
    s: f64,
    x: f64,
    t: f64,
    r: f64,
    b: f64,
    sigma: f64,
    tau: f64,
) -> f64 {
    let s2 = sigma * sigma;
    let l = t - tau;
    let (m1, m2) = if b.abs() < 1e-8 {
        (
            1.0,
            2.0 * (s2 * t).exp() / (s2 * s2 * l * l)
                - 2.0 * (s2 * tau).exp() * (1.0 + s2 * l) / (s2 * s2 * l * l),
        )
    } else {
        (
            ((b * t).exp() - (b * tau).exp()) / (b * l),
            2.0 * ((2.0 * b + s2) * t).exp() / ((b + s2) * (2.0 * b + s2) * l * l)
                + 2.0 * ((2.0 * b + s2) * tau).exp() / (b * l * l)
                    * (1.0 / (2.0 * b + s2) - (b * l).exp() / (b + s2)),
        )
    };
    let v = (m2 / (m1 * m1)).ln().sqrt();
    black(cp, s * m1, x, v, (-r * t).exp())
}

/// eSSVI slice total variance at log-moneyness `k`.
pub fn essvi_total_variance(k: f64, theta: f64, rho: f64, psi: f64) -> f64 {
    let pk = psi * k / theta;
    0.5 * theta * (1.0 + rho * pk + ((pk + rho) * (pk + rho) + 1.0 - rho * rho).sqrt())
}

/// Raw SVI total variance (Gatheral 2004).
pub fn svi_total_variance(k: f64, a: f64, b: f64, rho: f64, m: f64, s: f64) -> f64 {
    let x = k - m;
    a + b * (rho * x + (x * x + s * s).sqrt())
}
