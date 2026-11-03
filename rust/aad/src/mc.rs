//! Monte Carlo pricer with pathwise adjoint Greeks; the port of
//! `qp_core.aad.monte_carlo`.
//!
//! `ln S_{i+1} = ln S_i + (r - q - sigma^2/2) dt + sigma sqrt(dt) Z_i`; price
//! `exp(-rT) mean(payoff)`. European and arithmetic-Asian calls are Lipschitz,
//! so the pathwise derivative is unbiased (Broadie and Glasserman 1996;
//! Giles and Glasserman 2006, "Smoking adjoints"). Each path is recorded,
//! swept and dropped with [`Tape::mark`] and [`Tape::rewind`].

use crate::{Real, Tape, Var};
use qp_core::rng::Xoshiro256PlusPlus;

/// The payoff on the simulated path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Payoff {
    /// `max(S_T - K, 0)`.
    European,
    /// `max(mean(S_1..S_n) - K, 0)`.
    Asian,
}

/// Row-major `(paths, steps)` standard normals from xoshiro256++ (qp_core).
pub fn normals(paths: usize, steps: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xoshiro256PlusPlus::new(seed);
    (0..paths * steps).map(|_| rng.standard_normal()).collect()
}

fn path_payoff<T: Real>(log_s0: T, drift: T, vol: T, k: T, z: &[f64], payoff: Payoff) -> Option<T> {
    let under = match payoff {
        // ln S_T depends on the inputs only through sum(z): one exp, four
        // tape nodes per path whatever the number of steps.
        Payoff::European => {
            let zsum: f64 = z.iter().sum();
            (log_s0 + drift * z.len() as f64 + vol * zsum).exp()
        }
        Payoff::Asian => {
            let mut x = log_s0;
            let mut total: Option<T> = None;
            for &zi in z {
                x = x + drift + vol * zi;
                let s = x.exp();
                total = Some(total.map_or(s, |a| a + s));
            }
            total? / z.len() as f64
        }
    };
    if under.val() > k.val() {
        Some(under - k)
    } else {
        None
    }
}

/// Plain-float price on `z` (`paths * steps` normals, row-major).
#[allow(clippy::too_many_arguments)]
pub fn price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    z: &[f64],
    steps: usize,
    payoff: Payoff,
) -> f64 {
    let paths = z.len() / steps;
    let dt = t / steps as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let log_s0 = s.ln();
    let mut acc = 0.0;
    for p in 0..paths {
        if let Some(v) = path_payoff(
            log_s0,
            drift,
            vol,
            k,
            &z[p * steps..(p + 1) * steps],
            payoff,
        ) {
            acc += v;
        }
    }
    (-r * t).exp() * acc / paths as f64
}

/// Price, pathwise adjoint Greeks and the standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McGreeks {
    /// Price.
    pub price: f64,
    /// dV/dS.
    pub delta: f64,
    /// dV/dsigma.
    pub vega: f64,
    /// dV/dr.
    pub rho: f64,
    /// dV/dq.
    pub rho_q: f64,
    /// dV/dK.
    pub d_k: f64,
    /// dV/dT.
    pub d_t: f64,
    /// Standard error of the price.
    pub std_error: f64,
}

/// One backward sweep per path; the tape never holds more than one path.
#[allow(clippy::too_many_arguments)]
pub fn greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    z: &[f64],
    steps: usize,
    payoff: Payoff,
) -> McGreeks {
    let paths = z.len() / steps;
    let tape = Tape::new();
    let x: Vec<Var<'_>> = [s, k, t, r, q, sigma]
        .iter()
        .map(|&v| tape.var(v))
        .collect();
    let (vs, vk, vt, vr, vq, vsig) = (x[0], x[1], x[2], x[3], x[4], x[5]);
    let dt = vt / steps as f64;
    let drift = (vr - vq - vsig * vsig * 0.5) * dt;
    let vol = vsig * dt.sqrt();
    let log_s0 = vs.ln();
    let disc = (-(vr * vt)).exp();
    let mark = tape.mark();
    tape.zero_adjoints();
    let (mut total, mut sq) = (0.0, 0.0);
    let dv = disc.value();
    let n = paths as f64;
    for p in 0..paths {
        if let Some(pay) = path_payoff(
            log_s0,
            drift,
            vol,
            vk,
            &z[p * steps..(p + 1) * steps],
            payoff,
        ) {
            let pv = dv * pay.value();
            total += pv;
            sq += pv * pv;
            tape.backward(pay, dv / n, mark);
        }
        tape.rewind(mark);
    }
    // d price / d disc = mean payoff.
    tape.backward(disc, total / dv / n, mark);
    tape.propagate(mark, 0);
    let g: Vec<f64> = x.iter().map(|&v| tape.adjoint(v)).collect();
    let price = total / n;
    let var = (sq / n - price * price).max(0.0);
    McGreeks {
        price,
        delta: g[0],
        vega: g[5],
        rho: g[3],
        rho_q: g[4],
        d_k: g[1],
        d_t: g[2],
        std_error: (var / n).sqrt(),
    }
}
