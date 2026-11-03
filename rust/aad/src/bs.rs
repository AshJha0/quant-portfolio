//! Black-Scholes price with adjoint Greeks; the port of `qp_core.aad.black_scholes`.
//!
//! `d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt T)`, `d2 = d1 - sigma sqrt T`,
//! price `phi (S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2))` (Hull 2022, ch. 19).

use crate::{Real, Tape};

/// Generic price; `phi` is +1 (call) or -1 (put).
pub fn price<T: Real>(s: T, k: T, t: T, r: T, q: T, sigma: T, phi: f64) -> T {
    let sd = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + sigma * sigma * 0.5) * t) / sd;
    let d2 = d1 - sd;
    (s * (-(q * t)).exp() * (d1 * phi).ncdf() - k * (-(r * t)).exp() * (d2 * phi).ncdf()) * phi
}

/// Price and first-order Greeks from one backward sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
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
    /// dV/dT (theta is minus this).
    pub d_t: f64,
}

/// Adjoint Greeks of [`price`].
pub fn greeks(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64, is_call: bool) -> Greeks {
    let tape = Tape::with_capacity(64);
    let x: Vec<_> = [s, k, t, r, q, sigma]
        .iter()
        .map(|&v| tape.var(v))
        .collect();
    let v = price(
        x[0],
        x[1],
        x[2],
        x[3],
        x[4],
        x[5],
        if is_call { 1.0 } else { -1.0 },
    );
    let g = tape.gradient(v, &x);
    Greeks {
        price: v.value(),
        delta: g[0],
        vega: g[5],
        rho: g[3],
        rho_q: g[4],
        d_k: g[1],
        d_t: g[2],
    }
}
