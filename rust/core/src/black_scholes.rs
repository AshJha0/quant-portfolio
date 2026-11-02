//! Black-Scholes in cost-of-carry form, shared by the equity
//! (Black-Scholes-Merton) and FX (Garman-Kohlhagen) options engines.
//!
//! # The plain idea
//!
//! A stock paying a continuous dividend yield `q` and a currency earning
//! the foreign rate `r_f` are the same underlying to a pricer: both grow at
//! the cost of carry `b` (`b = r - q` for a stock, `b = r_d - r_f` for a
//! currency) and the payoff is discounted at `r` (`r_d`). Garman and
//! Kohlhagen (1983) made exactly this observation, so one kernel prices
//! both, and the engines keep only their own names, validation and
//! boundary rules.
//!
//! # The formal statement
//!
//! With `phi = +1` for a call, `-1` for a put and the carry yield
//! `q = r - b`:
//!
//! ```text
//! d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),  d2 = d1 - sigma sqrt T
//! V  = phi [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)]
//! delta = phi e^{-qT} N(phi d1)             delta_forward = phi N(phi d1)
//! gamma = e^{-qT} n(d1) / (S sigma sqrt T)  vega  = S e^{-qT} n(d1) sqrt T
//! theta = -S e^{-qT} n(d1) sigma / (2 sqrt T) + phi [q S e^{-qT} N(phi d1) - r K e^{-rT} N(phi d2)]
//! rho   = phi K T e^{-rT} N(phi d2)         rho_q = -phi S T e^{-qT} N(phi d1)
//! vanna = -e^{-qT} n(d1) d2 / sigma         volga = vega d1 d2 / sigma
//! ```
//!
//! The functions take `r` and `q` rather than `b`, so both discount factors
//! come from the caller's own numbers (recovering `q` as `r - b` would round
//! and move the engines' golden values in the last bit); `d1` uses
//! `r - q = b`.
//!
//! # Worked example
//!
//! `S = K = 100`, `T = 1`, `r = 5 %`, `q = 2 %`, `sigma = 20 %`:
//! `d1 = (0 + 0.03 + 0.02) / 0.2 = 0.25`, `d2 = 0.05`, and the call is
//! `100 e^{-0.02} N(0.25) - 100 e^{-0.05} N(0.05) = 9.2270` to four
//! decimals; the same number prices a currency call with `r_d = 5 %`,
//! `r_f = 2 %`.
//!
//! ```
//! use qp_core::black_scholes::{carry_d1_d2, carry_price, ErfcNormal};
//! let (d1, d2) = carry_d1_d2(100.0, 100.0, 1.0, 0.05, 0.02, 0.2);
//! assert!((d1 - 0.25).abs() < 1e-15 && (d2 - 0.05).abs() < 1e-15);
//! let c = carry_price::<ErfcNormal>(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, 1.0);
//! assert!((c - 9.2270).abs() < 5e-5);
//! ```
//!
//! # Floating point
//!
//! Every function is unchecked (`S, K, T, sigma > 0`, finite rates);
//! validation and the boundary branches stay in the engines, whose domains
//! differ. The expressions are evaluated in a fixed order and both engines
//! return the values they returned before sharing this module, bit for bit.
//! The engines have always used different normal functions (the equity
//! engine Cody's erfc with a divide for the density, the FX engine the
//! truncated `C[1]` erfc with a multiply), and the two differ in the last
//! bit, so the normal distribution is a type parameter ([`NormalDist`]).

use crate::special;

/// The standard normal cdf and density an engine prices with.
pub trait NormalDist {
    /// `N(x)`.
    fn cdf(x: f64) -> f64;
    /// `n(x)`.
    fn pdf(x: f64) -> f64;
}

/// [`special::norm_cdf`] and [`special::norm_pdf_div`]: the equity engine's pair.
#[derive(Debug, Clone, Copy)]
pub struct ErfcNormal;

impl NormalDist for ErfcNormal {
    #[inline(always)]
    fn cdf(x: f64) -> f64 {
        special::norm_cdf(x)
    }
    #[inline(always)]
    fn pdf(x: f64) -> f64 {
        special::norm_pdf_div(x)
    }
}

/// [`special::norm_cdf_c1_truncated`] and [`special::norm_pdf`]: the FX
/// engine's pair.
#[derive(Debug, Clone, Copy)]
pub struct TruncatedC1Normal;

impl NormalDist for TruncatedC1Normal {
    #[inline(always)]
    fn cdf(x: f64) -> f64 {
        special::norm_cdf_c1_truncated(x)
    }
    #[inline(always)]
    fn pdf(x: f64) -> f64 {
        special::norm_pdf(x)
    }
}

/// `(d1, d2)` of the carry form.
#[inline(always)]
#[must_use]
pub fn carry_d1_d2(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> (f64, f64) {
    let vol_sqrt_t = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    (d1, d1 - vol_sqrt_t)
}

/// `phi [disc_s N(phi d1) - disc_k N(phi d2)]` from precomputed terms.
#[inline(always)]
#[must_use]
pub fn carry_value<N: NormalDist>(disc_s: f64, disc_k: f64, d1: f64, d2: f64, phi: f64) -> f64 {
    phi * (disc_s * N::cdf(phi * d1) - disc_k * N::cdf(phi * d2))
}

/// Price for `S, K, T, sigma > 0`; `phi` is +1 (call) or -1 (put). The
/// result is non-finite if a discount factor overflows; the caller checks.
#[inline(always)]
#[must_use]
pub fn carry_price<N: NormalDist>(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    phi: f64,
) -> f64 {
    let disc_s = s * (-q * t).exp();
    let disc_k = k * (-r * t).exp();
    let (d1, d2) = carry_d1_d2(s, k, t, r, q, sigma);
    carry_value::<N>(disc_s, disc_k, d1, d2, phi)
}

/// Price, analytic Greeks and the terms they are built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarryGreeks {
    /// Present value.
    pub price: f64,
    /// `dV/dS`.
    pub delta: f64,
    /// `phi N(phi d1)`.
    pub delta_forward: f64,
    /// `d2V/dS2`.
    pub gamma: f64,
    /// `dV/dsigma`.
    pub vega: f64,
    /// `dV/dt` per year, phi form (module docs).
    pub theta: f64,
    /// `dV/dr`.
    pub rho: f64,
    /// `dV/dq` (the FX engine's `rho_foreign`).
    pub rho_q: f64,
    /// `d2V/(dS dsigma)`.
    pub vanna: f64,
    /// `d2V/dsigma2`.
    pub volga: f64,
    /// `d1`.
    pub d1: f64,
    /// `d2`.
    pub d2: f64,
    /// `e^{-qT}`.
    pub df_q: f64,
    /// `e^{-rT}`.
    pub df_r: f64,
    /// `N(phi d1)`.
    pub n1: f64,
    /// `N(phi d2)`.
    pub n2: f64,
    /// `-S e^{-qT} n(d1) sigma / (2 sqrt T)`.
    pub common_theta: f64,
}

/// Analytic Greeks for `S, K, T, sigma > 0`.
#[inline(always)]
#[must_use]
pub fn carry_greeks<N: NormalDist>(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    phi: f64,
) -> CarryGreeks {
    let (d1, d2) = carry_d1_d2(s, k, t, r, q, sigma);
    carry_greeks_with_d::<N>(s, k, t, r, q, sigma, phi, d1, d2)
}

/// [`carry_greeks`] with `d1` and `d2` already computed by
/// [`carry_d1_d2`] (an engine that validated through its own `d1` keeps the
/// one evaluation of `ln(S/K)`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
#[must_use]
pub fn carry_greeks_with_d<N: NormalDist>(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    phi: f64,
    d1: f64,
    d2: f64,
) -> CarryGreeks {
    let sqrt_t = t.sqrt();
    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();
    let pdf = N::pdf(d1);
    let n1 = N::cdf(phi * d1);
    let n2 = N::cdf(phi * d2);
    let vega = s * df_q * pdf * sqrt_t;
    let common_theta = -s * df_q * pdf * sigma / (2.0 * sqrt_t);
    CarryGreeks {
        price: phi * (s * df_q * n1 - k * df_r * n2),
        delta: phi * df_q * n1,
        delta_forward: phi * n1,
        gamma: df_q * pdf / (s * sigma * sqrt_t),
        vega,
        theta: common_theta + phi * (q * s * df_q * n1 - r * k * df_r * n2),
        rho: phi * k * t * df_r * n2,
        rho_q: -phi * s * t * df_q * n1,
        vanna: -df_q * pdf * d2 / sigma,
        volga: vega * d1 * d2 / sigma,
        d1,
        d2,
        df_q,
        df_r,
        n1,
        n2,
        common_theta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_call_parity() {
        for q in [-0.01, 0.0, 0.03] {
            let (s, k, t, r, v) = (103.0, 97.0, 0.7, 0.04, 0.3);
            let c = carry_price::<ErfcNormal>(s, k, t, r, q, v, 1.0);
            let p = carry_price::<ErfcNormal>(s, k, t, r, q, v, -1.0);
            let fwd = s * (-q * t).exp() - k * (-r * t).exp();
            assert!((c - p - fwd).abs() < 1e-12);
        }
    }

    #[test]
    fn greeks_match_finite_differences() {
        let (s, k, t, r, q, v, h) = (100.0, 95.0, 0.5, 0.03, 0.01, 0.25, 1e-4);
        for phi in [1.0, -1.0] {
            let p = |s_: f64, t_: f64, r_: f64, q_: f64, v_: f64| {
                carry_price::<TruncatedC1Normal>(s_, k, t_, r_, q_, v_, phi)
            };
            let g = carry_greeks::<TruncatedC1Normal>(s, k, t, r, q, v, phi);
            assert!((g.price - p(s, t, r, q, v)).abs() < 1e-13);
            assert!(
                (g.delta - (p(s + h, t, r, q, v) - p(s - h, t, r, q, v)) / (2.0 * h)).abs() < 1e-7
            );
            assert!(
                (g.vega - (p(s, t, r, q, v + h) - p(s, t, r, q, v - h)) / (2.0 * h)).abs() < 1e-6
            );
            assert!(
                (g.theta + (p(s, t + h, r, q, v) - p(s, t - h, r, q, v)) / (2.0 * h)).abs() < 1e-6
            );
            assert!(
                (g.rho - (p(s, t, r + h, q, v) - p(s, t, r - h, q, v)) / (2.0 * h)).abs() < 1e-6
            );
            assert!(
                (g.rho_q - (p(s, t, r, q + h, v) - p(s, t, r, q - h, v)) / (2.0 * h)).abs() < 1e-6
            );
        }
    }

    #[test]
    fn normal_pairs_differ_at_most_in_the_last_bits() {
        let a = carry_price::<ErfcNormal>(100.0, 90.0, 0.3, 0.02, 0.01, 0.4, -1.0);
        let b = carry_price::<TruncatedC1Normal>(100.0, 90.0, 0.3, 0.02, 0.01, 0.4, -1.0);
        assert!((a - b).abs() < 1e-13);
    }
}
