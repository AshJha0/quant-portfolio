//! Shared implied-volatility core: normalised Black, solved on the OTM side.
//!
//! The same algorithm, constants and branch thresholds are implemented in
//! the Python (`_iv_core.py`), C++ and Rust engines. Inputs to
//! [`total_vol`] are the normalised OTM time value `beta`, its complement
//! `u = b_max - beta` and `x = -|ln(F/K)| <= 0`; the result is the total
//! volatility `s = sigma sqrt(T)`. One evaluation is one pass of the
//! iteration loop.
//!
//! The algorithm lives in [`qp_core::iv`]; this module pins it to the
//! erf table this engine has always used (`qp_core::iv::CodyC1Truncated`).

use qp_core::iv::CodyC1Truncated;

/// `b(x,s) = e^{x/2} N(x/s+s/2) - e^{-x/2} N(x/s-s/2)` for `x <= 0`, `s > 0`.
#[inline]
pub fn normalised_otm_call(x: f64, s: f64) -> f64 {
    qp_core::iv::normalised_otm_call::<CodyC1Truncated>(x, s)
}

/// `b_max - b = e^{x/2} N(-d1) + e^{-x/2} N(d2)`: a sum, no cancellation.
#[inline]
pub fn normalised_otm_call_complement(x: f64, s: f64) -> f64 {
    qp_core::iv::normalised_otm_call_complement::<CodyC1Truncated>(x, s)
}

/// Solve `b(x, s) = beta` for `s`. Requires `beta > 0`, `u > 0`, `x <= 0`.
/// `evals` is incremented once per iteration.
#[inline]
pub fn total_vol(beta: f64, u: f64, x: f64, evals: &mut u32) -> Result<f64, &'static str> {
    qp_core::iv::total_vol::<CodyC1Truncated>(beta, u, x, evals)
}

/// Black implied vol from a validated quote with `lower < price < upper`.
#[inline]
pub fn black_implied_vol(
    price: f64,
    f: f64,
    k: f64,
    t: f64,
    df: f64,
    is_call: bool,
    evals: &mut u32,
) -> Result<f64, &'static str> {
    qp_core::iv::black_implied_vol::<CodyC1Truncated>(price, f, k, t, df, is_call, evals)
}
