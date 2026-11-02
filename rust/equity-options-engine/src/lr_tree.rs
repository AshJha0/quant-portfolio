//! Leisen-Reimer binomial tree, a port of `eq_options.lr_tree`.
//!
//! Leisen and Reimer (1996), Applied Mathematical Finance 3(4), 319-346,
//! Peizer-Pratt method 2 inversion, eq. (6). `n` is forced odd; the
//! European error falls like `1/n^2`.

use crate::binomial::Exercise;
use crate::black_scholes::{d1_d2, validate_inputs, validate_rates, OptionType, PricingError};

/// Peizer-Pratt method 2: binomial probability matching `N(z)` for odd `n`.
pub fn peizer_pratt(z: f64, n: usize) -> Result<f64, PricingError> {
    if n == 0 || n.is_multiple_of(2) {
        return Err(PricingError::InvalidInput(
            "peizer_pratt: n must be a positive odd integer".into(),
        ));
    }
    let nd = n as f64;
    let a = z / (nd + 1.0 / 3.0 + 0.1 / (nd + 1.0));
    let root = (-(-a * a * (nd + 1.0 / 6.0)).exp_m1()).sqrt();
    Ok(0.5 + 0.5f64.copysign(z) * root)
}

/// Leisen-Reimer price; `s, k, t, sigma` strictly positive, even `n_steps`
/// raised by one.
#[allow(clippy::too_many_arguments)]
pub fn lr_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
    exercise: Exercise,
    n_steps: usize,
) -> Result<f64, PricingError> {
    validate_inputs(s, k, t, sigma)?;
    validate_rates(r, q)?;
    if n_steps == 0 {
        return Err(PricingError::InvalidInput(
            "lr_price: n_steps must be >= 1".into(),
        ));
    }
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return Err(PricingError::InvalidInput(
            "lr_price: S, K, T and sigma must be strictly positive".into(),
        ));
    }
    let n = if !n_steps.is_multiple_of(2) {
        n_steps
    } else {
        n_steps + 1
    };
    let dt = t / n as f64;
    let (d1, d2) = d1_d2(s, k, t, r, sigma, q)?;
    let p_prime = peizer_pratt(d1, n)?;
    let p = peizer_pratt(d2, n)?;
    let growth = ((r - q) * dt).exp();
    let u = growth * p_prime / p;
    let d = (growth - p * u) / (1.0 - p);
    if !(p > 0.0 && p < 1.0 && d < u && d > 0.0) {
        return Err(PricingError::InvalidInput(
            "lr_price: Leisen-Reimer tree invalid for these inputs".into(),
        ));
    }
    let disc = (-r * t / n as f64).exp();
    let (pu, pd) = (disc * p, disc * (1.0 - p));
    let sign = match option_type {
        OptionType::Call => 1.0,
        OptionType::Put => -1.0,
    };
    let american = matches!(exercise, Exercise::American);
    let (log_u, log_d) = (u.ln(), d.ln());
    let mut v: Vec<f64> = (0..=n)
        .map(|j| (sign * (s * (j as f64 * log_u + (n - j) as f64 * log_d).exp() - k)).max(0.0))
        .collect();
    for i in (0..n).rev() {
        for kk in 0..=i {
            let mut cont = pu * v[kk + 1] + pd * v[kk];
            if american {
                let spot = s * (kk as f64 * log_u + (i - kk) as f64 * log_d).exp();
                cont = cont.max(sign * (spot - k));
            }
            v[kk] = cont;
        }
    }
    Ok(v[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes::bs_price;

    fn rel(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1e-3), "{a} vs {b}");
    }

    #[test]
    fn golden_vs_python() {
        let c = OptionType::Call;
        let p = OptionType::Put;
        let (eu, am) = (Exercise::European, Exercise::American);
        rel(
            lr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, c, eu, 101).unwrap(),
            10.450549336567306,
            1e-12,
        );
        rel(
            lr_price(100.0, 110.0, 0.5, 0.03, 0.3, 0.02, p, eu, 101).unwrap(),
            14.21511061827863,
            1e-12,
        );
        rel(
            lr_price(100.0, 110.0, 0.5, 0.03, 0.3, 0.02, p, am, 101).unwrap(),
            14.33188433420645,
            1e-12,
        );
        rel(
            lr_price(50.0, 60.0, 0.25, -0.01, 0.4, 0.0, c, am, 101).unwrap(),
            1.051752922776993,
            1e-12,
        );
    }

    #[test]
    fn second_order_convergence() {
        let c = OptionType::Call;
        let bs = bs_price(100.0, 105.0, 1.0, 0.04, 0.25, 0.01, c).unwrap();
        let e = |n| {
            (lr_price(
                100.0,
                105.0,
                1.0,
                0.04,
                0.25,
                0.01,
                c,
                Exercise::European,
                n,
            )
            .unwrap()
                - bs)
                .abs()
        };
        let ratio = e(101) / e(201);
        assert!(ratio > 3.0 && ratio < 5.0, "{ratio}");
    }

    #[test]
    fn rejects() {
        assert!(peizer_pratt(0.1, 10).is_err());
        assert!(lr_price(
            0.0,
            100.0,
            1.0,
            0.05,
            0.2,
            0.0,
            OptionType::Call,
            Exercise::European,
            11
        )
        .is_err());
    }
}
