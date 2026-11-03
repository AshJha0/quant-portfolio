//! Bachelier (normal) model, a port of `eq_options.bachelier`.
//!
//! Schachermayer and Teichmann (2008), Math. Finance 18(1), 155-170. `F` and
//! `K` may be negative; `sigma_n` is in price units per sqrt(year); theta
//! is `-dV/dT` per year with `F` held.

use crate::black_scholes::{OptionType, PricingError, norm_cdf, norm_pdf};

/// Bachelier price and Greeks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BachelierGreeks {
    /// Present value.
    pub price: f64,
    /// dV/dF.
    pub delta: f64,
    /// d2V/dF2.
    pub gamma: f64,
    /// dV/dsigma_n.
    pub vega: f64,
    /// -dV/dT, per year.
    pub theta: f64,
}

fn check(f: f64, k: f64, t: f64, sigma_n: f64, r: f64) -> Result<(), PricingError> {
    if !(f.is_finite() && k.is_finite() && t.is_finite() && sigma_n.is_finite() && r.is_finite()) {
        return Err(PricingError::InvalidInput(
            "bachelier: inputs must be finite".into(),
        ));
    }
    if t < 0.0 || sigma_n < 0.0 {
        return Err(PricingError::InvalidInput(
            "bachelier: T and sigma_n must be >= 0".into(),
        ));
    }
    Ok(())
}

fn sign_of(option_type: OptionType) -> f64 {
    match option_type {
        OptionType::Call => 1.0,
        OptionType::Put => -1.0,
    }
}

/// Bachelier price.
pub fn bachelier_price(
    f: f64,
    k: f64,
    t: f64,
    sigma_n: f64,
    r: f64,
    option_type: OptionType,
) -> Result<f64, PricingError> {
    check(f, k, t, sigma_n, r)?;
    let df = (-r * t).exp();
    let sign = sign_of(option_type);
    let s = sigma_n * t.sqrt();
    if s == 0.0 {
        return Ok(df * (sign * (f - k)).max(0.0));
    }
    let d = (f - k) / s;
    Ok(df * (sign * (f - k) * norm_cdf(sign * d) + s * norm_pdf(d)))
}

/// Analytic Bachelier Greeks; needs `t > 0` and `sigma_n > 0`.
pub fn bachelier_greeks(
    f: f64,
    k: f64,
    t: f64,
    sigma_n: f64,
    r: f64,
    option_type: OptionType,
) -> Result<BachelierGreeks, PricingError> {
    check(f, k, t, sigma_n, r)?;
    if t <= 0.0 || sigma_n <= 0.0 {
        return Err(PricingError::InvalidInput(
            "bachelier_greeks: need T > 0 and sigma_n > 0".into(),
        ));
    }
    let df = (-r * t).exp();
    let sq = t.sqrt();
    let s = sigma_n * sq;
    let d = (f - k) / s;
    let pdf = norm_pdf(d);
    let price = bachelier_price(f, k, t, sigma_n, r, option_type)?;
    let delta = match option_type {
        OptionType::Call => df * norm_cdf(d),
        OptionType::Put => -df * norm_cdf(-d),
    };
    Ok(BachelierGreeks {
        price,
        delta,
        gamma: df * pdf / s,
        vega: df * sq * pdf,
        theta: r * price - df * sigma_n * pdf / (2.0 * sq),
    })
}

/// Implied normal vol: safeguarded Newton on the out-of-the-money side,
/// relative tolerance 1e-14.
pub fn bachelier_implied_vol(
    price: f64,
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    option_type: OptionType,
) -> Result<f64, PricingError> {
    check(f, k, t, 0.0, r)?;
    if !price.is_finite() || t <= 0.0 {
        return Err(PricingError::InvalidInput(
            "need a finite price and T > 0".into(),
        ));
    }
    let df = (-r * t).exp();
    let mut sign = sign_of(option_type);
    let mut ot = option_type;
    if price <= df * (sign * (f - k)).max(0.0) {
        return Err(PricingError::ArbitrageBound(
            "price at or below intrinsic".into(),
        ));
    }
    let mut price = price;
    if sign * (f - k) > 0.0 {
        price -= df * sign * (f - k);
        sign = -sign;
        ot = if sign > 0.0 {
            OptionType::Call
        } else {
            OptionType::Put
        };
    }
    let mut lo = 0.0;
    let mut hi = (f - k).abs().max(1.0) / t.sqrt();
    while bachelier_price(f, k, t, hi, r, ot)? < price {
        hi *= 2.0;
        if hi > 1e300 {
            return Err(PricingError::ArbitrageBound(
                "price above any finite normal vol".into(),
            ));
        }
    }
    let mut sig = (price / df * (2.0 * std::f64::consts::PI / t).sqrt()).min(hi);
    for _ in 0..200 {
        let diff = bachelier_price(f, k, t, sig, r, ot)? - price;
        if diff.abs() <= 4.0 * f64::EPSILON * price {
            return Ok(sig);
        }
        if diff > 0.0 {
            hi = sig;
        } else {
            lo = sig;
        }
        let d = if sig > 0.0 {
            (f - k) / (sig * t.sqrt())
        } else {
            f64::INFINITY
        };
        let vega = df * t.sqrt() * norm_pdf(d);
        let step = if vega > 0.0 {
            sig - diff / vega
        } else {
            f64::NAN
        };
        let next = if step >= lo && step <= hi {
            step
        } else {
            0.5 * (lo + hi)
        };
        if (next - sig).abs() <= 1e-14 * next {
            return Ok(next);
        }
        sig = next;
    }
    Ok(sig)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1e-3), "{a} vs {b}");
    }

    #[test]
    fn golden_vs_python() {
        let g = bachelier_greeks(1.0, 1.3, 2.0, 0.4, 0.03, OptionType::Call).unwrap();
        rel(g.price, 0.10047506715747369, 1e-13);
        rel(g.delta, 0.2805907804284276, 1e-13);
        rel(g.gamma, 0.5770384415187559, 1e-13);
        rel(g.vega, 0.46163075321500496, 1e-13);
        rel(g.theta, -0.04314882330677628, 1e-13);
        let g = bachelier_greeks(-0.5, 0.2, 1.0, 0.8, 0.01, OptionType::Put).unwrap();
        rel(g.price, 0.7762912746894666, 1e-13);
        rel(g.delta, -0.8011612427960295, 1e-13);
        rel(g.theta, -0.09997628961922823, 1e-13);
    }

    #[test]
    fn worked_example_and_implied_vol() {
        let atm = bachelier_price(100.0, 100.0, 1.0, 20.0, 0.0, OptionType::Call).unwrap();
        assert!((atm - 20.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-13);
        for k in [-1.0, 0.5, 1.0, 1.5, 4.0] {
            for ot in [OptionType::Call, OptionType::Put] {
                let p = bachelier_price(1.0, k, 1.5, 0.6, 0.02, ot).unwrap();
                let v = bachelier_implied_vol(p, 1.0, k, 1.5, 0.02, ot).unwrap();
                assert!((v - 0.6).abs() < 6e-11, "{k} {v}");
            }
        }
        assert!(bachelier_implied_vol(0.0, 1.0, 1.0, 1.0, 0.0, OptionType::Call).is_err());
    }
}
