//! Charm, speed, colour, zomma and veta, a port of `eq_options.greeks_higher`.
//!
//! Haug (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed.,
//! sections 2.2-2.4. Time derivatives are `d/dt = -d/dT` per year,
//! matching theta.

use crate::black_scholes::{OptionType, PricingError, d1_d2, norm_cdf, norm_pdf};

/// Higher-order Black-Scholes-Merton Greeks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HigherGreeks {
    /// dDelta/dt, per year.
    pub charm: f64,
    /// dGamma/dS.
    pub speed: f64,
    /// dGamma/dt, per year.
    pub colour: f64,
    /// dGamma/dsigma.
    pub zomma: f64,
    /// dVega/dt, per year.
    pub veta: f64,
}

/// Analytic higher-order Greeks; `s, k, t, sigma` strictly positive.
pub fn higher_greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> Result<HigherGreeks, PricingError> {
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return Err(PricingError::InvalidInput(
            "higher_greeks: S, K, T and sigma must be strictly positive".into(),
        ));
    }
    let (d1, d2) = d1_d2(s, k, t, r, sigma, q)?;
    let sqrt_t = t.sqrt();
    let st = sigma * sqrt_t;
    let dq = (-q * t).exp();
    let phi = norm_pdf(d1);
    let gamma = dq * phi / (s * st);
    let drift = (2.0 * (r - q) * t - d2 * st) / (2.0 * t * st);
    let charm = match option_type {
        OptionType::Call => q * dq * norm_cdf(d1) - dq * phi * drift,
        OptionType::Put => -q * dq * norm_cdf(-d1) - dq * phi * drift,
    };
    Ok(HigherGreeks {
        charm,
        speed: -gamma / s * (d1 / st + 1.0),
        colour: dq * phi / (2.0 * s * t * st)
            * (2.0 * q * t + 1.0 + (2.0 * (r - q) * t - d2 * st) * d1 / st),
        zomma: gamma * (d1 * d2 - 1.0) / sigma,
        veta: s * dq * phi * sqrt_t * (q + (r - q) * d1 / st - (1.0 + d1 * d2) / (2.0 * t)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::greeks::bs_greeks;

    fn rel(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1e-3), "{a} vs {b}");
    }

    #[test]
    fn golden_vs_python() {
        let g = higher_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
        rel(g.charm, -0.06566706071046415, 1e-12);
        rel(g.speed, -0.0005159554770107896, 1e-12);
        rel(g.colour, 0.01053018223535657, 1e-12);
        rel(g.zomma, -0.08888505717594967, 1e-12);
        rel(g.veta, -16.46367022098065, 1e-12);
        let g = higher_greeks(100.0, 110.0, 0.5, 0.03, 0.3, 0.02, OptionType::Put).unwrap();
        rel(g.charm, -0.22965667914717408, 1e-12);
        rel(g.speed, 8.967850765599677e-05, 1e-12);
        rel(g.colour, 0.01477156521455538, 1e-12);
        rel(g.zomma, -0.048947752194907886, 1e-12);
        rel(g.veta, -30.91807129987024, 1e-12);
    }

    #[test]
    fn speed_matches_fd_of_gamma() {
        let h = 1e-3;
        let ot = OptionType::Call;
        let gp = bs_greeks(100.0 + h, 95.0, 0.7, 0.02, 0.25, 0.01, ot).unwrap();
        let gm = bs_greeks(100.0 - h, 95.0, 0.7, 0.02, 0.25, 0.01, ot).unwrap();
        let hg = higher_greeks(100.0, 95.0, 0.7, 0.02, 0.25, 0.01, ot).unwrap();
        assert!((hg.speed - (gp.gamma - gm.gamma) / (2.0 * h)).abs() < 1e-8);
    }
}
