//! Monte Carlo pricing of European options under GBM (exact scheme).
//!
//! The terminal stock price is simulated exactly:
//! `S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)`, `Z ~ N(0, 1)`,
//! so there is *no* time-discretisation bias — only statistical error,
//! which is reported as a standard error and 95% confidence interval.
//!
//! # Variance reduction (semantics mirror the Python reference)
//!
//! * **Antithetic variates**: pairs `(Z, -Z)`; the standard error is
//!   computed on the *pair averages* (the correct estimator for
//!   correlated pairs).
//! * **Control variate**: the discounted terminal stock `exp(-rT) S_T`
//!   with known mean `S exp(-qT)` (martingale property); the optimal
//!   coefficient is estimated from the sample covariance.
//!
//! Every entry point takes an explicit `u64` seed feeding the in-crate
//! SplitMix64 -> xoshiro256++ generator ([`crate::rng`]). Same seed =>
//! **bit-identical** result on every platform.
//!
//! Conventions: continuously compounded annualised `r`, `q` (ACT/365F),
//! `t` in years, `sigma` annualised.

use crate::black_scholes::{bs_price, validate_inputs, OptionType, PricingError};
use crate::rng::Xoshiro256PlusPlus;

/// Two-sided 95% normal quantile (matches the Python reference constant).
const Z95: f64 = 1.959_963_984_540_054;

/// Monte Carlo estimate with statistical error bars.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{mc_price, OptionType};
/// let res = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                    50_000, true, true, 42).unwrap();
/// assert!(res.std_error > 0.0);
/// assert!(res.ci_low < res.price && res.price < res.ci_high);
/// assert!(res.contains(10.450583572185565)); // analytic BS well inside CI
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McResult {
    /// Point estimate of the present value, currency units.
    pub price: f64,
    /// Standard error of the estimator.
    pub std_error: f64,
    /// Lower edge of the two-sided 95% confidence interval.
    pub ci_low: f64,
    /// Upper edge of the two-sided 95% confidence interval.
    pub ci_high: f64,
    /// Number of simulated paths (antithetic pairs count as 2 paths).
    pub n_paths: usize,
}

impl McResult {
    /// Return `true` if `x` lies inside the 95% confidence interval.
    ///
    /// # Examples
    ///
    /// ```
    /// use eq_options_engine::McResult;
    /// let r = McResult { price: 10.0, std_error: 0.1,
    ///                    ci_low: 9.804, ci_high: 10.196, n_paths: 1000 };
    /// assert!(r.contains(10.1));
    /// assert!(!r.contains(11.0));
    /// ```
    pub fn contains(&self, x: f64) -> bool {
        self.ci_low <= x && x <= self.ci_high
    }
}

/// Build an [`McResult`] from i.i.d. per-draw samples (ddof = 1).
fn summary(samples: &[f64], n_paths: usize) -> McResult {
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / (n as f64);
    let se = if n > 1 {
        let var = samples
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / ((n - 1) as f64);
        (var / (n as f64)).sqrt()
    } else {
        0.0
    };
    McResult {
        price: mean,
        std_error: se,
        ci_low: mean - Z95 * se,
        ci_high: mean + Z95 * se,
        n_paths,
    }
}

/// Monte Carlo price of a European option under exact-scheme GBM.
///
/// # Arguments
///
/// * `s`, `k`, `t`, `r`, `sigma`, `q`, `option_type` — as in
///   [`bs_price`].
/// * `n_paths` — total number of paths (rounded up to even when
///   `antithetic`), `>= 2`.
/// * `antithetic` — use antithetic pairs `(Z, -Z)`.
/// * `control_variate` — use the discounted terminal stock as a control
///   variate with the sample-optimal coefficient.
/// * `seed` — explicit RNG seed; same seed => bit-identical result.
///
/// `t == 0` or `sigma == 0` are deterministic; the exact Black-Scholes
/// value is returned with `std_error = 0`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] on invalid inputs or `n_paths < 2`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{mc_price, OptionType};
/// let a = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                  10_000, true, true, 7).unwrap();
/// let b = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                  10_000, true, true, 7).unwrap();
/// assert_eq!(a.price.to_bits(), b.price.to_bits()); // bit-reproducible
/// ```
#[allow(clippy::too_many_arguments)]
pub fn mc_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
    n_paths: usize,
    antithetic: bool,
    control_variate: bool,
    seed: u64,
) -> Result<McResult, PricingError> {
    validate_inputs(s, k, t, sigma)?;
    if n_paths < 2 {
        return Err(PricingError::InvalidInput(format!(
            "n_paths must be >= 2, got {n_paths}"
        )));
    }
    if t == 0.0 || sigma == 0.0 {
        let exact = bs_price(s, k, t, r, sigma, q, option_type)?;
        return Ok(McResult {
            price: exact,
            std_error: 0.0,
            ci_low: exact,
            ci_high: exact,
            n_paths,
        });
    }

    let mut rng = Xoshiro256PlusPlus::new(seed);
    let disc = (-r * t).exp();
    let sign = option_type.sign();
    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let vol_sqrt_t = sigma * t.sqrt();
    let terminal = |z: f64| s * (drift + vol_sqrt_t * z).exp();

    // Draw normals: fresh draws first, then (for antithetic) their mirrors,
    // matching the Python layout `concatenate([z, -z])`.
    let (z, n_eff): (Vec<f64>, usize) = if antithetic {
        let half = n_paths.div_ceil(2);
        let mut z = vec![0.0_f64; 2 * half];
        let (fresh, mirror) = z.split_at_mut(half);
        rng.fill_standard_normal(fresh);
        for i in 0..half {
            mirror[i] = -fresh[i];
        }
        (z, 2 * half)
    } else {
        let mut z = vec![0.0_f64; n_paths];
        rng.fill_standard_normal(&mut z);
        (z, n_paths)
    };

    let mut payoff: Vec<f64> = z
        .iter()
        .map(|&zi| disc * (sign * (terminal(zi) - k)).max(0.0))
        .collect();

    if control_variate {
        // Control: discounted terminal stock, known mean S exp(-qT).
        let control: Vec<f64> = z.iter().map(|&zi| disc * terminal(zi)).collect();
        let control_mean = s * (-q * t).exp();
        let n = n_eff as f64;
        let mean_p = payoff.iter().sum::<f64>() / n;
        let mean_c = control.iter().sum::<f64>() / n;
        let mut cov_pc = 0.0;
        let mut var_c = 0.0;
        for i in 0..n_eff {
            let dp = payoff[i] - mean_p;
            let dc = control[i] - mean_c;
            cov_pc += dp * dc;
            var_c += dc * dc;
        }
        cov_pc /= n - 1.0;
        var_c /= n - 1.0;
        let beta = if var_c > 0.0 { cov_pc / var_c } else { 0.0 };
        for i in 0..n_eff {
            payoff[i] -= beta * (control[i] - control_mean);
        }
    }

    if antithetic {
        let half = n_eff / 2;
        let samples: Vec<f64> = (0..half)
            .map(|i| 0.5 * (payoff[i] + payoff[half + i]))
            .collect();
        Ok(summary(&samples, n_eff))
    } else {
        Ok(summary(&payoff, n_eff))
    }
}
