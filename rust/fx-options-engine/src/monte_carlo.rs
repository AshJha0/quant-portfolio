//! Monte Carlo pricing of FX options under the domestic risk-neutral
//! measure.
//!
//! Under the domestic money-market numeraire the spot follows
//!
//! ```text
//! dS/S = (r_d - r_f) dt + sigma dW^d,
//! ```
//!
//! so terminal spot is sampled exactly:
//! `S_T = S exp((r_d - r_f - sigma^2/2) T + sigma sqrt(T) Z)`.
//!
//! Variance reduction:
//!
//! * **Antithetic variates** — pairs `(Z, -Z)`, estimator averaged per
//!   pair (standard error computed from independent pair averages).
//! * **Control variate** — the discounted terminal spot `e^{-r_d T} S_T`
//!   is a martingale-adjusted quantity with known mean `S e^{-r_f T}`
//!   (the value today of one unit of foreign currency delivered at `T`);
//!   the optimal coefficient is estimated from the sample covariance.
//!
//! When both are enabled the control-variate coefficient is fitted on the
//! *antithetic pair averages*, not on the raw per-draw values: pairing
//! already reshapes the payoff/control covariance, so the beta that
//! minimises the variance of the paired estimator is generally different
//! from (and better than) the per-draw beta. Fitting on raw draws and
//! pairing afterwards would leave `antithetic+control_variate` no better
//! than `control_variate` alone on some inputs.
//!
//! Determinism: draws come from the crate's own
//! [`Xoshiro256StarStar`](crate::rng::Xoshiro256StarStar) seeded with an
//! explicit `u64`, and normals are produced by inverse-CDF transform —
//! the same seed gives **bit-identical** results on every platform and
//! release (asserted in the test suite).

use crate::rng::Xoshiro256StarStar;
use crate::{invalid, validate_inputs, FxResult, OptionType};

/// Monte Carlo estimate with sampling-error diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct McResult {
    /// Point estimate, domestic ccy per unit foreign notional.
    pub price: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Lower edge of the 95% confidence interval (`price - 1.96 SE`).
    pub ci_low: f64,
    /// Upper edge of the 95% confidence interval (`price + 1.96 SE`).
    pub ci_high: f64,
    /// Number of underlying draws (antithetic pairs count as 2).
    pub n_paths: u64,
    /// Description of variance-reduction techniques applied.
    pub method: &'static str,
}

/// Largest accepted `n_paths`.
///
/// The estimator materialises one `f64` per path (two with a control
/// variate), so 1e9 paths already asks for 8-16 GB. Larger requests
/// return [`crate::FxError::InvalidInput`] instead of being left to
/// abort the process inside the allocator, and the cap also keeps the
/// internal `2 * n_paths.div_ceil(2)` path count away from `u64`/`usize`
/// overflow.
pub const MAX_PATHS: u64 = 1_000_000_000;

fn mean_of(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Monte Carlo GK price of a European FX vanilla.
///
/// # Arguments
///
/// * `s`, `k`, `t`, `r_d`, `r_f`, `sigma` — as in [`crate::gk_price`];
///   requires `t > 0`.
/// * `option_type` — call or put on the base currency.
/// * `n_paths` — number of terminal draws, in `[2, MAX_PATHS]` (rounded
///   up to even when antithetic). With `antithetic` and `n_paths = 2`
///   there is a single independent unit, so `std_error` is reported as
///   `0.0` (no sample variance exists) and the confidence interval
///   collapses to the point estimate.
/// * `seed` — explicit RNG seed; the same seed is bit-reproducible.
/// * `antithetic`, `control_variate` — variance-reduction switches.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs, `t <= 0`, or
/// `n_paths` outside `[2, MAX_PATHS]`.
///
/// ```
/// use fx_options_engine::{gk_price, mc_price, OptionType};
/// let (s, k, t, rd, rf, vol) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);
/// let mc = mc_price(s, k, t, rd, rf, vol, OptionType::Call, 50_000, 7,
///                   true, true).unwrap();
/// let exact = gk_price(s, k, t, rd, rf, vol, OptionType::Call).unwrap();
/// assert!((mc.price - exact).abs() < 3.0 * mc.std_error);
/// assert_eq!(mc.method, "antithetic+control_variate");
/// ```
#[allow(clippy::too_many_arguments)]
pub fn mc_price(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
    n_paths: u64,
    seed: u64,
    antithetic: bool,
    control_variate: bool,
) -> FxResult<McResult> {
    let phi = option_type.phi();
    validate_inputs(s, k, t, r_d, r_f, sigma)?;
    if t <= 0.0 {
        return invalid("mc_price requires T > 0");
    }
    if n_paths < 2 {
        return invalid(format!("n_paths must be >= 2, got {n_paths}"));
    }
    if n_paths > MAX_PATHS {
        return invalid(format!(
            "n_paths must be <= {MAX_PATHS}, got {n_paths} (the estimator \
             buffers 8-16 bytes per path)"
        ));
    }

    let drift = (r_d - r_f - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let df_d = (-r_d * t).exp();

    // Terminal spots: draws z then mirrored -z when antithetic
    // (layout [z_0..z_{h-1}, -z_0..-z_{h-1}] so pair i = (i, i+half)).
    let mut rng = Xoshiro256StarStar::new(seed);
    let (half, total) = if antithetic {
        let h = n_paths.div_ceil(2) as usize;
        (h, 2 * h)
    } else {
        (0, n_paths as usize)
    };
    let mut discounted = vec![0.0f64; total];
    let mut control = if control_variate {
        vec![0.0f64; total]
    } else {
        Vec::new()
    };

    {
        let mut fill = |idx: usize, z: f64| {
            let s_t = s * (drift + vol * z).exp();
            discounted[idx] = df_d * (phi * (s_t - k)).max(0.0);
            if control_variate {
                control[idx] = df_d * s_t;
            }
        };
        if antithetic {
            for i in 0..half {
                let z = rng.next_normal();
                fill(i, z);
                fill(i + half, -z);
            }
        } else {
            for i in 0..total {
                let z = rng.next_normal();
                fill(i, z);
            }
        }
    }

    // Reduce to one value per *independent* draw first: the antithetic
    // pair average `0.5 (X(Z) + X(-Z))` when antithetic, the raw path
    // otherwise.  The control variate must then be fitted on these same
    // independent units, not on the raw (correlated, paired) draws: the
    // optimal beta for the pair-average estimator is
    // Cov(pair-avg payoff, pair-avg control) / Var(pair-avg control),
    // which differs from the individual-draw beta because antithetic
    // pairing already reshapes the payoff/control covariance structure.
    // Fitting beta on individual draws and pairing afterwards (as an
    // earlier version of this function did) under-uses the control
    // variate once antithetic is also on — the combined SE could end up
    // *worse* than control-variate alone.
    let mut samples: Vec<f64> = if antithetic {
        (0..half)
            .map(|i| 0.5 * (discounted[i] + discounted[i + half]))
            .collect()
    } else {
        discounted
    };

    // Control-variate adjustment: subtract beta * (control - known mean),
    // beta = sample cov(payoff, control) / var(control), computed on the
    // same independent units as `samples`.
    if control_variate {
        let control_mean_known = s * (-r_f * t).exp();
        let control_units: Vec<f64> = if antithetic {
            (0..half)
                .map(|i| 0.5 * (control[i] + control[i + half]))
                .collect()
        } else {
            control
        };
        let mp = mean_of(&samples);
        let mc = mean_of(&control_units);
        let mut cov = 0.0;
        let mut var_c = 0.0;
        for i in 0..samples.len() {
            let dc = control_units[i] - mc;
            cov += (samples[i] - mp) * dc;
            var_c += dc * dc;
        }
        let beta = if var_c > 0.0 { cov / var_c } else { 0.0 };
        for i in 0..samples.len() {
            samples[i] -= beta * (control_units[i] - control_mean_known);
        }
    }

    let price = mean_of(&samples);
    let ss: f64 = samples.iter().map(|v| (v - price) * (v - price)).sum();
    let n = samples.len() as f64;
    // A single independent unit (n_paths = 2 with antithetic pairing
    // collapses to one pair average) has no sample variance: report
    // SE = 0 rather than dividing by n - 1 = 0 and returning NaN.
    let se = if samples.len() > 1 {
        (ss / (n - 1.0)).sqrt() / n.sqrt()
    } else {
        0.0
    };

    let method = match (antithetic, control_variate) {
        (true, true) => "antithetic+control_variate",
        (true, false) => "antithetic+plain",
        (false, true) => "control_variate",
        (false, false) => "plain",
    };
    Ok(McResult {
        price,
        std_error: se,
        ci_low: price - 1.96 * se,
        ci_high: price + 1.96 * se,
        n_paths: total as u64,
        method,
    })
}
