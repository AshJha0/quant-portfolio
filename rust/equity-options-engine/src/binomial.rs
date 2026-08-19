//! Cox-Ross-Rubinstein binomial tree for European and American options.
//!
//! Conventions match [`crate::black_scholes`]: continuously compounded
//! annualised `r` and `q` (ACT/365F), `t` in years, `sigma` annualised.
//!
//! The backward induction runs in a **single reusable vector** (O(n)
//! memory) with one pass per time slice — no 2-D tree is ever
//! materialised.
//!
//! # Edge-case policy (identical to the Python reference)
//!
//! * `t == 0` -> intrinsic value.
//! * `sigma == 0` -> the world is deterministic: the European price is the
//!   discounted forward intrinsic (identical to Black-Scholes), and the
//!   American price is the maximum over the time grid of the discounted
//!   intrinsic along the deterministic path `S exp((r - q) t_i)`.
//! * Negative `s`, `k`, `t`, `sigma` and `n_steps == 0` are errors.

use crate::black_scholes::{
    bs_price, validate_inputs, validate_rates, OptionType, PricingError,
};

/// Largest accepted `n_steps`.
///
/// Backward induction costs O(n^2) time and O(n) memory, so 1e7 steps is
/// already far past any practical accuracy/latency trade-off (the tree
/// converges to Black-Scholes at O(1/n)). Larger requests are rejected
/// as [`PricingError::InvalidInput`] instead of being left to hang or to
/// abort in the allocator.
pub const MAX_STEPS: usize = 10_000_000;

/// Exercise style of the option.
///
/// # Examples
///
/// ```
/// use eq_options_engine::Exercise;
/// assert_ne!(Exercise::European, Exercise::American);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exercise {
    /// Exercisable at expiry only.
    European,
    /// Exercisable at any node on the tree.
    American,
}

/// Price under `sigma == 0`: the stock grows deterministically at `r - q`.
fn deterministic_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    exercise: Exercise,
    n_steps: usize,
) -> Result<f64, PricingError> {
    if exercise == Exercise::European {
        return bs_price(s, k, t, r, 0.0, q, option_type);
    }
    let sign = option_type.sign();
    let mut best = 0.0_f64;
    for i in 0..=n_steps {
        let ti = t * (i as f64) / (n_steps as f64);
        let path = s * ((r - q) * ti).exp();
        let payoff = (sign * (path - k)).max(0.0);
        best = best.max((-r * ti).exp() * payoff);
    }
    Ok(best)
}

/// Cox-Ross-Rubinstein binomial price of a European or American option.
///
/// Uses `u = exp(sigma sqrt(dt))`, `d = 1/u` and risk-neutral probability
/// `p = (exp((r - q) dt) - d) / (u - d)`. Terminal node prices are formed
/// in log space for stability at large `n_steps`; American options compare
/// continuation value with intrinsic at every node. Convergence to
/// Black-Scholes for European options is O(1/n) with an oscillating
/// odd/even term.
///
/// # Arguments
///
/// * `s`, `k`, `t`, `r`, `sigma`, `q`, `option_type` — as in
///   [`bs_price`].
/// * `exercise` — [`Exercise::European`] or [`Exercise::American`].
/// * `n_steps` — number of time steps, in `[1, MAX_STEPS]`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] on negative/NaN inputs, `n_steps` outside
/// `[1, MAX_STEPS]`,
/// or if the risk-neutral probability falls outside (0, 1) — a sign that
/// `dt` is too large for the given `r - q` and `sigma`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, crr_price, Exercise, OptionType};
/// let ot = OptionType::Put;
/// let tree = crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot, Exercise::European, 2000).unwrap();
/// let bs = bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot).unwrap();
/// assert!((tree - bs).abs() < 2e-3);
///
/// // American put is worth at least the European:
/// let amer = crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot, Exercise::American, 2000).unwrap();
/// assert!(amer >= tree - 1e-12);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn crr_price(
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
    if n_steps < 1 {
        return Err(PricingError::InvalidInput(
            "n_steps must be >= 1, got 0".into(),
        ));
    }
    if n_steps > MAX_STEPS {
        return Err(PricingError::InvalidInput(format!(
            "n_steps must be <= {MAX_STEPS}, got {n_steps} \
             (backward induction is O(n^2) work and O(n) memory)"
        )));
    }

    let sign = option_type.sign();
    if t == 0.0 {
        return Ok((sign * (s - k)).max(0.0));
    }
    if s == 0.0 {
        return Ok(match (option_type, exercise) {
            (OptionType::Call, _) => 0.0,
            (OptionType::Put, Exercise::American) => k,
            (OptionType::Put, Exercise::European) => k * (-r * t).exp(),
        });
    }
    if k == 0.0 {
        if option_type == OptionType::Put {
            return Ok(0.0);
        }
        // Zero-strike call: American early exercise is optimal iff q > 0.
        if exercise == Exercise::American && q > 0.0 {
            return Ok(s);
        }
        return Ok(s * (-q * t).exp());
    }
    if sigma == 0.0 {
        return deterministic_price(s, k, t, r, q, option_type, exercise, n_steps);
    }

    let dt = t / (n_steps as f64);
    let log_u = sigma * dt.sqrt();
    let u = log_u.exp();
    let d = 1.0 / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    if !(p > 0.0 && p < 1.0) {
        return Err(PricingError::InvalidInput(format!(
            "risk-neutral probability p={p:.6} outside (0, 1); \
             increase n_steps or check r, q, sigma"
        )));
    }
    let disc = (-r * dt).exp();
    let (pu, pd) = (disc * p, disc * (1.0 - p));

    // Terminal stock prices S u^j d^(n-j) = exp(log S + (2j - n) log u),
    // j = 0..=n, formed in log space for numerical stability.
    let log_s = s.ln();
    let n = n_steps as f64;
    let mut values: Vec<f64> = (0..=n_steps)
        .map(|j| {
            let st = (log_s + (2.0 * (j as f64) - n) * log_u).exp();
            (sign * (st - k)).max(0.0)
        })
        .collect();

    let american = exercise == Exercise::American;
    for i in (0..n_steps).rev() {
        for j in 0..=i {
            let mut v = pu * values[j + 1] + pd * values[j];
            if american {
                let node = (log_s + (2.0 * (j as f64) - (i as f64)) * log_u).exp();
                v = v.max(sign * (node - k));
            }
            values[j] = v;
        }
    }
    Ok(values[0])
}

/// American-minus-European value on the same CRR tree.
///
/// Using the *same* tree for both legs cancels the O(1/n) discretisation
/// error, so the premium is accurate to much better than either price.
/// Floored at 0 to remove residual floating-point noise.
///
/// # Errors
///
/// Propagates any [`PricingError`] from [`crr_price`].
///
/// # Examples
///
/// ```
/// use eq_options_engine::{early_exercise_premium, OptionType};
/// // ITM American put on a non-dividend stock carries a premium:
/// let p = early_exercise_premium(90.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put, 500).unwrap();
/// assert!(p > 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn early_exercise_premium(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
    n_steps: usize,
) -> Result<f64, PricingError> {
    let amer = crr_price(s, k, t, r, sigma, q, option_type, Exercise::American, n_steps)?;
    let euro = crr_price(s, k, t, r, sigma, q, option_type, Exercise::European, n_steps)?;
    Ok((amer - euro).max(0.0))
}
