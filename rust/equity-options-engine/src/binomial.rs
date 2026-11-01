//! Cox-Ross-Rubinstein binomial tree for European and American options.
//!
//! Conventions match [`crate::black_scholes`]: continuously compounded
//! annualised `r` and `q` (ACT/365F), `t` in years, `sigma` annualised.
//!
//! The backward induction runs in a **single reusable vector** (O(n)
//! memory) with one pass per time slice: no 2-D tree is ever
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

use std::cell::Cell;

use crate::black_scholes::{
    bs_price, finite_result, validate_inputs, validate_rates, OptionType, PricingError,
};
use crate::fast::ErrorCode;

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
#[allow(clippy::too_many_arguments)]
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
    finite_result(
        deterministic_american(s, k, t, r, q, option_type, n_steps),
        "crr_price",
    )
}

/// American value under `sigma == 0`: the best discounted exercise over
/// the `n_steps + 1` tree dates. Can be non-finite; callers check.
#[inline]
pub(crate) fn deterministic_american(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
    n_steps: usize,
) -> f64 {
    let sign = option_type.sign();
    let mut best = 0.0_f64;
    for i in 0..=n_steps {
        let ti = t * (i as f64) / (n_steps as f64);
        // e^{-r t_i} max(+-(S e^{(r-q) t_i} - K), 0) without forming the
        // forward, which overflows when (r - q) t_i > 709.
        let payoff = (sign * (s * (-q * ti).exp() - k * (-r * ti).exp())).max(0.0);
        best = best.max(payoff);
    }
    best
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
/// * `s`, `k`, `t`, `r`, `sigma`, `q`, `option_type`: as in
///   [`bs_price`].
/// * `exercise`: [`Exercise::European`] or [`Exercise::American`].
/// * `n_steps`: number of time steps, in `[1, MAX_STEPS]`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] on negative/NaN inputs, `n_steps` outside
/// `[1, MAX_STEPS]`,
/// or if the risk-neutral probability falls outside (0, 1): a sign that
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
            // Exercise now pays K; waiting pays K e^{-rT}, which is larger
            // when r < 0.
            (OptionType::Put, Exercise::American) => k.max(finite_result(k * (-r * t).exp(), "crr_price")?),
            (OptionType::Put, Exercise::European) => finite_result(k * (-r * t).exp(), "crr_price")?,
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
        return finite_result(s * (-q * t).exp(), "crr_price");
    }
    if sigma == 0.0 {
        return deterministic_price(s, k, t, r, q, option_type, exercise, n_steps);
    }

    let mut ws = CrrWorkspace::new();
    match crr_tree(s, k, t, r, sigma, q, option_type, exercise, n_steps, &mut ws) {
        Ok(v) => Ok(v),
        Err(ErrorCode::BadProbability) => {
            // Error path only: rebuild the message with the value of p.
            let dt = t / (n_steps as f64);
            let u = (sigma * dt.sqrt()).exp();
            let d = 1.0 / u;
            let p = (((r - q) * dt).exp() - d) / (u - d);
            Err(PricingError::InvalidInput(format!(
                "risk-neutral probability p={p:.6} outside (0, 1);                  increase n_steps or check r, q, sigma"
            )))
        }
        Err(_) => finite_result(f64::NAN, "crr_price"),
    }
}

/// Reusable buffers for the CRR tree.
///
/// Holds the node values and the exercise values. Capacity grows to the
/// largest `n_steps` seen and is then reused, so repeated pricing with
/// [`crate::fast::try_crr_price`] does not touch the heap.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{fast::try_crr_price, CrrWorkspace, Exercise, OptionType};
/// let mut ws = CrrWorkspace::with_steps(500);
/// let v = try_crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put,
///                       Exercise::American, 500, &mut ws).unwrap();
/// assert!(v > 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CrrWorkspace {
    values: Vec<f64>,
    exercise_value: Vec<f64>,
}

impl CrrWorkspace {
    /// Empty workspace. The first call sizes it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Workspace pre-sized for trees of up to `n_steps` steps (American
    /// or European), so the first call does not allocate either.
    pub fn with_steps(n_steps: usize) -> Self {
        Self {
            values: Vec::with_capacity(n_steps + 1),
            exercise_value: Vec::with_capacity(2 * n_steps + 1),
        }
    }
}

/// Resize `v` to `len` without shrinking capacity. Allocates only when
/// `len` exceeds the capacity already held.
#[inline]
fn resize_buf(v: &mut Vec<f64>, len: usize) {
    v.clear();
    v.resize(len, 0.0);
}

/// Backward induction on the CRR tree for validated inputs with
/// `s, k, t, sigma > 0` and `1 <= n_steps <= MAX_STEPS`.
///
/// Returns [`ErrorCode::BadProbability`] when the risk-neutral
/// probability leaves (0, 1), [`ErrorCode::NonFiniteResult`] when the
/// value overflows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn crr_tree(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
    exercise: Exercise,
    n_steps: usize,
    ws: &mut CrrWorkspace,
) -> Result<f64, ErrorCode> {
    let sign = option_type.sign();
    let dt = t / (n_steps as f64);
    let log_u = sigma * dt.sqrt();
    let u = log_u.exp();
    let d = 1.0 / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    if !(p > 0.0 && p < 1.0) {
        return Err(ErrorCode::BadProbability);
    }
    let disc = (-r * dt).exp();
    let (pu, pd) = (disc * p, disc * (1.0 - p));

    // Every node spot is S u^m for m = -n..=n. Step i, node j sits at
    // m = 2j - i. The American tree forms the exercise values once in log
    // space (O(n) exps) so backward induction does no transcendental
    // work. The European tree needs only the terminal layer.
    let log_s = s.ln();
    let n_f = n_steps as f64;
    let american = exercise == Exercise::American;
    resize_buf(&mut ws.values, n_steps + 1);
    if american {
        resize_buf(&mut ws.exercise_value, 2 * n_steps + 1);
        for (idx, e) in ws.exercise_value.iter_mut().enumerate() {
            let m = idx as f64 - n_f;
            *e = sign * ((log_s + m * log_u).exp() - k);
        }
        for (v, e) in ws.values.iter_mut().zip(ws.exercise_value.iter().step_by(2)) {
            *v = e.max(0.0);
        }
    } else {
        for (j, v) in ws.values.iter_mut().enumerate() {
            let m = (2 * j) as f64 - n_f;
            *v = (sign * ((log_s + m * log_u).exp() - k)).max(0.0);
        }
    }

    let values = Cell::from_mut(ws.values.as_mut_slice()).as_slice_of_cells();
    for i in (0..n_steps).rev() {
        // Nodes 0..=i of step i read nodes j and j + 1 of step i + 1.
        let Some(layer) = values.get(..i + 2) else {
            debug_assert!(false, "values shorter than n_steps + 1");
            return Err(ErrorCode::NonFiniteResult);
        };
        if american {
            let offset = n_steps - i; // index of m = -i
            let Some(ex) = ws.exercise_value.get(offset..) else {
                debug_assert!(false, "exercise values shorter than 2 n_steps + 1");
                return Err(ErrorCode::NonFiniteResult);
            };
            for (w, &e) in layer.windows(2).zip(ex.iter().step_by(2)) {
                let v = pu * w[1].get() + pd * w[0].get();
                w[0].set(v.max(e));
            }
        } else {
            for w in layer.windows(2) {
                w[0].set(pu * w[1].get() + pd * w[0].get());
            }
        }
    }
    match values.first() {
        Some(v) if v.get().is_finite() => Ok(v.get()),
        _ => Err(ErrorCode::NonFiniteResult),
    }
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
