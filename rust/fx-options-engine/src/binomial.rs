//! Cox–Ross–Rubinstein binomial tree for FX options.
//!
//! The foreign interest rate enters exactly like a continuous dividend
//! yield: risk-neutral drift of the spot under the domestic measure is
//! `r_d - r_f`, so the up-move probability is
//!
//! ```text
//! p = (e^{(r_d - r_f) dt} - d) / (u - d),   u = e^{sigma sqrt(dt)},  d = 1/u.
//! ```
//!
//! Supports European and American exercise.  American FX options trade
//! OTC; the economically interesting case is an American *call* on a
//! high-yielding foreign currency (`r_f > r_d`): the foreign carry lost
//! by holding the option instead of the currency makes early exercise
//! optimal, giving the American call a strictly positive premium over
//! European: mirroring the dividend-yield story for equities.
//!
//! The lattice is the shared [`qp_core::crr`] kernel (W20), called with
//! `r = r_d` and `q = r_f`.

use qp_core::crr::{CrrError, CrrWorkspace, crr_lattice};

use crate::garman_kohlhagen::gk_price;
use crate::{FxResult, OptionType, invalid, validate_inputs};

/// Largest accepted `steps`.
///
/// Backward induction costs O(n^2) time and O(n) memory, so 1e7 steps is
/// already far beyond any useful accuracy/latency trade-off (the tree
/// converges at O(1/n)). Larger requests return
/// [`crate::FxError::InvalidInput`] rather than hanging for hours or
/// aborting inside the allocator.
pub const MAX_STEPS: u32 = 10_000_000;

/// Exercise style for [`binomial_price`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exercise {
    /// Exercise at expiry only.
    European,
    /// Exercise at any tree node.
    American,
}

/// CRR binomial price of an FX option.
///
/// # Arguments
///
/// * `s`, `k`, `t`, `r_d`, `r_f`, `sigma`: as in [`crate::gk_price`].
/// * `option_type`: call or put on the base currency.
/// * `steps`: number of tree steps, in `[1, MAX_STEPS]`.
/// * `exercise`: [`Exercise::European`] or [`Exercise::American`].
///
/// Returns the price in domestic currency per unit foreign notional.
///
/// # Errors
///
/// [`crate::FxError::InvalidInput`] on invalid inputs, `steps` outside
/// `[1, MAX_STEPS]`, or
/// if the tree probability falls outside `(0, 1)` (time step too coarse
/// for the drift/vol combination).
///
/// ```
/// use fx_options_engine::{binomial_price, gk_price, Exercise, OptionType};
/// let (s, k, t, rd, rf, vol) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);
/// let tree = binomial_price(s, k, t, rd, rf, vol, OptionType::Call, 500,
///                           Exercise::European).unwrap();
/// let gk = gk_price(s, k, t, rd, rf, vol, OptionType::Call).unwrap();
/// assert!((tree - gk).abs() < 5e-5); // tree -> GK convergence
/// ```
pub fn binomial_price(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
    steps: u32,
    exercise: Exercise,
) -> FxResult<f64> {
    let phi = option_type.phi();
    validate_inputs(s, k, t, r_d, r_f, sigma)?;
    if steps < 1 {
        return invalid(format!("steps must be a positive integer, got {steps}"));
    }
    if steps > MAX_STEPS {
        return invalid(format!(
            "steps must be <= {MAX_STEPS}, got {steps} (backward induction \
             is O(n^2) work and O(n) memory)"
        ));
    }
    if t == 0.0 {
        return Ok((phi * (s - k)).max(0.0));
    }
    if sigma == 0.0 {
        // Degenerate tree; defer to the analytic limit (European) or
        // deterministic exercise optimisation (American on a drifting spot).
        if exercise == Exercise::European {
            return gk_price(s, k, t, r_d, r_f, sigma, option_type);
        }
        let mut best: f64 = 0.0;
        for i in 0..=steps {
            let ti = t * f64::from(i) / f64::from(steps);
            // Discounted payoff without forming the forward (overflow-safe).
            let value = (phi * (s * (-r_f * ti).exp() - k * (-r_d * ti).exp())).max(0.0);
            best = best.max(value);
        }
        return crate::finite_result(best, "binomial_price");
    }

    // Shared lattice (qp_core::crr, W20): the equity tree with q = r_f.
    let mut ws = CrrWorkspace::with_steps(steps as usize);
    match crr_lattice(
        s,
        k,
        t,
        r_d,
        r_f,
        sigma,
        phi,
        exercise == Exercise::American,
        steps as usize,
        &mut ws,
    ) {
        Ok(v) => Ok(v),
        Err(CrrError::BadProbability(p)) => invalid(format!(
            "risk-neutral probability {p:.6} outside (0, 1); increase steps              (dt too large for |r_d - r_f| vs sigma)"
        )),
        Err(CrrError::NonFinite) => crate::finite_result(f64::NAN, "binomial_price"),
    }
}
