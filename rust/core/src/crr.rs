//! The one Cox-Ross-Rubinstein lattice shared by the equity and FX engines
//! (W20, Round 8).
//!
//! The plain idea. Cut `[0, T]` into `n` steps; the spot moves up by `u` or
//! down by `d = 1/u` each step with risk-neutral probability `p`; roll the
//! payoff back to today, taking the larger of continuation and exercise at
//! every node for American options.
//!
//! The formal statement. `dt = T/n`, `u = exp(sigma sqrt(dt))`, `d = 1/u`,
//! `p = (exp((r - q) dt) - d) / (u - d)` (Cox, Ross and Rubinstein 1979,
//! J. Financial Economics 7, eqs. 2-3, with Merton's continuous yield `q`).
//! FX is the same lattice with `r = r_d` and `q = r_f` (Garman and
//! Kohlhagen 1983). Node spots `S u^m` are formed as `exp(ln S + m ln u)`.
//! Same operations and order as the C++ `qp/crr.hpp` and the Python
//! `qp_core.trees.crr_lattice`.
//!
//! Worked example. `S = K = 100`, `T = 1`, `r = 5 %`, `q = 2 %`,
//! `sigma = 20 %`, European call, one step: 11.07354 (Python corpus row 0,
//! `tests/data/kernel_corpus.json`).
//!
//! The caller validates inputs and handles `T = 0`, `sigma = 0`, `S = 0`
//! and `K = 0` under its own documented policy.

use std::cell::Cell;

/// Why [`crr_lattice`] produced no value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrrError {
    /// The risk-neutral probability (carried) is outside `(0, 1)`: the
    /// time step is too coarse for `r - q` and `sigma`.
    BadProbability(f64),
    /// The lattice value is not finite.
    NonFinite,
}

/// Reusable buffers for the CRR tree.
///
/// Holds the node values and the exercise values. Capacity grows to the
/// largest `n_steps` seen and is then reused, so repeated pricing with
/// `eq_options_engine::fast::try_crr_price` does not touch the heap.
///
/// # Examples
///
/// ```
/// use qp_core::crr::{CrrWorkspace, crr_lattice};
/// let mut ws = CrrWorkspace::with_steps(500);
/// let v = crr_lattice(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, -1.0, true, 500, &mut ws).unwrap();
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

/// Lattice value for validated inputs with `s, k, t, sigma > 0` and
/// `n_steps >= 1`; `sign` is `+1.0` for a call and `-1.0` for a put.
///
/// # Errors
///
/// [`CrrError::BadProbability`] when the risk-neutral probability leaves
/// `(0, 1)`, [`CrrError::NonFinite`] when the value overflows.
#[allow(clippy::too_many_arguments)]
pub fn crr_lattice(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    sign: f64,
    american: bool,
    n_steps: usize,
    ws: &mut CrrWorkspace,
) -> Result<f64, CrrError> {
    let dt = t / (n_steps as f64);
    let log_u = sigma * dt.sqrt();
    let u = log_u.exp();
    let d = 1.0 / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    if !(p > 0.0 && p < 1.0) {
        return Err(CrrError::BadProbability(p));
    }
    let disc = (-r * dt).exp();
    let (pu, pd) = (disc * p, disc * (1.0 - p));

    // Every node spot is S u^m for m = -n..=n. Step i, node j sits at
    // m = 2j - i. The American tree forms the exercise values once in log
    // space (O(n) exps) so backward induction does no transcendental
    // work. The European tree needs only the terminal layer.
    let log_s = s.ln();
    let n_f = n_steps as f64;
    resize_buf(&mut ws.values, n_steps + 1);
    if american {
        resize_buf(&mut ws.exercise_value, 2 * n_steps + 1);
        for (idx, e) in ws.exercise_value.iter_mut().enumerate() {
            let m = idx as f64 - n_f;
            *e = sign * ((log_s + m * log_u).exp() - k);
        }
        for (v, e) in ws
            .values
            .iter_mut()
            .zip(ws.exercise_value.iter().step_by(2))
        {
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
            return Err(CrrError::NonFinite);
        };
        if american {
            let offset = n_steps - i; // index of m = -i
            let Some(ex) = ws.exercise_value.get(offset..) else {
                debug_assert!(false, "exercise values shorter than 2 n_steps + 1");
                return Err(CrrError::NonFinite);
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
        _ => Err(CrrError::NonFinite),
    }
}
