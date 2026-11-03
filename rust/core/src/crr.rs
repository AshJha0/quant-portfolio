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
    next: Vec<f64>,
    exercise_value: Vec<f64>,
    exercise_odd: Vec<f64>,
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
            next: Vec::with_capacity(n_steps + 1),
            exercise_value: Vec::with_capacity(n_steps + 1),
            exercise_odd: Vec::with_capacity(n_steps),
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
    let n = n_steps;
    // Exercise values phi (S u^m - K), m = idx - n, split by index parity so
    // that step i reads a contiguous run (node j of step i sits at
    // idx = (n - i) + 2 j, whose parity is fixed within the step). The
    // induction double-buffers, so the inner loop has no loop-carried
    // dependency or stride and vectorises; the arithmetic per node is
    // unchanged (pu V_up + pd V_down, then max with exercise).
    let ex_at = |idx: usize| sign * ((log_s + (idx as f64 - n_f) * log_u).exp() - k);
    resize_buf(&mut ws.values, n + 1);
    resize_buf(&mut ws.next, n + 1);
    if american {
        ws.exercise_value.clear();
        ws.exercise_value.extend((0..=n).map(|h| ex_at(2 * h)));
        ws.exercise_odd.clear();
        ws.exercise_odd.extend((0..n).map(|h| ex_at(2 * h + 1)));
        for (v, e) in ws.values.iter_mut().zip(ws.exercise_value.iter()) {
            *v = e.max(0.0);
        }
    } else {
        for (j, v) in ws.values.iter_mut().enumerate() {
            *v = ex_at(2 * j).max(0.0);
        }
    }

    for i in (0..n).rev() {
        let m = i + 1;
        let cur = ws.values.get(..m + 1).unwrap_or(&[]);
        let nxt = ws.next.get_mut(..m).unwrap_or(&mut []);
        let lo = cur.get(..m).unwrap_or(&[]);
        let hi = cur.get(1..).unwrap_or(&[]);
        if american {
            let offset = n - i; // index of m = -i
            let ex = if offset.is_multiple_of(2) {
                ws.exercise_value.get(offset / 2..).unwrap_or(&[])
            } else {
                ws.exercise_odd.get(offset / 2..).unwrap_or(&[])
            };
            let ex = ex.get(..m).unwrap_or(&[]);
            debug_assert!(nxt.len() == m && ex.len() == m);
            for (((o, &a), &b), &e) in nxt.iter_mut().zip(lo).zip(hi).zip(ex) {
                *o = (pu * b + pd * a).max(e);
            }
        } else {
            for ((o, &a), &b) in nxt.iter_mut().zip(lo).zip(hi) {
                *o = pu * b + pd * a;
            }
        }
        std::mem::swap(&mut ws.values, &mut ws.next);
    }
    let values = &ws.values;
    match values.first() {
        Some(&v) if v.is_finite() => Ok(v),
        _ => Err(CrrError::NonFinite),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-place lattice this module used before the double-buffered
    /// rewrite (Round 8 perf fix): spot `exp(ln S + m ln u)`, then
    /// `V_j <- max(pu V_{j+1} + pd V_j, exercise)` in place.
    #[allow(clippy::too_many_arguments, clippy::indexing_slicing)]
    fn reference(
        s: f64,
        k: f64,
        t: f64,
        r: f64,
        q: f64,
        sigma: f64,
        sign: f64,
        american: bool,
        n: usize,
    ) -> f64 {
        let dt = t / (n as f64);
        let log_u = sigma * dt.sqrt();
        let u = log_u.exp();
        let d = 1.0 / u;
        let p = (((r - q) * dt).exp() - d) / (u - d);
        let disc = (-r * dt).exp();
        let (pu, pd) = (disc * p, disc * (1.0 - p));
        let log_s = s.ln();
        let ex: Vec<f64> = (0..=2 * n)
            .map(|idx| sign * ((log_s + (idx as f64 - n as f64) * log_u).exp() - k))
            .collect();
        let mut v: Vec<f64> = (0..=n).map(|j| ex[2 * j].max(0.0)).collect();
        for i in (0..n).rev() {
            for j in 0..=i {
                let c = pu * v[j + 1] + pd * v[j];
                v[j] = if american {
                    c.max(ex[n - i + 2 * j])
                } else {
                    c
                };
            }
        }
        v[0]
    }

    #[test]
    fn double_buffered_lattice_is_bit_identical_to_in_place_reference() {
        let mut ws = CrrWorkspace::new();
        let mut checked = 0;
        for &s in &[50.0, 90.0, 100.0, 110.0, 250.0] {
            for &sigma in &[0.05, 0.2, 0.6] {
                for &(r, q) in &[(0.05, 0.02), (0.0, 0.0), (0.01, 0.07), (-0.005, 0.03)] {
                    for &n in &[1usize, 2, 3, 17, 100, 501] {
                        for &sign in &[1.0, -1.0] {
                            for &american in &[false, true] {
                                let got = crr_lattice(
                                    s, 100.0, 0.75, r, q, sigma, sign, american, n, &mut ws,
                                );
                                let want =
                                    reference(s, 100.0, 0.75, r, q, sigma, sign, american, n);
                                if let Ok(g) = got {
                                    assert_eq!(
                                        g.to_bits(),
                                        want.to_bits(),
                                        "s={s} sigma={sigma} r={r} q={q} n={n} sign={sign} am={american}"
                                    );
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 1400, "{checked}");
    }
}
