//! Risk kernels: quote-to-node Jacobian algebra, par-rate DV01, quote gamma
//! with the bootstrap-curvature term, vega bucketing on vol grids and the
//! P&L-explain Taylor attribution (plan W14, Rust parity D1).
//!
//! Rust port of `python/core/src/qp_core/risk/{jacobian,vega_grid,pnl_explain}.py`
//! and of the algebra in `python/equity/04-fixed-income/src/fi_rates/risk/par.py`.
//! The C++ port is `cpp/core/include/qp/risk.hpp`. Plain idea, formal
//! statement, worked examples and sources are in the Python docstrings; in
//! short:
//!
//! * A curve is stored as nodes `z` and built from quotes `q = Q(z)`. With
//!   `J = dQ/dz`, the implicit-function theorem gives quote (instrument
//!   space) risk `g_q = J^-T g_z` (Andersen and Piterbarg, *Interest Rate
//!   Modeling* vol. 3, 2010, ch. 26). Per basis point: `1e-4 g_q`.
//! * Quote gamma is **not** `J^-T Gamma_z J^-1`: the bootstrap map is curved,
//!   and the correct second-order term is
//!   `J^-T (Gamma_z - sum_i g_q,i H_i) J^-1` with `H_i` the node Hessian of
//!   model quote `i`. The Python lane measured the shortcut at about 200% off
//!   on its test book; [`quote_gamma`] carries the full term.
//! * Vega bucketing: multilinear weights on a 2-D or 3-D grid with flat
//!   extrapolation (they sum to one), and bucket vegas by central
//!   differences of a caller's PV.
//! * P&L explain: per-factor `delta_i dx_i + 0.5 dx_i (S dx)_i` with
//!   `S = (Gamma + Gamma^T) / 2`, which sums to the second-order Taylor term.
//!
//! Parity: `tests/risk_parity.rs` reproduces the Python values in
//! `tests/data/risk_parity.txt` to 1e-12 (relative to the largest entry of
//! each output). The finite-difference inputs (Jacobian, Hessians) are frozen
//! in that file because a 1e-6 central difference amplifies last-bit `exp`
//! differences far beyond 1e-12; the algebra on top of them is what is
//! compared. [`fd_jacobian`] and [`cross_gamma`] are ported too and checked
//! against analytic derivatives.
//!
//! # Example
//!
//! ```
//! use qp_core::risk::node_to_quote_risk;
//! // Each quote moves at half the speed of its node: J = diag(-0.5, -0.5).
//! let jac = [-0.5, 0.0, 0.0, -0.5];
//! let g_q = node_to_quote_risk(&jac, 2, &[-2.0, -4.0]).unwrap();
//! assert_eq!(g_q, vec![4.0, 8.0]);
//! ```

use crate::Error;

/// Largest 1-norm condition number accepted by [`node_to_quote_risk`].
///
/// The Python reference rejects a 2-norm condition above 1e12; the 1-norm
/// condition is within a factor `n` of it, so for the curve sizes in use
/// (n <= 60) the two refuse the same Jacobians except in a thin band.
pub const MAX_CONDITION: f64 = 1e12;

fn check_square(m: &[f64], n: usize, name: &str) -> Result<(), Error> {
    if n == 0 || m.len() != n * n {
        return Err(Error::InvalidInput(format!(
            "{name} must be a non-empty {n}x{n} row-major matrix, got {} entries",
            m.len()
        )));
    }
    if m.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "{name} contains NaN or infinite values"
        )));
    }
    Ok(())
}

/// LU factorisation with partial pivoting of a row-major `n x n` matrix.
struct Lu {
    n: usize,
    a: Vec<f64>,
    piv: Vec<usize>,
}

impl Lu {
    fn new(m: &[f64], n: usize) -> Result<Self, Error> {
        let mut a = m.to_vec();
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut p = k;
            for i in k + 1..n {
                if a[i * n + k].abs() > a[p * n + k].abs() {
                    p = i;
                }
            }
            if a[p * n + k] == 0.0 {
                return Err(Error::InvalidInput(
                    "Jacobian is numerically singular (zero pivot)".into(),
                ));
            }
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
                piv.swap(k, p);
            }
            for i in k + 1..n {
                let f = a[i * n + k] / a[k * n + k];
                a[i * n + k] = f;
                for j in k + 1..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
            }
        }
        Ok(Self { n, a, piv })
    }

    /// Solves `A x = b`.
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x: Vec<f64> = self.piv.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.a[i * n + j] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= self.a[i * n + j] * x[j];
            }
            x[i] /= self.a[i * n + i];
        }
        x
    }

    /// Row-major inverse.
    fn inverse(&self) -> Vec<f64> {
        let n = self.n;
        let mut inv = vec![0.0; n * n];
        let mut e = vec![0.0; n];
        for j in 0..n {
            e.iter_mut().for_each(|v| *v = 0.0);
            e[j] = 1.0;
            let col = self.solve(&e);
            for i in 0..n {
                inv[i * n + j] = col[i];
            }
        }
        inv
    }
}

fn transpose(m: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = m[i * n + j];
        }
    }
    t
}

fn norm1(m: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| m[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Inverse of a square Jacobian, refusing a numerically singular one
/// (1-norm condition above [`MAX_CONDITION`]).
pub fn checked_inverse(jacobian: &[f64], n: usize) -> Result<Vec<f64>, Error> {
    check_square(jacobian, n, "Jacobian")?;
    let inv = Lu::new(jacobian, n)?.inverse();
    let cond = norm1(jacobian, n) * norm1(&inv, n);
    if !cond.is_finite() || cond > MAX_CONDITION {
        return Err(Error::InvalidInput(format!(
            "Jacobian is numerically singular (condition number {cond:.3e})"
        )));
    }
    Ok(inv)
}

/// Quote risk `dP/dq` from node risk `dP/dz`: solves `J^T g_q = g_z`.
///
/// `jacobian` is `J = dQ/dz`, row-major, quotes by nodes. Errors on a
/// non-square, non-finite or numerically singular `J`: a curve whose quotes
/// do not pin down its nodes has no instrument-space risk.
pub fn node_to_quote_risk(
    jacobian: &[f64],
    n: usize,
    node_risk: &[f64],
) -> Result<Vec<f64>, Error> {
    check_square(jacobian, n, "Jacobian")?;
    if node_risk.len() != n {
        return Err(Error::InvalidInput(format!(
            "node risk has {} rows, Jacobian has {n} nodes",
            node_risk.len()
        )));
    }
    checked_inverse(jacobian, n)?;
    Ok(Lu::new(&transpose(jacobian, n), n)?.solve(node_risk))
}

/// Inverse map: `dP/dz = J^T dP/dq`.
pub fn quote_to_node_risk(
    jacobian: &[f64],
    n: usize,
    quote_risk: &[f64],
) -> Result<Vec<f64>, Error> {
    check_square(jacobian, n, "Jacobian")?;
    if quote_risk.len() != n {
        return Err(Error::InvalidInput("quote risk length must equal n".into()));
    }
    Ok((0..n)
        .map(|j| (0..n).map(|i| jacobian[i * n + j] * quote_risk[i]).sum())
        .collect())
}

/// Par-rate DV01 per basis point from quote risk per unit rate (`1e-4 g_q`).
pub fn par_dv01(quote_risk: &[f64]) -> Vec<f64> {
    quote_risk.iter().map(|g| g * 1e-4).collect()
}

/// Quote-space cross-gamma `J^-T (Gamma_z - sum_i g_q,i H_i) J^-1`.
///
/// `gamma_z` is the node Hessian of the book (row-major `n x n`),
/// `quote_risk` is `g_q` from [`node_to_quote_risk`], and
/// `quote_hessians` holds the `n` node Hessians `H_i` of the model quotes,
/// one after another (`n * n * n` values). Dropping the sum gives the
/// shortcut `J^-T Gamma_z J^-1`, which is wrong whenever the bootstrap map
/// is curved (any real curve).
pub fn quote_gamma(
    jacobian: &[f64],
    n: usize,
    gamma_z: &[f64],
    quote_risk: &[f64],
    quote_hessians: &[f64],
) -> Result<Vec<f64>, Error> {
    let inv = checked_inverse(jacobian, n)?;
    check_square(gamma_z, n, "node gamma")?;
    if quote_risk.len() != n || quote_hessians.len() != n * n * n {
        return Err(Error::InvalidInput(format!(
            "need {n} quote risks and {} Hessian entries",
            n * n * n
        )));
    }
    let mut core = gamma_z.to_vec();
    for (jk, c) in core.iter_mut().enumerate() {
        let corr: f64 = (0..n)
            .map(|i| quote_risk[i] * quote_hessians[i * n * n + jk])
            .sum();
        *c -= corr;
    }
    Ok(matmul(&matmul(&transpose(&inv, n), &core, n), &inv, n))
}

fn check_step(h: f64) -> Result<(), Error> {
    if !h.is_finite() || h <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "finite-difference step must be > 0, got {h}"
        )));
    }
    Ok(())
}

/// Central-difference Jacobian `J[i][j] = d f_i / d x_j`, row-major
/// (`m x n` where `m = f(x).len()`). Truncation error `O(h^2)` per entry.
pub fn fd_jacobian<F: Fn(&[f64]) -> Vec<f64>>(f: F, x: &[f64], h: f64) -> Result<Vec<f64>, Error> {
    check_step(h)?;
    if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "x must be a non-empty finite vector".into(),
        ));
    }
    let n = x.len();
    let mut cols = Vec::with_capacity(n);
    let mut xp = x.to_vec();
    for j in 0..n {
        xp[j] = x[j] + h;
        let up = f(&xp);
        xp[j] = x[j] - h;
        let dn = f(&xp);
        xp[j] = x[j];
        cols.push(
            up.iter()
                .zip(&dn)
                .map(|(u, d)| (u - d) / (2.0 * h))
                .collect::<Vec<f64>>(),
        );
    }
    let m = cols[0].len();
    let mut out = vec![0.0; m * n];
    for (j, col) in cols.iter().enumerate() {
        for i in 0..m {
            out[i * n + j] = col[i];
        }
    }
    Ok(out)
}

/// Full Hessian by central differences (row-major, symmetric by
/// construction): diagonal `(f(x+h) - 2 f(x) + f(x-h)) / h^2`, cross terms
/// by the four-point stencil `(f(++) - f(+-) - f(-+) + f(--)) / (4 h^2)`.
pub fn cross_gamma<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], h: f64) -> Result<Vec<f64>, Error> {
    check_step(h)?;
    if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "x must be a non-empty finite vector".into(),
        ));
    }
    let n = x.len();
    let f0 = f(x);
    let mut out = vec![0.0; n * n];
    let at = |di: usize, si: f64, dj: usize, sj: f64| {
        let mut v = x.to_vec();
        v[di] += si * h;
        v[dj] += sj * h;
        f(&v)
    };
    for i in 0..n {
        let mut v = x.to_vec();
        v[i] = x[i] + h;
        let up = f(&v);
        v[i] = x[i] - h;
        let dn = f(&v);
        out[i * n + i] = (up - 2.0 * f0 + dn) / (h * h);
        for j in i + 1..n {
            let g = (at(i, 1.0, j, 1.0) - at(i, 1.0, j, -1.0) - at(i, -1.0, j, 1.0)
                + at(i, -1.0, j, -1.0))
                / (4.0 * h * h);
            out[i * n + j] = g;
            out[j * n + i] = g;
        }
    }
    Ok(out)
}

/// Node vols on a rectangular 2-D or 3-D grid (row-major vols).
#[derive(Clone, Debug, PartialEq)]
pub struct VolGrid {
    axes: Vec<Vec<f64>>,
    vols: Vec<f64>,
}

impl VolGrid {
    /// Validates as the Python `VolGrid`: 2 or 3 strictly increasing finite
    /// axes, vols finite and `>= 0`, `vols.len()` the product of the axis sizes.
    pub fn new(axes: Vec<Vec<f64>>, vols: Vec<f64>) -> Result<Self, Error> {
        if !(2..=3).contains(&axes.len()) {
            return Err(Error::InvalidInput(format!(
                "a vol grid has 2 or 3 axes, got {}",
                axes.len()
            )));
        }
        for (k, a) in axes.iter().enumerate() {
            if a.is_empty()
                || a.iter().any(|v| !v.is_finite())
                || a.windows(2).any(|w| w[1] <= w[0])
            {
                return Err(Error::InvalidInput(format!(
                    "axis {k} must be non-empty, finite and strictly increasing"
                )));
            }
        }
        let size: usize = axes.iter().map(Vec::len).product();
        if vols.len() != size {
            return Err(Error::InvalidInput(format!(
                "{} vols for a grid of {size} nodes",
                vols.len()
            )));
        }
        if vols.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::InvalidInput(
                "node vols must be finite and >= 0".into(),
            ));
        }
        Ok(Self { axes, vols })
    }

    /// Axis coordinates.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }

    /// Node vols, row-major.
    pub fn vols(&self) -> &[f64] {
        &self.vols
    }

    /// Multilinear interpolation weights over the nodes at `coords`
    /// (row-major, flat extrapolation, sum to one).
    pub fn weights(&self, coords: &[f64]) -> Result<Vec<f64>, Error> {
        if coords.len() != self.axes.len() {
            return Err(Error::InvalidInput(format!(
                "expected {} coordinates, got {}",
                self.axes.len(),
                coords.len()
            )));
        }
        let mut w = vec![1.0];
        for (a, &x) in self.axes.iter().zip(coords) {
            let mut wa = vec![0.0; a.len()];
            if a.len() == 1 || x <= a[0] {
                wa[0] = 1.0;
            } else if x >= a[a.len() - 1] {
                wa[a.len() - 1] = 1.0;
            } else {
                // searchsorted(side="right") - 1
                let i = a.partition_point(|&v| v <= x) - 1;
                let t = (x - a[i]) / (a[i + 1] - a[i]);
                wa[i] = 1.0 - t;
                wa[i + 1] = t;
            }
            w = w
                .iter()
                .flat_map(|&p| wa.iter().map(move |&q| p * q))
                .collect();
        }
        Ok(w)
    }

    /// Multilinear vol at `coords`.
    pub fn vol(&self, coords: &[f64]) -> Result<f64, Error> {
        Ok(self
            .weights(coords)?
            .iter()
            .zip(&self.vols)
            .map(|(w, v)| w * v)
            .sum())
    }

    /// The same grid with node `index` (row-major) moved by `h`.
    pub fn bumped(&self, index: usize, h: f64) -> Self {
        let mut g = self.clone();
        g.vols[index] += h;
        g
    }
}

/// Bucket vegas (value per unit vol), row-major like the grid.
///
/// Central difference with bump `h`; a node whose vol is below `h` is
/// bumped one-sided (forward difference) so the down bump never goes
/// negative, as in the Python `vega_grid`.
pub fn vega_grid<F: Fn(&VolGrid) -> f64>(pv: F, grid: &VolGrid, h: f64) -> Result<Vec<f64>, Error> {
    if !h.is_finite() || h <= 0.0 {
        return Err(Error::InvalidInput(format!("bump must be > 0, got {h}")));
    }
    let mut base: Option<f64> = None;
    let mut out = Vec::with_capacity(grid.vols.len());
    for idx in 0..grid.vols.len() {
        let up = pv(&grid.bumped(idx, h));
        if grid.vols[idx] >= h {
            out.push((up - pv(&grid.bumped(idx, -h))) / (2.0 * h));
        } else {
            let b = *base.get_or_insert_with(|| pv(grid));
            out.push((up - b) / h);
        }
    }
    Ok(out)
}

/// Per-factor second-order P&L `delta_i dx_i + 0.5 dx_i (S dx)_i`,
/// `S = (Gamma + Gamma^T) / 2`; sums to `delta . dx + 0.5 dx^T Gamma dx`.
pub fn taylor_attribution(
    delta: &[f64],
    moves: &[f64],
    gamma: Option<&[f64]>,
) -> Result<Vec<f64>, Error> {
    let n = delta.len();
    if n == 0 || moves.len() != n {
        return Err(Error::InvalidInput(
            "delta and move must be 1-D vectors of equal length".into(),
        ));
    }
    let mut out: Vec<f64> = delta.iter().zip(moves).map(|(d, m)| d * m).collect();
    if let Some(g) = gamma {
        if g.len() != n * n {
            return Err(Error::InvalidInput(format!("gamma must be {n}x{n}")));
        }
        for i in 0..n {
            let s: f64 = (0..n)
                .map(|j| 0.5 * (g[i * n + j] + g[j * n + i]) * moves[j])
                .sum();
            out[i] += 0.5 * moves[i] * s;
        }
    }
    Ok(out)
}

/// One period's P&L explain (all values in the reporting currency).
#[derive(Clone, Debug, PartialEq)]
pub struct PnlExplain {
    /// Hypothetical P&L (full revaluation, positions held fixed).
    pub hpl: f64,
    /// Time effects (carry, roll-down, decay), in the caller's order.
    pub time: Vec<f64>,
    /// Risk-based P&L per factor ([`taylor_attribution`]).
    pub risk: Vec<f64>,
    /// P&L of trades booked in the period (kept out of HPL, FRTB MAR32).
    pub new_trades: f64,
}

impl PnlExplain {
    /// Total time effect.
    pub fn theta(&self) -> f64 {
        self.time.iter().sum()
    }
    /// Risk-theoretical P&L: time effects plus the factor attribution.
    pub fn rtpl(&self) -> f64 {
        self.theta() + self.risk.iter().sum::<f64>()
    }
    /// `HPL - RTPL`.
    pub fn unexplained(&self) -> f64 {
        self.hpl - self.rtpl()
    }
    /// HPL plus new-trade P&L.
    pub fn total(&self) -> f64 {
        self.hpl + self.new_trades
    }
}

/// Assembles an explain; refuses non-finite inputs (NaN policy: refuse).
pub fn explain(
    hpl: f64,
    time: &[f64],
    delta: &[f64],
    moves: &[f64],
    gamma: Option<&[f64]>,
    new_trades: f64,
) -> Result<PnlExplain, Error> {
    let risk = taylor_attribution(delta, moves, gamma)?;
    let finite = hpl.is_finite()
        && new_trades.is_finite()
        && time.iter().all(|v| v.is_finite())
        && risk.iter().all(|v| v.is_finite());
    if !finite {
        return Err(Error::InvalidInput(
            "explain inputs must be finite (NaN policy: refuse)".into(),
        ));
    }
    Ok(PnlExplain {
        hpl,
        time: time.to_vec(),
        risk,
        new_trades,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singular_jacobian_is_refused() {
        let e = node_to_quote_risk(&[1.0, 2.0, 2.0, 4.0], 2, &[1.0, 1.0]).unwrap_err();
        assert!(e.to_string().contains("singular"));
    }

    #[test]
    fn round_trip_quote_node() {
        let jac = [1.02, 0.0, 0.1, 0.0, 2.0, 0.3, 0.05, 0.2, 0.9];
        let gz = [1.0, -2.0, 3.5];
        let gq = node_to_quote_risk(&jac, 3, &gz).unwrap();
        let back = quote_to_node_risk(&jac, 3, &gq).unwrap();
        for (a, b) in back.iter().zip(&gz) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn fd_kernels_match_analytic_derivatives() {
        // f(x, y) = exp(x) sin(y) + x^2 y: grad and Hessian known in closed form.
        let x = [0.3, 0.7];
        let f = |v: &[f64]| v[0].exp() * v[1].sin() + v[0] * v[0] * v[1];
        let jac = fd_jacobian(|v| vec![f(v)], &x, 1e-6).unwrap();
        let g = [
            x[0].exp() * x[1].sin() + 2.0 * x[0] * x[1],
            x[0].exp() * x[1].cos() + x[0] * x[0],
        ];
        for (a, b) in jac.iter().zip(&g) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
        let h = cross_gamma(f, &x, 1e-4).unwrap();
        let want = [
            x[0].exp() * x[1].sin() + 2.0 * x[1],
            x[0].exp() * x[1].cos() + 2.0 * x[0],
            x[0].exp() * x[1].cos() + 2.0 * x[0],
            -x[0].exp() * x[1].sin(),
        ];
        for (a, b) in h.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn weights_are_a_partition_of_unity_and_vega_is_one_sided_near_zero() {
        let g = VolGrid::new(
            vec![vec![1.0, 2.0], vec![0.0, 1.0, 2.0]],
            vec![0.0, 0.2, 0.3, 0.25, 0.1, 0.4],
        )
        .unwrap();
        let w = g.weights(&[1.5, 0.25]).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        // Linear PV: vega equals the weights exactly, including the zero-vol node.
        let v = vega_grid(|gr| gr.vol(&[1.5, 0.25]).unwrap(), &g, 1e-4).unwrap();
        for (a, b) in v.iter().zip(&w) {
            assert!((a - b).abs() < 1e-10);
        }
        assert!(VolGrid::new(vec![vec![1.0]], vec![0.1]).is_err());
        assert!(VolGrid::new(vec![vec![1.0, 1.0], vec![0.0]], vec![0.1, 0.1]).is_err());
    }

    #[test]
    fn explain_refuses_nan() {
        assert!(explain(f64::NAN, &[], &[1.0], &[1.0], None, 0.0).is_err());
        let ex = explain(10.0, &[-1.0], &[100.0], &[0.1], Some(&[40.0]), 0.0).unwrap();
        assert!((ex.rtpl() - 9.2).abs() < 1e-12);
    }
}
