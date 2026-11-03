//! Tape-based reverse-mode algorithmic differentiation (plan W19).
//!
//! Plain idea: every arithmetic step of a pricer is recorded on a tape with
//! its local partial derivatives; one backward walk over the tape gives the
//! derivative of the price with respect to every input at once. Formal
//! statement: for nodes `v_i = phi_i(v_j)` the adjoints
//! `vbar_j = sum_i vbar_i d phi_i / d v_j`, seeded with `ybar = 1` and
//! evaluated in reverse order (Griewank and Walther 2008, *Evaluating
//! Derivatives*, ch. 3; Savine 2018, *Modern Computational Finance*, ch. 9).
//!
//! The Rust port of `qp_core.aad`: the same tape layout (at most two parents
//! per node), the same `mark`/`rewind` for per-path Monte Carlo sweeps, and
//! the same three pricers ([`bs`], [`mc`], [`swap`]). Pricers are generic
//! over [`Real`], implemented by `f64` (the plain price, the denominator of
//! the cost ratio) and by [`Var`] (the recorded price).
//!
//! ```
//! use qp_aad::{Real, Tape};
//! let tape = Tape::new();
//! let (x0, x1) = (tape.var(1.0), tape.var(2.0));
//! let y = x0 * x1 + x0.exp();
//! let g = tape.gradient(y, &[x0, x1]);
//! assert!((g[0] - (2.0 + 1f64.exp())).abs() < 1e-15 && g[1] == 1.0);
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub mod bs;
pub mod linalg;
pub mod mc;
pub mod swap;

const NONE: u32 = u32::MAX;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[derive(Clone, Copy)]
struct Node {
    a: u32,
    b: u32,
    da: f64,
    db: f64,
}

/// A recording of one computation.
#[derive(Default)]
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
    adj: RefCell<Vec<f64>>,
}

/// A value recorded on a [`Tape`].
#[derive(Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    idx: u32,
    val: f64,
}

impl std::fmt::Debug for Var<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Var({})", self.val)
    }
}

impl Tape {
    /// An empty tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty tape with room for `n` nodes (avoids regrowth on small pricers).
    pub fn with_capacity(n: usize) -> Self {
        Self {
            nodes: RefCell::new(Vec::with_capacity(n)),
            adj: RefCell::new(Vec::with_capacity(n)),
        }
    }

    /// Number of recorded nodes.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// True when nothing is recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push(&self, val: f64, a: u32, da: f64, b: u32, db: f64) -> Var<'_> {
        let mut n = self.nodes.borrow_mut();
        n.push(Node { a, b, da, db });
        Var {
            tape: self,
            idx: (n.len() - 1) as u32,
            val,
        }
    }

    /// A new independent input.
    pub fn var(&self, val: f64) -> Var<'_> {
        self.push(val, NONE, 0.0, NONE, 0.0)
    }

    /// Position to come back to with [`Tape::rewind`].
    pub fn mark(&self) -> usize {
        self.len()
    }

    /// Forget every node recorded after `mark`; adjoints before it stay.
    pub fn rewind(&self, mark: usize) {
        self.nodes.borrow_mut().truncate(mark);
        self.adj.borrow_mut().truncate(mark);
    }

    /// Set every adjoint to zero.
    pub fn zero_adjoints(&self) {
        let n = self.len();
        let mut adj = self.adj.borrow_mut();
        adj.clear();
        adj.resize(n, 0.0);
    }

    /// Add `seed * d output / d node` into every node's adjoint, sweeping
    /// from `output` down to node `stop` (nodes below `stop` only receive).
    pub fn backward(&self, output: Var<'_>, seed: f64, stop: usize) {
        assert!(
            std::ptr::eq(output.tape, self),
            "output was recorded on a different tape"
        );
        {
            let n = self.len();
            let mut adj = self.adj.borrow_mut();
            if adj.len() < n {
                adj.resize(n, 0.0);
            }
            adj[output.idx as usize] += seed;
        }
        self.propagate(output.idx as usize + 1, stop);
    }

    /// Push the adjoints of nodes `start-1` down to `stop` into their parents.
    pub fn propagate(&self, start: usize, stop: usize) {
        let nodes = self.nodes.borrow();
        let mut adj = self.adj.borrow_mut();
        if adj.len() < nodes.len() {
            adj.resize(nodes.len(), 0.0);
        }
        for i in (stop..start).rev() {
            let w = adj[i];
            if w == 0.0 {
                continue;
            }
            let n = nodes[i];
            if n.a != NONE {
                adj[n.a as usize] += w * n.da;
                if n.b != NONE {
                    adj[n.b as usize] += w * n.db;
                }
            }
        }
    }

    /// The accumulated adjoint of `v`.
    pub fn adjoint(&self, v: Var<'_>) -> f64 {
        self.adj
            .borrow()
            .get(v.idx as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// `[d output / d x for x in inputs]` from one reverse sweep.
    pub fn gradient(&self, output: Var<'_>, inputs: &[Var<'_>]) -> Vec<f64> {
        self.zero_adjoints();
        self.backward(output, 1.0, 0);
        inputs.iter().map(|&x| self.adjoint(x)).collect()
    }
}

/// Numbers a generic pricer can run on: `f64` or a tape [`Var`].
pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// `e^x`.
    fn exp(self) -> Self;
    /// Natural log.
    fn ln(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Standard normal CDF, `0.5 erfc(-x / sqrt 2)` (Cody erfc of `qp_core`).
    fn ncdf(self) -> Self;
    /// `c - self`.
    fn rsub(self, c: f64) -> Self;
    /// `c / self`.
    fn rdiv(self, c: f64) -> Self;
    /// The value.
    fn val(self) -> f64;
}

impl Real for f64 {
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn ncdf(self) -> f64 {
        qp_core::special::norm_cdf(self)
    }
    fn rsub(self, c: f64) -> f64 {
        c - self
    }
    fn rdiv(self, c: f64) -> f64 {
        c / self
    }
    fn val(self) -> f64 {
        self
    }
}

impl<'t> Var<'t> {
    #[inline]
    fn unary(self, val: f64, d: f64) -> Self {
        self.tape.push(val, self.idx, d, NONE, 0.0)
    }
    #[inline]
    fn binary(self, o: Self, val: f64, da: f64, db: f64) -> Self {
        debug_assert!(
            std::ptr::eq(self.tape, o.tape),
            "operands on different tapes"
        );
        self.tape.push(val, self.idx, da, o.idx, db)
    }
    /// The value.
    pub fn value(self) -> f64 {
        self.val
    }
}

impl<'t> Real for Var<'t> {
    fn exp(self) -> Self {
        let e = self.val.exp();
        self.unary(e, e)
    }
    fn ln(self) -> Self {
        self.unary(self.val.ln(), 1.0 / self.val)
    }
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        self.unary(s, 0.5 / s)
    }
    fn ncdf(self) -> Self {
        let v = self.val;
        self.unary(
            qp_core::special::norm_cdf(v),
            INV_SQRT_2PI * (-0.5 * v * v).exp(),
        )
    }
    fn rsub(self, c: f64) -> Self {
        self.unary(c - self.val, -1.0)
    }
    fn rdiv(self, c: f64) -> Self {
        let q = c / self.val;
        self.unary(q, -q / self.val)
    }
    fn val(self) -> f64 {
        self.val
    }
}

impl<'t> Add for Var<'t> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        self.binary(o, self.val + o.val, 1.0, 1.0)
    }
}
impl<'t> Sub for Var<'t> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self.binary(o, self.val - o.val, 1.0, -1.0)
    }
}
impl<'t> Mul for Var<'t> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        self.binary(o, self.val * o.val, o.val, self.val)
    }
}
impl<'t> Div for Var<'t> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let q = self.val / o.val;
        self.binary(o, q, 1.0 / o.val, -q / o.val)
    }
}
impl<'t> Neg for Var<'t> {
    type Output = Self;
    fn neg(self) -> Self {
        self.unary(-self.val, -1.0)
    }
}
impl<'t> Add<f64> for Var<'t> {
    type Output = Self;
    fn add(self, c: f64) -> Self {
        self.unary(self.val + c, 1.0)
    }
}
impl<'t> Sub<f64> for Var<'t> {
    type Output = Self;
    fn sub(self, c: f64) -> Self {
        self.unary(self.val - c, 1.0)
    }
}
impl<'t> Mul<f64> for Var<'t> {
    type Output = Self;
    fn mul(self, c: f64) -> Self {
        self.unary(self.val * c, c)
    }
}
impl<'t> Div<f64> for Var<'t> {
    type Output = Self;
    fn div(self, c: f64) -> Self {
        self.unary(self.val / c, 1.0 / c)
    }
}
