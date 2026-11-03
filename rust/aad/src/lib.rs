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
    pub const fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
            adj: RefCell::new(Vec::new()),
        }
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

    /// Forget every node but keep the allocation, so a tape reused across
    /// calls records without allocating (see [`bs::greeks_on`]).
    pub fn clear(&self) {
        self.rewind(0);
    }

    #[inline]
    fn push(&self, val: f64, a: u32, da: f64, b: u32, db: f64) -> Var<'_> {
        let mut n = self.nodes.borrow_mut();
        let node = Node { a, b, da, db };
        n.push(node);
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

    /// `N` new independent inputs recorded with one borrow of the tape.
    pub fn vars<const N: usize>(&self, vals: [f64; N]) -> [Var<'_>; N] {
        let mut n = self.nodes.borrow_mut();
        let first = n.len();
        n.extend(std::iter::repeat_n(
            Node {
                a: NONE,
                b: NONE,
                da: 0.0,
                db: 0.0,
            },
            N,
        ));
        let mut i = 0;
        vals.map(|val| {
            i += 1;
            Var {
                tape: self,
                idx: (first + i - 1) as u32,
                val,
            }
        })
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

    /// Zero every adjoint, then sweep from `output` (seed 1) down to node
    /// `stop`: [`Tape::zero_adjoints`] and [`Tape::backward`] in one borrow.
    pub fn reverse(&self, output: Var<'_>, stop: usize) {
        assert!(
            std::ptr::eq(output.tape, self),
            "output was recorded on a different tape"
        );
        let nodes = self.nodes.borrow();
        let mut adj = self.adj.borrow_mut();
        adj.clear();
        adj.resize(nodes.len(), 0.0);
        adj[output.idx as usize] = 1.0;
        Self::sweep(&nodes, &mut adj, output.idx as usize + 1, stop);
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
        Self::sweep(
            &self.nodes.borrow(),
            &mut self.adj.borrow_mut(),
            start,
            stop,
        );
    }

    #[inline]
    fn sweep(nodes: &[Node], adj: &mut Vec<f64>, start: usize, stop: usize) {
        if adj.len() < nodes.len() {
            adj.resize(nodes.len(), 0.0);
        }
        let adj = &mut adj[..nodes.len()];
        for (i, n) in nodes[stop..start].iter().enumerate().rev() {
            let w = adj[stop + i];
            // Inputs (`a == NONE`) have no parents; a node with one parent
            // carries `b == NONE`. A zero adjoint is skipped so that a node
            // off the output's path never turns an infinite partial into NaN.
            if w != 0.0 && n.a != NONE {
                adj[n.a as usize] += w * n.da;
                if n.b != NONE {
                    adj[n.b as usize] += w * n.db;
                }
            }
        }
    }

    /// The accumulated adjoint of `v`.
    #[inline]
    pub fn adjoint(&self, v: Var<'_>) -> f64 {
        self.adj
            .borrow()
            .get(v.idx as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// The accumulated adjoints of `N` variables, read with one borrow.
    pub fn adjoints<const N: usize>(&self, vs: [Var<'_>; N]) -> [f64; N] {
        let adj = self.adj.borrow();
        vs.map(|v| adj.get(v.idx as usize).copied().unwrap_or(0.0))
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

    // Fused operations. Each default is the plain composition, so `f64`
    // computes exactly the same bits as the unfused expression; [`Var`]
    // records each as one node with its local partials preaccumulated
    // (statement-level preaccumulation: Griewank and Walther 2008, s.10.2;
    // Savine 2018, ch. 15). Fewer nodes means a shorter reverse sweep.

    /// `exp(-(self * o))`.
    fn exp_neg_mul(self, o: Self) -> Self {
        (-(self * o)).exp()
    }
    /// `ln(self / o)`.
    fn ln_div(self, o: Self) -> Self {
        (self / o).ln()
    }
    /// `ncdf(self * c)` for a constant `c`.
    fn ncdf_scaled(self, c: f64) -> Self {
        (self * c).ncdf()
    }
    /// `self * self * 0.5`.
    fn half_square(self) -> Self {
        self * self * 0.5
    }
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
    fn exp_neg_mul(self, o: Self) -> Self {
        let e = (-(self.val * o.val)).exp();
        self.binary(o, e, -o.val * e, -self.val * e)
    }
    fn ln_div(self, o: Self) -> Self {
        self.binary(o, (self.val / o.val).ln(), 1.0 / self.val, -1.0 / o.val)
    }
    fn ncdf_scaled(self, c: f64) -> Self {
        let v = self.val * c;
        self.unary(
            qp_core::special::norm_cdf(v),
            c * INV_SQRT_2PI * (-0.5 * v * v).exp(),
        )
    }
    fn half_square(self) -> Self {
        self.unary(self.val * self.val * 0.5, self.val)
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
        let inv = 1.0 / o.val;
        self.binary(o, q, inv, -q * inv)
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
        // `x * 1` is `x` bit for bit: record nothing (call payoffs, phi = 1).
        if c == 1.0 {
            return self;
        }
        self.unary(self.val * c, c)
    }
}
impl<'t> Div<f64> for Var<'t> {
    type Output = Self;
    fn div(self, c: f64) -> Self {
        self.unary(self.val / c, 1.0 / c)
    }
}
