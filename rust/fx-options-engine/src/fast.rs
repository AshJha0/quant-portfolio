//! Low-latency kernels: non-allocating, non-panicking, batch-capable.
//!
//! Every function here returns the same bits as its counterpart in the
//! classic API ([`crate::gk_price`], [`crate::analytic_greeks`],
//! [`crate::implied_vol()`], [`crate::binomial_price`], [`crate::mc_price`]).
//! The test suite checks this with `f64::to_bits` on varied inputs.
//!
//! The differences are in the contract, not the numbers:
//!
//! * Errors are a small `Copy` enum, [`KernelError`]. No `String` is built,
//!   on the success path or the error path.
//! * Kernels that need memory take it from a caller-owned workspace
//!   ([`BinomialWorkspace`], [`McWorkspace`]). After the first call at a
//!   given size they do not touch the heap.
//! * Struct-of-arrays batch pricers ([`gk_price_batch`],
//!   [`greeks_batch`]) run a branch-free validation pass, then one scalar
//!   kernel per element.
//! * [`mc_price_par`] fills the path buffers on `n_threads` scoped threads
//!   with a fixed partition. Each chunk starts its own copy of the RNG at
//!   the chunk's offset in the single stream, so the draws are the ones
//!   [`crate::mc_price`] would make. The reductions stay sequential, so the
//!   result is the same bits for any thread count.
//!
//! ```
//! use fx_options_engine::fast::{gk_price_k, KernelError};
//! use fx_options_engine::{gk_price, OptionType};
//! let a = gk_price_k(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925, OptionType::Call).unwrap();
//! let b = gk_price(1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925, OptionType::Call).unwrap();
//! assert_eq!(a.to_bits(), b.to_bits());
//! assert_eq!(gk_price_k(-1.0, 1.0, 1.0, 0.0, 0.0, 0.1, OptionType::Call),
//!            Err(KernelError::NonPositiveSpot));
//! ```

use std::fmt;

use crate::garman_kohlhagen::MIN_VOL;
use crate::iv_core::black_implied_vol;
use crate::rng::Xoshiro256StarStar;
use crate::{Exercise, GreeksResult, McResult, OptionType};
use qp_core::black_scholes::{TruncatedC1Normal, carry_d1_d2, carry_greeks, carry_value};

/// Error code of the low-latency kernels. `Copy`, one byte, no heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelError {
    /// An input is NaN or infinite.
    NonFinite,
    /// Spot (or forward) is not positive.
    NonPositiveSpot,
    /// Strike is not positive.
    NonPositiveStrike,
    /// Time to expiry is negative.
    NegativeTime,
    /// Volatility is negative.
    NegativeVol,
    /// The kernel needs `T > 0`.
    ZeroTime,
    /// The kernel needs `sigma > 0` and `sigma * sqrt(T)` above `1e-12`.
    ZeroVol,
    /// A finite input produced a non-finite result.
    Overflow,
    /// `n_paths` is outside `[2, MAX_PATHS]` or gives too few samples.
    PathCount,
    /// `steps` is outside `[1, MAX_STEPS]`.
    Steps,
    /// The tree probability is outside `[0, 1]`.
    TreeProbability,
    /// A price is outside the no-arbitrage bounds.
    ArbitrageBounds,
    /// A price sits at the `sigma -> inf` bound: no finite implied vol.
    NotIdentifiable,
    /// A root finder did not converge.
    NoConvergence,
    /// Batch input or output slices have different lengths.
    LengthMismatch,
    /// `n_threads` is zero.
    Threads,
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for KernelError {}

/// Error of a batch call: the first failing element and its error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchError {
    /// Index of the first element that failed (0 for a length mismatch).
    pub index: usize,
    /// What went wrong.
    pub error: KernelError,
}

/// Result alias for the kernels.
pub type KResult<T> = Result<T, KernelError>;

/// Same checks, in the same order, as the classic `validate_inputs`.
#[inline]
fn validate(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64) -> KResult<()> {
    if !(s.is_finite()
        && k.is_finite()
        && t.is_finite()
        && r_d.is_finite()
        && r_f.is_finite()
        && sigma.is_finite())
    {
        return Err(KernelError::NonFinite);
    }
    if s <= 0.0 {
        return Err(KernelError::NonPositiveSpot);
    }
    if k <= 0.0 {
        return Err(KernelError::NonPositiveStrike);
    }
    if t < 0.0 {
        return Err(KernelError::NegativeTime);
    }
    if sigma < 0.0 {
        return Err(KernelError::NegativeVol);
    }
    Ok(())
}

#[inline]
fn finite(v: f64) -> KResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(KernelError::Overflow)
    }
}

/// GK price on validated inputs. Same operations as `gk_price`.
#[inline(always)]
fn gk_core(s: f64, k: f64, t: f64, r_d: f64, r_f: f64, sigma: f64, phi: f64) -> f64 {
    if t == 0.0 {
        return (phi * (s - k)).max(0.0);
    }
    let disc_s = s * (-r_f * t).exp();
    let disc_k = k * (-r_d * t).exp();
    let vol_sqrt_t = sigma * t.sqrt();
    if vol_sqrt_t <= MIN_VOL {
        (phi * (disc_s - disc_k)).max(0.0)
    } else {
        let (d1v, d2v) = carry_d1_d2(s, k, t, r_d, r_f, sigma);
        carry_value::<TruncatedC1Normal>(disc_s, disc_k, d1v, d2v, phi)
    }
}

/// Garman-Kohlhagen price. Same bits as [`crate::gk_price`].
///
/// # Errors
///
/// [`KernelError`] on invalid inputs or an overflowing result.
#[inline]
pub fn gk_price_k(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
) -> KResult<f64> {
    validate(s, k, t, r_d, r_f, sigma)?;
    finite(gk_core(s, k, t, r_d, r_f, sigma, option_type.phi()))
}

/// Analytic Greeks on validated inputs with `t > 0`, `sigma > 0`.
#[inline(always)]
fn greeks_core(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    phi: f64,
) -> KResult<GreeksResult> {
    let vol_sqrt_t = sigma * t.sqrt();
    if vol_sqrt_t <= MIN_VOL {
        return Err(KernelError::ZeroVol);
    }
    // Shared carry-form kernel, as in [`crate::analytic_greeks`].
    Ok(crate::greeks::from_carry(
        &carry_greeks::<TruncatedC1Normal>(s, k, t, r_d, r_f, sigma, phi),
    ))
}

/// Analytic GK Greeks. Same bits as [`crate::analytic_greeks`].
///
/// # Errors
///
/// [`KernelError`] on invalid inputs, `t = 0` ([`KernelError::ZeroTime`])
/// or `sigma = 0` ([`KernelError::ZeroVol`]).
#[inline]
pub fn greeks_k(
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
) -> KResult<GreeksResult> {
    validate(s, k, t, r_d, r_f, sigma)?;
    if t <= 0.0 {
        return Err(KernelError::ZeroTime);
    }
    if sigma <= 0.0 {
        return Err(KernelError::ZeroVol);
    }
    greeks_core(s, k, t, r_d, r_f, sigma, option_type.phi())
}

/// Implied volatility from a GK premium. Same bits as
/// [`crate::implied_vol()`].
///
/// # Errors
///
/// [`KernelError`] on invalid inputs, `t = 0`, a price outside the
/// no-arbitrage bounds, or no convergence.
pub fn implied_vol_k(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    option_type: OptionType,
) -> KResult<f64> {
    let phi = option_type.phi();
    validate(s, k, t, r_d, r_f, 0.0)?;
    if t <= 0.0 {
        return Err(KernelError::ZeroTime);
    }
    if !price.is_finite() {
        return Err(KernelError::NonFinite);
    }
    let df_d = finite((-r_d * t).exp())?;
    let df_f = finite((-r_f * t).exp())?;
    let forward = finite(s * df_f / df_d)?;
    let lower = df_d * (phi * (forward - k)).max(0.0);
    let upper = if phi > 0.0 { s * df_f } else { k * df_d };
    let tau_lo = if lower > 0.0 {
        4.0 * f64::EPSILON * df_d * forward.max(k)
    } else {
        0.0
    };
    let tau = 4.0 * f64::EPSILON * upper;
    if price < lower - tau_lo || price > upper + tau {
        return Err(KernelError::ArbitrageBounds);
    }
    if price - lower <= tau_lo {
        return Ok(0.0);
    }
    if upper - price <= tau {
        return Err(KernelError::NotIdentifiable);
    }
    let mut evals = 0;
    black_implied_vol(price, forward, k, t, df_d, phi > 0.0, &mut evals).map_err(|msg| {
        if msg.starts_with("price") {
            KernelError::ArbitrageBounds
        } else {
            KernelError::NoConvergence
        }
    })
}

// ---------------------------------------------------------------------------
// Struct-of-arrays batch pricing
// ---------------------------------------------------------------------------

/// Struct-of-arrays inputs for the batch pricers. All slices must have the
/// same length.
#[derive(Debug, Clone, Copy)]
pub struct GkBatch<'a> {
    /// Spots.
    pub s: &'a [f64],
    /// Strikes.
    pub k: &'a [f64],
    /// Times to expiry in years.
    pub t: &'a [f64],
    /// Domestic rates.
    pub r_d: &'a [f64],
    /// Foreign rates.
    pub r_f: &'a [f64],
    /// Volatilities.
    pub sigma: &'a [f64],
    /// Call or put per element.
    pub option_type: &'a [OptionType],
}

impl GkBatch<'_> {
    /// Common length, or [`KernelError::LengthMismatch`].
    ///
    /// # Errors
    ///
    /// [`KernelError::LengthMismatch`] when the slices differ in length.
    pub fn len(&self) -> KResult<usize> {
        let n = self.s.len();
        if self.k.len() == n
            && self.t.len() == n
            && self.r_d.len() == n
            && self.r_f.len() == n
            && self.sigma.len() == n
            && self.option_type.len() == n
        {
            Ok(n)
        } else {
            Err(KernelError::LengthMismatch)
        }
    }

    /// True when the batch has no elements (or the lengths differ).
    pub fn is_empty(&self) -> bool {
        self.len().map_or(true, |n| n == 0)
    }
}

/// Branch-free scan: true when every element passes the price-kernel
/// checks. Written as a flag fold over equal-length slices so the
/// compiler can vectorise it.
#[inline]
fn all_valid(b: &GkBatch<'_>, n: usize, need_positive: bool) -> bool {
    let (s, k, t) = (&b.s[..n], &b.k[..n], &b.t[..n]);
    let (rd, rf, sg) = (&b.r_d[..n], &b.r_f[..n], &b.sigma[..n]);
    let mut ok = true;
    for i in 0..n {
        let fin = s[i].is_finite()
            & k[i].is_finite()
            & t[i].is_finite()
            & rd[i].is_finite()
            & rf[i].is_finite()
            & sg[i].is_finite();
        let pos = (s[i] > 0.0) & (k[i] > 0.0) & (t[i] >= 0.0) & (sg[i] >= 0.0);
        let strict = !need_positive | ((t[i] > 0.0) & (sg[i] > 0.0));
        ok &= fin & pos & strict;
    }
    ok
}

/// Batch GK prices: `out[i] = gk_price(b[i])`, same bits as the scalar call.
///
/// # Errors
///
/// [`BatchError`] naming the first failing element. On error `out` may be
/// partly written.
pub fn gk_price_batch(b: &GkBatch<'_>, out: &mut [f64]) -> Result<(), BatchError> {
    let n = b.len().map_err(|error| BatchError { index: 0, error })?;
    if out.len() != n {
        return Err(BatchError {
            index: 0,
            error: KernelError::LengthMismatch,
        });
    }
    if !all_valid(b, n, false) {
        for i in 0..n {
            validate(b.s[i], b.k[i], b.t[i], b.r_d[i], b.r_f[i], b.sigma[i])
                .map_err(|error| BatchError { index: i, error })?;
        }
    }
    let (s, k, t) = (&b.s[..n], &b.k[..n], &b.t[..n]);
    let (rd, rf, sg, ty) = (&b.r_d[..n], &b.r_f[..n], &b.sigma[..n], &b.option_type[..n]);
    let out = &mut out[..n];
    let mut bad = usize::MAX;
    for i in 0..n {
        let v = gk_core(s[i], k[i], t[i], rd[i], rf[i], sg[i], ty[i].phi());
        out[i] = v;
        if !v.is_finite() && bad == usize::MAX {
            bad = i;
        }
    }
    if bad != usize::MAX {
        return Err(BatchError {
            index: bad,
            error: KernelError::Overflow,
        });
    }
    Ok(())
}

/// Struct-of-arrays output of [`greeks_batch`]. All slices must have the
/// batch length.
#[derive(Debug)]
pub struct GreeksBatchOut<'a> {
    /// Premiums.
    pub price: &'a mut [f64],
    /// Spot deltas.
    pub delta_spot: &'a mut [f64],
    /// Forward deltas.
    pub delta_forward: &'a mut [f64],
    /// Gammas.
    pub gamma: &'a mut [f64],
    /// Vegas.
    pub vega: &'a mut [f64],
    /// Thetas per year.
    pub theta: &'a mut [f64],
    /// Domestic rhos.
    pub rho_domestic: &'a mut [f64],
    /// Foreign rhos.
    pub rho_foreign: &'a mut [f64],
    /// Vannas.
    pub vanna: &'a mut [f64],
    /// Volgas.
    pub volga: &'a mut [f64],
}

impl GreeksBatchOut<'_> {
    fn all_len(&self, n: usize) -> bool {
        [
            self.price.len(),
            self.delta_spot.len(),
            self.delta_forward.len(),
            self.gamma.len(),
            self.vega.len(),
            self.theta.len(),
            self.rho_domestic.len(),
            self.rho_foreign.len(),
            self.vanna.len(),
            self.volga.len(),
        ]
        .iter()
        .all(|&l| l == n)
    }
}

/// Batch analytic Greeks, same bits as [`crate::analytic_greeks`] per
/// element.
///
/// # Errors
///
/// [`BatchError`] naming the first failing element. On error `out` may be
/// partly written.
pub fn greeks_batch(b: &GkBatch<'_>, out: &mut GreeksBatchOut<'_>) -> Result<(), BatchError> {
    let n = b.len().map_err(|error| BatchError { index: 0, error })?;
    if !out.all_len(n) {
        return Err(BatchError {
            index: 0,
            error: KernelError::LengthMismatch,
        });
    }
    if !all_valid(b, n, true) {
        for i in 0..n {
            validate(b.s[i], b.k[i], b.t[i], b.r_d[i], b.r_f[i], b.sigma[i])
                .and_then(|()| {
                    if b.t[i] <= 0.0 {
                        Err(KernelError::ZeroTime)
                    } else if b.sigma[i] <= 0.0 {
                        Err(KernelError::ZeroVol)
                    } else {
                        Ok(())
                    }
                })
                .map_err(|error| BatchError { index: i, error })?;
        }
    }
    let (s, k, t) = (&b.s[..n], &b.k[..n], &b.t[..n]);
    let (rd, rf, sg, ty) = (&b.r_d[..n], &b.r_f[..n], &b.sigma[..n], &b.option_type[..n]);
    for i in 0..n {
        let g = greeks_core(s[i], k[i], t[i], rd[i], rf[i], sg[i], ty[i].phi())
            .map_err(|error| BatchError { index: i, error })?;
        out.price[i] = g.price;
        out.delta_spot[i] = g.delta_spot;
        out.delta_forward[i] = g.delta_forward;
        out.gamma[i] = g.gamma;
        out.vega[i] = g.vega;
        out.theta[i] = g.theta;
        out.rho_domestic[i] = g.rho_domestic;
        out.rho_foreign[i] = g.rho_foreign;
        out.vanna[i] = g.vanna;
        out.volga[i] = g.volga;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CRR tree with a reusable workspace
// ---------------------------------------------------------------------------

/// Reusable buffers for [`binomial_price_ws`]. Grows to the largest
/// `steps` seen and never shrinks.
#[derive(Debug, Default, Clone)]
pub struct BinomialWorkspace {
    ex_even: Vec<f64>,
    ex_odd: Vec<f64>,
    cur: Vec<f64>,
    next: Vec<f64>,
}

impl BinomialWorkspace {
    /// Empty workspace. The first call allocates.
    pub fn new() -> Self {
        Self::default()
    }

    /// Workspace sized for trees up to `steps` steps: later calls with
    /// `steps` at or below this do not allocate.
    pub fn with_steps(steps: u32) -> Self {
        let n = steps as usize;
        Self {
            ex_even: Vec::with_capacity(n + 1),
            ex_odd: Vec::with_capacity(n),
            cur: Vec::with_capacity(n + 1),
            next: Vec::with_capacity(n + 1),
        }
    }
}

/// CRR binomial price using `ws` for its buffers. Same bits as
/// [`crate::binomial_price`].
///
/// # Errors
///
/// [`KernelError`] on invalid inputs, `steps` out of range, a tree
/// probability outside `[0, 1]`, or an overflowing result.
pub fn binomial_price_ws(
    ws: &mut BinomialWorkspace,
    s: f64,
    k: f64,
    t: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    option_type: OptionType,
    steps: u32,
    exercise: Exercise,
) -> KResult<f64> {
    let phi = option_type.phi();
    validate(s, k, t, r_d, r_f, sigma)?;
    if !(1..=crate::binomial::MAX_STEPS).contains(&steps) {
        return Err(KernelError::Steps);
    }
    if t == 0.0 {
        return Ok((phi * (s - k)).max(0.0));
    }
    if sigma == 0.0 {
        if exercise == Exercise::European {
            return finite(gk_core(s, k, t, r_d, r_f, sigma, phi));
        }
        let mut best: f64 = 0.0;
        for i in 0..=steps {
            let ti = t * f64::from(i) / f64::from(steps);
            let value = (phi * (s * (-r_f * ti).exp() - k * (-r_d * ti).exp())).max(0.0);
            best = best.max(value);
        }
        return finite(best);
    }
    let n = steps as usize;
    let dt = t / f64::from(steps);
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = ((r_d - r_f) * dt).exp();
    let p = (growth - d) / (u - d);
    if !(0.0..=1.0).contains(&p) {
        return Err(KernelError::TreeProbability);
    }
    let disc = (-r_d * dt).exp();
    let q = 1.0 - p;

    // Node payoffs phi (S u^m - K), m = idx - n, split by index parity so
    // that step i reads a contiguous run: node j of step i sits at
    // idx = (n - i) + 2j, whose parity is fixed for the whole step.
    let ex_at = |idx: usize| phi * (s * u.powf(idx as f64 - n as f64) - k);
    ws.ex_even.clear();
    ws.ex_even.extend((0..=n).map(|h| ex_at(2 * h)));
    ws.ex_odd.clear();
    ws.ex_odd.extend((0..n).map(|h| ex_at(2 * h + 1)));
    ws.cur.clear();
    ws.cur.extend(ws.ex_even.iter().map(|e| e.max(0.0)));
    ws.next.clear();
    ws.next.resize(n + 1, 0.0);

    // Double buffer: next[j] = f(cur[j], cur[j + 1]). Same arithmetic as
    // the classic in-place loop, but with no loop-carried dependency, so
    // the loop body can vectorise.
    let american = exercise == Exercise::American;
    for step in (0..n).rev() {
        let m = step + 1;
        let offset = n - step;
        let cur = ws.cur.get(..m + 1).unwrap_or(&[]);
        let nxt = ws.next.get_mut(..m).unwrap_or(&mut []);
        let lo = cur.get(..m).unwrap_or(&[]);
        let hi = cur.get(1..).unwrap_or(&[]);
        if american {
            let ex = if offset.is_multiple_of(2) {
                ws.ex_even.get(offset / 2..).unwrap_or(&[])
            } else {
                ws.ex_odd.get(offset / 2..).unwrap_or(&[])
            };
            let ex = ex.get(..m).unwrap_or(&[]);
            for (((o, &a), &b), &e) in nxt.iter_mut().zip(lo).zip(hi).zip(ex) {
                *o = (disc * (p * b + q * a)).max(e);
            }
        } else {
            for ((o, &a), &b) in nxt.iter_mut().zip(lo).zip(hi) {
                *o = disc * (p * b + q * a);
            }
        }
        std::mem::swap(&mut ws.cur, &mut ws.next);
    }
    finite(ws.cur.first().copied().unwrap_or(f64::NAN))
}

// ---------------------------------------------------------------------------
// Monte Carlo with a reusable workspace and an optional thread partition
// ---------------------------------------------------------------------------

/// Reusable path buffers for [`mc_price_ws`] and [`mc_price_par`].
#[derive(Debug, Default, Clone)]
pub struct McWorkspace {
    discounted: Vec<f64>,
    control: Vec<f64>,
}

impl McWorkspace {
    /// Empty workspace. The first call allocates.
    pub fn new() -> Self {
        Self::default()
    }

    /// Workspace sized for `n_paths` paths with a control variate: later
    /// calls with up to `n_paths` paths do not allocate.
    pub fn with_paths(n_paths: u64) -> Self {
        let n = usize::try_from(n_paths.saturating_add(1)).unwrap_or(0);
        Self {
            discounted: Vec::with_capacity(n),
            control: Vec::with_capacity(n),
        }
    }
}

/// Parameters shared by the fill workers.
#[derive(Clone, Copy)]
struct McParams {
    s: f64,
    k: f64,
    phi: f64,
    drift: f64,
    vol: f64,
    df_d: f64,
}

/// Fill `d_lo[j]` (and the mirrored `d_hi[j]` when antithetic) for the
/// draws `rng` produces next. Returns true on an overflowing spot.
#[inline]
fn fill_chunk(
    p: McParams,
    rng: &mut Xoshiro256StarStar,
    d_lo: &mut [f64],
    d_hi: &mut [f64],
    c_lo: &mut [f64],
    c_hi: &mut [f64],
) -> bool {
    // One specialised loop per variance-reduction case, each a zip over
    // equal-length slices: no per-path branch on the flags and no bounds
    // checks. The draw order and the arithmetic match the classic fill.
    let mut overflow = false;
    let mut spot = |z: f64| -> f64 {
        let s_t = p.s * (p.drift + p.vol * z).exp();
        overflow |= !s_t.is_finite();
        s_t
    };
    let pay = |s_t: f64| p.df_d * (p.phi * (s_t - p.k)).max(0.0);
    match (!d_hi.is_empty(), !c_lo.is_empty()) {
        (true, true) => {
            for (((a, b), c), e) in d_lo
                .iter_mut()
                .zip(d_hi.iter_mut())
                .zip(c_lo.iter_mut())
                .zip(c_hi.iter_mut())
            {
                let z = rng.next_normal();
                let s1 = spot(z);
                *a = pay(s1);
                *c = p.df_d * s1;
                let s2 = spot(-z);
                *b = pay(s2);
                *e = p.df_d * s2;
            }
        }
        (true, false) => {
            for (a, b) in d_lo.iter_mut().zip(d_hi.iter_mut()) {
                let z = rng.next_normal();
                *a = pay(spot(z));
                *b = pay(spot(-z));
            }
        }
        (false, true) => {
            for (a, c) in d_lo.iter_mut().zip(c_lo.iter_mut()) {
                let s1 = spot(rng.next_normal());
                *a = pay(s1);
                *c = p.df_d * s1;
            }
        }
        (false, false) => {
            for a in d_lo.iter_mut() {
                *a = pay(spot(rng.next_normal()));
            }
        }
    }
    overflow
}

fn mean_of(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Monte Carlo GK price using `ws` for the path buffers. Same bits as
/// [`crate::mc_price`] for the same arguments.
///
/// # Errors
///
/// [`KernelError`] on invalid inputs, `t = 0`, a bad path count, or
/// overflowing simulated spots.
pub fn mc_price_ws(
    ws: &mut McWorkspace,
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
) -> KResult<McResult> {
    mc_price_par(
        ws,
        s,
        k,
        t,
        r_d,
        r_f,
        sigma,
        option_type,
        n_paths,
        seed,
        antithetic,
        control_variate,
        1,
    )
}

/// Monte Carlo GK price with the path fill split over `n_threads` scoped
/// threads (fixed partition, one RNG copy per chunk started at the chunk's
/// offset in the single stream). Same bits as [`crate::mc_price`] for every
/// `n_threads`. `n_threads = 1` runs on the calling thread and spawns
/// nothing.
///
/// # Errors
///
/// As [`mc_price_ws`], plus [`KernelError::Threads`] for `n_threads = 0`.
pub fn mc_price_par(
    ws: &mut McWorkspace,
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
    n_threads: usize,
) -> KResult<McResult> {
    let phi = option_type.phi();
    validate(s, k, t, r_d, r_f, sigma)?;
    if t <= 0.0 {
        return Err(KernelError::ZeroTime);
    }
    if n_threads == 0 {
        return Err(KernelError::Threads);
    }
    let units = if antithetic {
        n_paths.div_ceil(2)
    } else {
        n_paths
    };
    let needed = if control_variate { 3 } else { 2 };
    if units < needed || n_paths > crate::monte_carlo::MAX_PATHS {
        return Err(KernelError::PathCount);
    }
    let params = McParams {
        s,
        k,
        phi,
        drift: (r_d - r_f - 0.5 * sigma * sigma) * t,
        vol: sigma * t.sqrt(),
        df_d: (-r_d * t).exp(),
    };
    // units <= MAX_PATHS fits usize on 64-bit targets; a narrower target
    // reports PathCount instead of truncating.
    let units = usize::try_from(units).map_err(|_| KernelError::PathCount)?;
    let total = if antithetic { 2 * units } else { units };

    if ws.discounted.len() < total {
        ws.discounted.resize(total, 0.0);
    }
    if control_variate && ws.control.len() < total {
        ws.control.resize(total, 0.0);
    }
    let disc = ws
        .discounted
        .get_mut(..total)
        .ok_or(KernelError::PathCount)?;
    let ctrl: &mut [f64] = if control_variate {
        ws.control.get_mut(..total).ok_or(KernelError::PathCount)?
    } else {
        &mut []
    };
    let (d_lo, d_hi) = if antithetic {
        disc.split_at_mut(units)
    } else {
        (disc, &mut [][..])
    };
    let (c_lo, c_hi): (&mut [f64], &mut [f64]) = if antithetic && control_variate {
        ctrl.split_at_mut(units)
    } else {
        (ctrl, &mut [])
    };

    let rng0 = Xoshiro256StarStar::new(seed);
    let workers = n_threads.min(units).max(1);
    let overflow = if workers == 1 {
        let mut rng = rng0;
        fill_chunk(params, &mut rng, d_lo, d_hi, c_lo, c_hi)
    } else {
        let chunk = units.div_ceil(workers);
        std::thread::scope(|sc| {
            let mut handles = Vec::with_capacity(workers);
            let mut dl = d_lo;
            let mut dh = d_hi;
            let mut cl = c_lo;
            let mut ch = c_hi;
            let mut start = 0usize;
            while !dl.is_empty() {
                let m = chunk.min(dl.len());
                let (a, rest_dl) = dl.split_at_mut(m);
                dl = rest_dl;
                let (b, rest_dh) = if dh.is_empty() {
                    (&mut [][..], dh)
                } else {
                    dh.split_at_mut(m)
                };
                dh = rest_dh;
                let (c, rest_cl) = if cl.is_empty() {
                    (&mut [][..], cl)
                } else {
                    cl.split_at_mut(m)
                };
                cl = rest_cl;
                let (e, rest_ch) = if ch.is_empty() {
                    (&mut [][..], ch)
                } else {
                    ch.split_at_mut(m)
                };
                ch = rest_ch;
                let mut rng = rng0.clone();
                let skip = start;
                handles.push(sc.spawn(move || {
                    // One u64 per normal: skipping `skip` outputs puts this
                    // copy exactly where the single stream would be.
                    for _ in 0..skip {
                        rng.next_u64();
                    }
                    fill_chunk(params, &mut rng, a, b, c, e)
                }));
                start += m;
            }
            handles
                .into_iter()
                .fold(false, |acc, h| acc | h.join().unwrap_or(true))
        })
    };
    if overflow {
        return Err(KernelError::Overflow);
    }

    // Reduce in place to one value per independent unit (pair average when
    // antithetic), then fit the control variate on those units, exactly as
    // the classic path does with fresh vectors.
    let disc = ws
        .discounted
        .get_mut(..total)
        .ok_or(KernelError::PathCount)?;
    if antithetic {
        let (lo, hi) = disc.split_at_mut(units);
        for (a, b) in lo.iter_mut().zip(hi.iter()) {
            *a = 0.5 * (*a + *b);
        }
    }
    let samples = disc.get_mut(..units).ok_or(KernelError::PathCount)?;
    if control_variate {
        let ctrl = ws.control.get_mut(..total).ok_or(KernelError::PathCount)?;
        if antithetic {
            let (lo, hi) = ctrl.split_at_mut(units);
            for (a, b) in lo.iter_mut().zip(hi.iter()) {
                *a = 0.5 * (*a + *b);
            }
        }
        let cu = ctrl.get(..units).ok_or(KernelError::PathCount)?;
        let control_mean_known = s * (-r_f * t).exp();
        let mp = mean_of(samples);
        let mc = mean_of(cu);
        let mut cov = 0.0;
        let mut var_c = 0.0;
        for (x, c) in samples.iter().zip(cu) {
            let dc = c - mc;
            cov += (x - mp) * dc;
            var_c += dc * dc;
        }
        let beta = if var_c > 0.0 { cov / var_c } else { 0.0 };
        for (x, c) in samples.iter_mut().zip(cu) {
            *x -= beta * (c - control_mean_known);
        }
    }
    let price = mean_of(samples);
    let ss: f64 = samples.iter().map(|v| (v - price) * (v - price)).sum();
    let n = samples.len() as f64;
    let se = (ss / (n - 1.0)).sqrt() / n.sqrt();
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
