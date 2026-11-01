//! Non-allocating, non-panicking entry points for latency-sensitive code.
//!
//! The functions here return the same numbers as the classic API, bit for
//! bit, because both call the same internal kernels. They differ in three
//! ways:
//!
//! * Errors are an [`ErrorCode`]: a one-byte `Copy` enum. No `String` is
//!   built on any path, success or failure.
//! * The tree and Monte Carlo pricers take a caller-owned workspace
//!   ([`CrrWorkspace`], [`McWorkspace`]). Once it has grown to the largest
//!   request, a call does not touch the heap.
//! * Struct-of-arrays batch pricers ([`bs_price_batch`],
//!   [`bs_greeks_batch`]) price a whole slice in one call.
//!
//! An [`ErrorCode`] converts into a [`PricingError`] with `?` or `.into()`
//! when the caller wants the classic error type. The message is a fixed
//! description of the code, not the detailed classic message.
//!
//! # Examples
//!
//! ```
//! use eq_options_engine::fast::{try_bs_price, ErrorCode};
//! use eq_options_engine::{bs_price, OptionType};
//!
//! let a = try_bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
//! let b = bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
//! assert_eq!(a.to_bits(), b.to_bits());
//!
//! let err = try_bs_price(-1.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call);
//! assert_eq!(err, Err(ErrorCode::NegativeInput));
//! ```

use std::fmt;

use crate::binomial::{crr_tree, deterministic_american, CrrWorkspace, Exercise, MAX_STEPS};
use crate::black76::{b76_d1_d2_core, black76_greeks_core, black76_price_core, Black76Greeks};
use crate::black_scholes::{
    bs_price_core, check_inputs, check_rates, d1_d2_core, OptionType, PricingError,
};
use crate::greeks::{bs_greeks_core, BsGreeks};
use crate::iv_core::black_implied_vol;
use crate::monte_carlo::{mc_core, McResult, McWorkspace, MAX_PATHS};

/// Why a non-allocating pricer rejected its inputs or result.
///
/// # Examples
///
/// ```
/// use eq_options_engine::fast::ErrorCode;
/// use eq_options_engine::PricingError;
/// let e: PricingError = ErrorCode::NanInput.into();
/// assert!(matches!(e, PricingError::InvalidInput(_)));
/// assert_eq!(ErrorCode::NanInput.to_string(), "an input is NaN");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    /// One of `S`, `K`, `T`, `sigma` is NaN.
    NanInput,
    /// One of `S`, `K`, `T`, `sigma` is infinite.
    InfiniteInput,
    /// One of `S`, `K`, `T`, `sigma` is negative.
    NegativeInput,
    /// `r` or `q` is NaN or infinite.
    NonFiniteRate,
    /// A kernel that needs strictly positive inputs got a zero.
    NonPositiveInput,
    /// The quoted price is NaN or infinite.
    NonFinitePrice,
    /// The implied-vol quote is at or outside a no-arbitrage bound.
    ArbitrageBound,
    /// The implied-vol solver did not converge.
    NoConvergence,
    /// `n_steps` is 0 or above [`MAX_STEPS`].
    StepsOutOfRange,
    /// `n_paths` gives too few independent samples or exceeds [`MAX_PATHS`].
    PathsOutOfRange,
    /// The CRR risk-neutral probability is outside (0, 1).
    BadProbability,
    /// A simulated terminal price overflowed a double.
    PathOverflow,
    /// The result is not a finite double.
    NonFiniteResult,
    /// Batch input and output slices have different lengths.
    LengthMismatch,
}

impl ErrorCode {
    /// Fixed description of the code.
    pub const fn as_str(self) -> &'static str {
        match self {
            ErrorCode::NanInput => "an input is NaN",
            ErrorCode::InfiniteInput => "an input is infinite",
            ErrorCode::NegativeInput => "an input is negative",
            ErrorCode::NonFiniteRate => "a rate is not finite",
            ErrorCode::NonPositiveInput => "an input must be strictly positive",
            ErrorCode::NonFinitePrice => "the quoted price is not finite",
            ErrorCode::ArbitrageBound => "the price is at or outside a no-arbitrage bound",
            ErrorCode::NoConvergence => "the solver did not converge",
            ErrorCode::StepsOutOfRange => "n_steps is out of range",
            ErrorCode::PathsOutOfRange => "n_paths is out of range",
            ErrorCode::BadProbability => "risk-neutral probability outside (0, 1)",
            ErrorCode::PathOverflow => "a simulated terminal price overflows a double",
            ErrorCode::NonFiniteResult => "the result is not a finite double",
            ErrorCode::LengthMismatch => "batch slices have different lengths",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for ErrorCode {}

impl From<ErrorCode> for PricingError {
    fn from(e: ErrorCode) -> Self {
        let msg = e.as_str().to_string();
        match e {
            ErrorCode::ArbitrageBound => PricingError::ArbitrageBound(msg),
            ErrorCode::NoConvergence => PricingError::NoConvergence(msg),
            _ => PricingError::InvalidInput(msg),
        }
    }
}

#[inline]
fn finite(v: f64) -> Result<f64, ErrorCode> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(ErrorCode::NonFiniteResult)
    }
}

/// Black-Scholes price. Same value as [`crate::bs_price`]; rejects the
/// same inputs.
///
/// # Errors
///
/// [`ErrorCode`] for invalid inputs or a non-finite result.
#[inline]
pub fn try_bs_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> Result<f64, ErrorCode> {
    check_inputs(s, k, t, sigma)?;
    check_rates(r, q)?;
    finite(bs_price_core(s, k, t, r, sigma, q, option_type))
}

/// Analytic Greeks. Same values as [`crate::bs_greeks`]; rejects the same
/// inputs (it needs strictly positive `S`, `K`, `T`, `sigma`).
///
/// # Errors
///
/// [`ErrorCode`] for invalid inputs.
#[inline]
pub fn try_bs_greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> Result<BsGreeks, ErrorCode> {
    check_inputs(s, k, t, sigma)?;
    check_rates(r, q)?;
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return Err(ErrorCode::NonPositiveInput);
    }
    let (d1, d2) = d1_d2_core(s, k, t, r, sigma, q);
    Ok(bs_greeks_core(s, k, t, r, sigma, q, option_type, d1, d2))
}

/// Implied volatility. Same value as [`crate::implied_vol`]; rejects the
/// same inputs.
///
/// # Errors
///
/// [`ErrorCode::ArbitrageBound`] for quotes at or outside the bounds,
/// [`ErrorCode::NoConvergence`] if the solver fails, other codes for
/// invalid inputs.
pub fn try_implied_vol(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    option_type: OptionType,
) -> Result<f64, ErrorCode> {
    check_inputs(s, k, t, 0.0)?;
    check_rates(r, q)?;
    if !price.is_finite() {
        return Err(ErrorCode::NonFinitePrice);
    }
    if t <= 0.0 || s <= 0.0 || k <= 0.0 {
        return Err(ErrorCode::NonPositiveInput);
    }
    let lower = finite(bs_price_core(s, k, t, r, 0.0, q, option_type))?;
    let upper = match option_type {
        OptionType::Call => finite(s * (-q * t).exp())?,
        OptionType::Put => k * (-r * t).exp(),
    };
    if price <= lower || price >= upper {
        return Err(ErrorCode::ArbitrageBound);
    }
    let df = finite((-r * t).exp())?;
    let forward = finite(s * ((r - q) * t).exp())?;
    let mut evals = 0;
    black_implied_vol(
        price,
        forward,
        k,
        t,
        df,
        option_type == OptionType::Call,
        &mut evals,
    )
    .map_err(|msg| {
        if msg.starts_with("price") {
            ErrorCode::ArbitrageBound
        } else {
            ErrorCode::NoConvergence
        }
    })
}

/// Black-76 price. Same value as [`crate::black76_price`].
///
/// # Errors
///
/// [`ErrorCode`] for invalid inputs.
#[inline]
pub fn try_black76_price(
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: OptionType,
) -> Result<f64, ErrorCode> {
    check_inputs(f, k, t, sigma)?;
    check_rates(r, 0.0)?;
    Ok(black76_price_core(f, k, t, r, sigma, option_type))
}

/// Black-76 Greeks. Same values as [`crate::black76_greeks`].
///
/// # Errors
///
/// [`ErrorCode`] for invalid inputs (strictly positive `F`, `K`, `T`,
/// `sigma` required).
#[inline]
pub fn try_black76_greeks(
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: OptionType,
) -> Result<Black76Greeks, ErrorCode> {
    check_rates(r, 0.0)?;
    check_inputs(f, k, t, sigma)?;
    if f <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return Err(ErrorCode::NonPositiveInput);
    }
    let (d1, d2) = b76_d1_d2_core(f, k, t, sigma);
    Ok(black76_greeks_core(f, k, t, r, sigma, option_type, d1, d2))
}

/// CRR binomial price using a caller-owned workspace. Same value as
/// [`crate::crr_price`]. Allocates only when `ws` must grow.
///
/// # Errors
///
/// [`ErrorCode`] for invalid inputs, `n_steps` outside
/// `[1, MAX_STEPS]`, a risk-neutral probability outside (0, 1), or a
/// non-finite result.
#[allow(clippy::too_many_arguments)]
pub fn try_crr_price(
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
    check_inputs(s, k, t, sigma)?;
    check_rates(r, q)?;
    if !(1..=MAX_STEPS).contains(&n_steps) {
        return Err(ErrorCode::StepsOutOfRange);
    }
    let sign = option_type.sign();
    if t == 0.0 {
        return Ok((sign * (s - k)).max(0.0));
    }
    if s == 0.0 {
        return match (option_type, exercise) {
            (OptionType::Call, _) => Ok(0.0),
            (OptionType::Put, Exercise::American) => Ok(k.max(finite(k * (-r * t).exp())?)),
            (OptionType::Put, Exercise::European) => finite(k * (-r * t).exp()),
        };
    }
    if k == 0.0 {
        if option_type == OptionType::Put {
            return Ok(0.0);
        }
        if exercise == Exercise::American && q > 0.0 {
            return Ok(s);
        }
        return finite(s * (-q * t).exp());
    }
    if sigma == 0.0 {
        return match exercise {
            Exercise::European => finite(bs_price_core(s, k, t, r, 0.0, q, option_type)),
            Exercise::American => {
                finite(deterministic_american(s, k, t, r, q, option_type, n_steps))
            }
        };
    }
    crr_tree(s, k, t, r, sigma, q, option_type, exercise, n_steps, ws)
}

/// Monte Carlo price using a caller-owned workspace. Same value as
/// [`crate::mc_price`] for every `threads`.
///
/// `threads <= 1` runs on the calling thread and does not allocate once
/// `ws` is sized. `threads > 1` spawns that many scoped threads once for
/// the call (fixed partition, one chunk each); thread start-up allocates,
/// so use it for large path counts only.
///
/// # Errors
///
/// [`ErrorCode`] for invalid inputs, too few or too many paths, overflow
/// of a simulated price, or a non-finite estimate.
#[allow(clippy::too_many_arguments)]
pub fn try_mc_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
    n_paths: usize,
    antithetic: bool,
    control_variate: bool,
    seed: u64,
    threads: usize,
    ws: &mut McWorkspace,
) -> Result<McResult, ErrorCode> {
    check_inputs(s, k, t, sigma)?;
    check_rates(r, q)?;
    let units = if antithetic {
        n_paths.div_ceil(2)
    } else {
        n_paths
    };
    let needed = if control_variate { 3 } else { 2 };
    if units < needed || n_paths > MAX_PATHS {
        return Err(ErrorCode::PathsOutOfRange);
    }
    if t == 0.0 || sigma == 0.0 {
        let exact = finite(bs_price_core(s, k, t, r, sigma, q, option_type))?;
        return Ok(McResult {
            price: exact,
            std_error: 0.0,
            ci_low: exact,
            ci_high: exact,
            n_paths,
        });
    }
    mc_core(
        s,
        k,
        t,
        r,
        sigma,
        q,
        option_type,
        n_paths,
        antithetic,
        control_variate,
        seed,
        threads,
        ws,
    )
}

/// Struct-of-arrays inputs for the batch pricers. All slices must have
/// the same length.
#[derive(Debug, Clone, Copy)]
pub struct BsBatch<'a> {
    /// Spot prices.
    pub s: &'a [f64],
    /// Strikes.
    pub k: &'a [f64],
    /// Times to expiry, years.
    pub t: &'a [f64],
    /// Risk-free rates.
    pub r: &'a [f64],
    /// Volatilities.
    pub sigma: &'a [f64],
    /// Dividend yields.
    pub q: &'a [f64],
    /// Call or put per row.
    pub option_type: &'a [OptionType],
}

impl BsBatch<'_> {
    /// Number of rows, or [`ErrorCode::LengthMismatch`].
    #[inline]
    fn len_checked(&self, out_len: usize) -> Result<usize, ErrorCode> {
        let n = self.s.len();
        let same = [
            self.k.len(),
            self.t.len(),
            self.r.len(),
            self.sigma.len(),
            self.q.len(),
            self.option_type.len(),
            out_len,
        ]
        .iter()
        .all(|&l| l == n);
        if same {
            Ok(n)
        } else {
            Err(ErrorCode::LengthMismatch)
        }
    }
}

/// Error from a batch pricer: the first failing row and its code. Rows
/// before `index` are written; rows from `index` on are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchError {
    /// First failing row (`usize::MAX` for a length mismatch).
    pub index: usize,
    /// Why it failed.
    pub code: ErrorCode,
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row {}: {}", self.index, self.code)
    }
}

impl std::error::Error for BatchError {}

const MISMATCH: BatchError = BatchError {
    index: usize::MAX,
    code: ErrorCode::LengthMismatch,
};

/// Price every row of `inputs` into `out`. Row `i` equals
/// `try_bs_price` on row `i`, bit for bit.
///
/// # Errors
///
/// [`BatchError`] with the first failing row, or a length mismatch.
///
/// # Examples
///
/// ```
/// use eq_options_engine::fast::{bs_price_batch, BsBatch};
/// use eq_options_engine::OptionType;
/// let s = [90.0, 100.0, 110.0];
/// let k = [100.0; 3];
/// let t = [0.5; 3];
/// let r = [0.03; 3];
/// let v = [0.2; 3];
/// let q = [0.0; 3];
/// let ot = [OptionType::Call; 3];
/// let mut out = [0.0; 3];
/// let b = BsBatch { s: &s, k: &k, t: &t, r: &r, sigma: &v, q: &q, option_type: &ot };
/// bs_price_batch(&b, &mut out).unwrap();
/// assert!(out[0] < out[1] && out[1] < out[2]);
/// ```
pub fn bs_price_batch(inputs: &BsBatch<'_>, out: &mut [f64]) -> Result<(), BatchError> {
    let n = inputs.len_checked(out.len()).map_err(|_| MISMATCH)?;
    let rows = out
        .iter_mut()
        .zip(&inputs.s[..n])
        .zip(&inputs.k[..n])
        .zip(&inputs.t[..n])
        .zip(&inputs.r[..n])
        .zip(&inputs.sigma[..n])
        .zip(&inputs.q[..n])
        .zip(&inputs.option_type[..n]);
    for (index, (((((((o, &s), &k), &t), &r), &sigma), &q), &ot)) in rows.enumerate() {
        *o = try_bs_price(s, k, t, r, sigma, q, ot).map_err(|code| BatchError { index, code })?;
    }
    Ok(())
}

/// Struct-of-arrays outputs for [`bs_greeks_batch`]. All slices must have
/// the batch length.
#[derive(Debug)]
pub struct GreeksOut<'a> {
    /// Prices.
    pub price: &'a mut [f64],
    /// Deltas.
    pub delta: &'a mut [f64],
    /// Gammas.
    pub gamma: &'a mut [f64],
    /// Vegas (per unit vol).
    pub vega: &'a mut [f64],
    /// Thetas (per year).
    pub theta: &'a mut [f64],
    /// Rhos (per unit rate).
    pub rho: &'a mut [f64],
    /// Vannas.
    pub vanna: &'a mut [f64],
    /// Volgas.
    pub volga: &'a mut [f64],
}

/// Price and Greeks for every row of `inputs`. Row `i` equals
/// `try_bs_greeks` on row `i`, bit for bit.
///
/// # Errors
///
/// [`BatchError`] with the first failing row, or a length mismatch.
pub fn bs_greeks_batch(inputs: &BsBatch<'_>, out: &mut GreeksOut<'_>) -> Result<(), BatchError> {
    let n = inputs.len_checked(out.price.len()).map_err(|_| MISMATCH)?;
    let lens = [
        out.delta.len(),
        out.gamma.len(),
        out.vega.len(),
        out.theta.len(),
        out.rho.len(),
        out.vanna.len(),
        out.volga.len(),
    ];
    if lens.iter().any(|&l| l != n) {
        return Err(MISMATCH);
    }
    let rows = inputs.s[..n]
        .iter()
        .zip(&inputs.k[..n])
        .zip(&inputs.t[..n])
        .zip(&inputs.r[..n])
        .zip(&inputs.sigma[..n])
        .zip(&inputs.q[..n])
        .zip(&inputs.option_type[..n]);
    for (index, ((((((&s, &k), &t), &r), &sigma), &q), &ot)) in rows.enumerate() {
        let g =
            try_bs_greeks(s, k, t, r, sigma, q, ot).map_err(|code| BatchError { index, code })?;
        // `index < n` and every output has length `n`.
        if let (Some(p), Some(de), Some(ga), Some(ve), Some(th), Some(rh), Some(va), Some(vo)) = (
            out.price.get_mut(index),
            out.delta.get_mut(index),
            out.gamma.get_mut(index),
            out.vega.get_mut(index),
            out.theta.get_mut(index),
            out.rho.get_mut(index),
            out.vanna.get_mut(index),
            out.volga.get_mut(index),
        ) {
            (*p, *de, *ga, *ve, *th, *rh, *va, *vo) = (
                g.price, g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga,
            );
        }
    }
    Ok(())
}
