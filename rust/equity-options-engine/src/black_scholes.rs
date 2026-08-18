//! Black-Scholes-Merton pricing for European equity options.
//!
//! # Conventions
//!
//! * Rates `r` and dividend yields `q` are continuously compounded,
//!   annualised (ACT/365F).
//! * `t` is the time to expiry in years; `sigma` is the annualised
//!   volatility of log-returns.
//! * Prices are in the same currency units as `s` and `k`.
//!
//! # Edge-case policy (documented + unit tested, identical to the Python
//! reference `eq_options.black_scholes`)
//!
//! * `t == 0`     -> intrinsic value `max(S - K, 0)` / `max(K - S, 0)`.
//! * `sigma == 0` -> discounted intrinsic on the forward,
//!   `exp(-rT) * max(±(F - K), 0)` with `F = S exp((r - q) T)`.
//! * `k == 0`     -> call is a forward on the stock, `S exp(-qT)`; put is 0.
//! * `s == 0`     -> call is 0; put is `K exp(-rT)`.
//! * Negative `s`, `k`, `t` or `sigma` (or NaN) return
//!   [`PricingError::InvalidInput`]. Negative `r` and `q` are fully
//!   supported.

use std::fmt;

/// Square root of 2*pi, used by the normal PDF.
pub(crate) const SQRT_2PI: f64 = 2.506_628_274_631_000_5;
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Option payoff direction.
///
/// # Examples
///
/// ```
/// use eq_options_engine::OptionType;
/// assert_eq!(OptionType::Call.sign(), 1.0);
/// assert_eq!(OptionType::Put.sign(), -1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    /// Right to buy at the strike: payoff `max(S - K, 0)`.
    Call,
    /// Right to sell at the strike: payoff `max(K - S, 0)`.
    Put,
}

impl OptionType {
    /// Payoff sign: `+1.0` for calls, `-1.0` for puts.
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptionType::Call => write!(f, "call"),
            OptionType::Put => write!(f, "put"),
        }
    }
}

/// Errors returned by the pricing and calibration routines.
///
/// This mirrors the Python reference, where the same conditions raise
/// `ValueError` (and the C++ engine, where they throw
/// `std::invalid_argument`) — Rust surfaces them as a `Result` instead of
/// unwinding.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, OptionType, PricingError};
/// let err = bs_price(-1.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap_err();
/// assert!(matches!(err, PricingError::InvalidInput(_)));
/// assert!(err.to_string().contains("S"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PricingError {
    /// An input violated its domain (negative price/vol/expiry, NaN, ...).
    InvalidInput(String),
    /// An iterative solver failed to converge or to bracket a root.
    NoConvergence(String),
    /// A quoted premium sits outside the static no-arbitrage bounds.
    ArbitrageBound(String),
}

impl fmt::Display for PricingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PricingError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            PricingError::NoConvergence(msg) => write!(f, "no convergence: {msg}"),
            PricingError::ArbitrageBound(msg) => write!(f, "arbitrage bound violated: {msg}"),
        }
    }
}

impl std::error::Error for PricingError {}

/// Validate common Black-Scholes inputs.
///
/// Requires `s >= 0`, `k >= 0`, `t >= 0`, `sigma >= 0` and no NaNs;
/// negative `r`/`q` are allowed (and therefore not checked here).
///
/// # Errors
///
/// [`PricingError::InvalidInput`] naming the offending parameter.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::validate_inputs;
/// assert!(validate_inputs(100.0, 100.0, 1.0, 0.2).is_ok());
/// assert!(validate_inputs(100.0, 100.0, -1.0, 0.2).is_err());
/// assert!(validate_inputs(f64::NAN, 100.0, 1.0, 0.2).is_err());
/// ```
pub fn validate_inputs(s: f64, k: f64, t: f64, sigma: f64) -> Result<(), PricingError> {
    for (name, value) in [("S", s), ("K", k), ("T", t), ("sigma", sigma)] {
        if value.is_nan() {
            return Err(PricingError::InvalidInput(format!(
                "{name} must not be NaN"
            )));
        }
        if value < 0.0 {
            return Err(PricingError::InvalidInput(format!(
                "{name} must be >= 0, got {value}"
            )));
        }
    }
    Ok(())
}

/// Complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Rust's `std` has no `erfc`, so this is W. J. Cody's rational Chebyshev
/// approximation (SIAM 1969; the netlib `CALERF` algorithm that underlies
/// most libm implementations): three regimes (`|x| <= 0.46875`,
/// `0.46875 < |x| <= 4`, `|x| > 4`) with the `exp(-x^2)` factor split as
/// `exp(-y16^2) exp(-(x-y16)(x+y16))` for tail accuracy. Verified against
/// reference values to better than `2e-15` relative error over
/// `[-27, 27]` (see the unit tests), comfortably inside the 1e-12
/// requirement for pricing.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::erfc;
/// assert!((erfc(0.0) - 1.0).abs() < 1e-15);
/// assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-14);
/// assert!((erfc(-1.0) - 1.8427007929497148).abs() < 1e-14);
/// ```
pub fn erfc(x: f64) -> f64 {
    // Cody's coefficients, regime |x| <= 0.46875: erf(x) = x P(x^2)/Q(x^2).
    const A: [f64; 5] = [
        3.161_123_743_870_565_6e0,
        1.138_641_541_510_501_6e2,
        3.774_852_376_853_020_2e2,
        3.209_377_589_138_469_5e3,
        1.857_777_061_846_031_5e-1,
    ];
    const B: [f64; 4] = [
        2.360_129_095_234_412_1e1,
        2.440_246_379_344_441_7e2,
        1.282_616_526_077_372_3e3,
        2.844_236_833_439_170_6e3,
    ];
    // Regime 0.46875 < |x| <= 4: erfc(x) = exp(-x^2) P(x)/Q(x).
    const C: [f64; 9] = [
        5.641_884_969_886_700_9e-1,
        8.883_149_794_388_376e0,
        6.611_919_063_714_163e1,
        2.986_351_381_974_001_3e2,
        8.819_522_212_417_691e2,
        1.712_047_612_634_070_6e3,
        2.051_078_377_826_071_5e3,
        1.230_339_354_797_997_2e3,
        2.153_115_354_744_038_5e-8,
    ];
    const D: [f64; 8] = [
        1.574_492_611_070_983_5e1,
        1.176_939_508_913_125e2,
        5.371_811_018_620_098_5e2,
        1.621_389_574_566_690_2e3,
        3.290_799_235_733_459_7e3,
        4.362_619_090_143_247e3,
        3.439_367_674_143_721_6e3,
        1.230_339_354_803_749_4e3,
    ];
    // Regime |x| > 4: erfc(x) = exp(-x^2)/x [1/sqrt(pi) - P(1/x^2)/Q(1/x^2)/x^2].
    const P: [f64; 6] = [
        3.053_266_349_612_323_4e-1,
        3.603_448_999_498_044_4e-1,
        1.257_817_261_112_292_5e-1,
        1.608_378_514_874_227_7e-2,
        6.587_491_615_298_378e-4,
        1.631_538_713_730_209_8e-2,
    ];
    const Q: [f64; 5] = [
        2.568_520_192_289_822_4e0,
        1.872_952_849_923_460_4e0,
        5.279_051_029_514_284e-1,
        6.051_834_131_244_132e-2,
        2.335_204_976_268_691_8e-3,
    ];
    const SQRPI: f64 = 5.641_895_835_477_562_9e-1; // 1/sqrt(pi)
    const THRESH: f64 = 0.46875;
    const XBIG: f64 = 26.543; // erfc underflows to 0 beyond this

    let y = x.abs();
    if y <= THRESH {
        // erfc = 1 - erf, erf via the rational approximation in x^2.
        let ysq = if y > 1.11e-16 { y * y } else { 0.0 };
        let mut xnum = A[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
        }
        return 1.0 - x * (xnum + A[3]) / (xden + B[3]);
    }

    let result = if y <= 4.0 {
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        let r = (xnum + C[7]) / (xden + D[7]);
        let y16 = (y * 16.0).floor() / 16.0;
        let del = (y - y16) * (y + y16);
        (-y16 * y16).exp() * (-del).exp() * r
    } else if y >= XBIG {
        0.0
    } else {
        let ysq = 1.0 / (y * y);
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let mut r = ysq * (xnum + P[4]) / (xden + Q[4]);
        r = (SQRPI - r) / y;
        let y16 = (y * 16.0).floor() / 16.0;
        let del = (y - y16) * (y + y16);
        (-y16 * y16).exp() * (-del).exp() * r
    };

    if x < 0.0 {
        2.0 - result
    } else {
        result
    }
}

/// Standard normal cumulative distribution function `Phi(x)`.
///
/// Computed as `0.5 * erfc(-x / sqrt(2))` — the erfc route keeps full
/// relative accuracy deep in the lower tail, where the naive
/// `0.5 (1 + erf)` form suffers catastrophic cancellation.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::norm_cdf;
/// assert!((norm_cdf(0.0) - 0.5).abs() < 1e-15);
/// assert!((norm_cdf(1.959963984540054) - 0.975).abs() < 1e-12);
/// assert!(norm_cdf(-37.0) > 0.0); // keeps tail mass until ~ -37.5
/// ```
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal probability density function `phi(x)`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::norm_pdf;
/// assert!((norm_pdf(0.0) - 0.3989422804014327).abs() < 1e-15);
/// ```
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

/// Intrinsic (exercise-now) value `max(S - K, 0)` or `max(K - S, 0)`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{black_scholes::intrinsic_value, OptionType};
/// assert_eq!(intrinsic_value(105.0, 100.0, OptionType::Call), 5.0);
/// assert_eq!(intrinsic_value(105.0, 100.0, OptionType::Put), 0.0);
/// ```
#[inline]
pub fn intrinsic_value(s: f64, k: f64, option_type: OptionType) -> f64 {
    (option_type.sign() * (s - k)).max(0.0)
}

/// Equity forward `F = S * exp((r - q) * T)`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::forward_price;
/// let f = forward_price(100.0, 1.0, 0.05, 0.02);
/// assert!((f - 100.0 * (0.03f64).exp()).abs() < 1e-12);
/// ```
#[inline]
pub fn forward_price(s: f64, t: f64, r: f64, q: f64) -> f64 {
    s * ((r - q) * t).exp()
}

/// Black-Scholes `d1` and `d2` terms.
///
/// `d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))` and
/// `d2 = d1 - sigma sqrt(T)`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] unless `s`, `k`, `t` and `sigma` are all
/// strictly positive (the terms are singular at the boundary).
///
/// # Examples
///
/// ```
/// use eq_options_engine::black_scholes::d1_d2;
/// let (d1, d2) = d1_d2(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
/// assert!((d1 - 0.35).abs() < 1e-12);
/// assert!((d2 - 0.15).abs() < 1e-12);
/// assert!(d1_d2(100.0, 100.0, 0.0, 0.05, 0.2, 0.0).is_err());
/// ```
pub fn d1_d2(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
) -> Result<(f64, f64), PricingError> {
    validate_inputs(s, k, t, sigma)?;
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return Err(PricingError::InvalidInput(format!(
            "d1/d2 require strictly positive S, K, T and sigma; \
             got S={s}, K={k}, T={t}, sigma={sigma}"
        )));
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    Ok((d1, d1 - sigma * sqrt_t))
}

/// Black-Scholes-Merton price of a European option with dividend yield.
///
/// # Arguments
///
/// * `s` — spot price (currency units), `s >= 0`.
/// * `k` — strike price (currency units), `k >= 0`.
/// * `t` — time to expiry in years (ACT/365F), `t >= 0`.
/// * `r` — continuously compounded annualised risk-free rate; negative
///   rates are supported.
/// * `sigma` — annualised log-return volatility, `sigma >= 0`.
/// * `q` — continuously compounded annualised dividend yield.
/// * `option_type` — call or put.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] if `s`, `k`, `t` or `sigma` is negative
/// or NaN.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{bs_price, OptionType};
/// let c = bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
/// assert!((c - 10.450583572185565).abs() < 1e-12); // golden value
///
/// // T = 0 returns intrinsic:
/// let expired = bs_price(105.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
/// assert_eq!(expired, 5.0);
/// ```
pub fn bs_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: OptionType,
) -> Result<f64, PricingError> {
    validate_inputs(s, k, t, sigma)?;
    if t == 0.0 {
        return Ok(intrinsic_value(s, k, option_type));
    }
    if k == 0.0 {
        // Zero-strike call is a (dividend-adjusted) forward on the stock.
        return Ok(match option_type {
            OptionType::Call => s * (-q * t).exp(),
            OptionType::Put => 0.0,
        });
    }
    if s == 0.0 {
        return Ok(match option_type {
            OptionType::Call => 0.0,
            OptionType::Put => k * (-r * t).exp(),
        });
    }
    if sigma == 0.0 {
        let forward = forward_price(s, t, r, q);
        let sign = option_type.sign();
        return Ok((-r * t).exp() * (sign * (forward - k)).max(0.0));
    }

    let (d1, d2) = d1_d2(s, k, t, r, sigma, q)?;
    let disc_s = s * (-q * t).exp();
    let disc_k = k * (-r * t).exp();
    Ok(match option_type {
        OptionType::Call => disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2),
        OptionType::Put => disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1),
    })
}
