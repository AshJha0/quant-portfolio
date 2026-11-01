//! Monte Carlo pricing of European options under GBM (exact scheme).
//!
//! The terminal stock price is simulated exactly:
//! `S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)`, `Z ~ N(0, 1)`,
//! so there is *no* time-discretisation bias — only statistical error,
//! which is reported as a standard error and 95% confidence interval.
//!
//! # Variance reduction (semantics mirror the Python reference)
//!
//! * **Antithetic variates**: pairs `(Z, -Z)`; the standard error is
//!   computed on the *pair averages* (the correct estimator for
//!   correlated pairs).
//! * **Control variate**: the discounted terminal stock `exp(-rT) S_T`
//!   with known mean `S exp(-qT)` (martingale property); the coefficient
//!   is fitted on the antithetic pair averages, the independent samples.
//!
//! Every entry point takes an explicit `u64` seed feeding the in-crate
//! SplitMix64 -> xoshiro256++ generator ([`crate::rng`]). Same seed =>
//! identical result on a given target and toolchain (`f64::exp`/`ln` come
//! from the platform libm, so other targets can differ in the last bits).
//!
//! Conventions: continuously compounded annualised `r`, `q` (ACT/365F),
//! `t` in years, `sigma` annualised.

use crate::black_scholes::{
    bs_price, validate_inputs, validate_rates, OptionType, PricingError,
};
use crate::fast::ErrorCode;
use crate::rng::{box_muller, uniform_from_bits, Xoshiro256PlusPlus};

/// Two-sided 95% normal quantile (matches the Python reference constant).
const Z95: f64 = 1.959_963_984_540_054;

/// Largest accepted `n_paths`.
///
/// The estimator materialises one `f64` per path (two with a control
/// variate), so 1e9 paths already asks for 8-16 GB. Anything above this
/// is rejected as [`PricingError::InvalidInput`] rather than being left
/// to abort the process in the allocator; it also keeps the internal
/// `2 * n_paths.div_ceil(2)` path count free of overflow on 32-bit
/// targets.
pub const MAX_PATHS: usize = 1_000_000_000;

/// Monte Carlo estimate with statistical error bars.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{mc_price, OptionType};
/// let res = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                    50_000, true, true, 42).unwrap();
/// assert!(res.std_error > 0.0);
/// assert!(res.ci_low < res.price && res.price < res.ci_high);
/// assert!(res.contains(10.450583572185565)); // analytic BS well inside CI
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McResult {
    /// Point estimate of the present value, currency units.
    pub price: f64,
    /// Standard error of the estimator.
    pub std_error: f64,
    /// Lower edge of the two-sided 95% confidence interval.
    pub ci_low: f64,
    /// Upper edge of the two-sided 95% confidence interval.
    pub ci_high: f64,
    /// Number of simulated paths (antithetic pairs count as 2 paths).
    pub n_paths: usize,
}

impl McResult {
    /// Return `true` if `x` lies inside the 95% confidence interval.
    ///
    /// # Examples
    ///
    /// ```
    /// use eq_options_engine::McResult;
    /// let r = McResult { price: 10.0, std_error: 0.1,
    ///                    ci_low: 9.804, ci_high: 10.196, n_paths: 1000 };
    /// assert!(r.contains(10.1));
    /// assert!(!r.contains(11.0));
    /// ```
    pub fn contains(&self, x: f64) -> bool {
        self.ci_low <= x && x <= self.ci_high
    }
}

/// Build an [`McResult`] from i.i.d. per-draw samples (ddof = 1).
fn summary(samples: &[f64], n_paths: usize) -> McResult {
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / (n as f64);
    let se = if n > 1 {
        let var = samples
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / ((n - 1) as f64);
        (var / (n as f64)).sqrt()
    } else {
        0.0
    };
    McResult {
        price: mean,
        std_error: se,
        ci_low: mean - Z95 * se,
        ci_high: mean + Z95 * se,
        n_paths,
    }
}

/// Monte Carlo price of a European option under exact-scheme GBM.
///
/// # Arguments
///
/// * `s`, `k`, `t`, `r`, `sigma`, `q`, `option_type` — as in
///   [`bs_price`].
/// * `n_paths` — total number of paths (rounded up to even when
///   `antithetic`), in `[2, MAX_PATHS]`.
/// * `antithetic` — use antithetic pairs `(Z, -Z)`.
/// * `control_variate` — use the discounted terminal stock as a control
///   variate, coefficient fitted on the antithetic pair averages.
/// * `seed` — explicit RNG seed; same seed => identical result on one target.
///
/// `t == 0` or `sigma == 0` are deterministic; the exact Black-Scholes
/// value is returned with `std_error = 0`.
///
/// # Errors
///
/// [`PricingError::InvalidInput`] on invalid inputs, `n_paths < 2`, or
/// `n_paths > MAX_PATHS`.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{mc_price, OptionType};
/// let a = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                  10_000, true, true, 7).unwrap();
/// let b = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                  10_000, true, true, 7).unwrap();
/// assert_eq!(a.price.to_bits(), b.price.to_bits()); // bit-reproducible
/// ```
#[allow(clippy::too_many_arguments)]
pub fn mc_price(
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
) -> Result<McResult, PricingError> {
    validate_inputs(s, k, t, sigma)?;
    validate_rates(r, q)?;
    // A standard error needs two independent samples (paths, or antithetic
    // pairs) left after fitting: three with the control variate, since two
    // points fit the regression line exactly and leave a zero residual.
    let units = if antithetic { n_paths.div_ceil(2) } else { n_paths };
    let needed = if control_variate { 3 } else { 2 };
    if units < needed {
        return Err(PricingError::InvalidInput(format!(
            "n_paths = {n_paths} gives {units} independent sample(s); at least              {needed} are needed for a standard error"
        )));
    }
    if n_paths > MAX_PATHS {
        return Err(PricingError::InvalidInput(format!(
            "n_paths must be <= {MAX_PATHS}, got {n_paths} \
             (the estimator buffers 8-16 bytes per path)"
        )));
    }
    if t == 0.0 || sigma == 0.0 {
        let exact = bs_price(s, k, t, r, sigma, q, option_type)?;
        return Ok(McResult {
            price: exact,
            std_error: 0.0,
            ci_low: exact,
            ci_high: exact,
            n_paths,
        });
    }

    let mut ws = McWorkspace::new();
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
        1,
        &mut ws,
    )
    .map_err(|e| match e {
        ErrorCode::PathOverflow => PricingError::InvalidInput(
            "simulated terminal prices overflow a double; drift or vol too large".into(),
        ),
        _ => PricingError::InvalidInput("Monte Carlo estimate is not finite".into()),
    })
}

/// Reusable buffers for the Monte Carlo estimator.
///
/// Holds one `f64` per independent sample (path, or antithetic pair),
/// one more with a control variate, and the raw `u64` stream for the
/// multi-threaded path. Capacity grows to the largest request seen and is
/// then reused, so repeated single-threaded pricing does not allocate.
///
/// # Examples
///
/// ```
/// use eq_options_engine::{fast::try_mc_price, McWorkspace, OptionType};
/// let mut ws = McWorkspace::new();
/// let a = try_mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                      10_000, true, true, 7, 1, &mut ws).unwrap();
/// let b = try_mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
///                      10_000, true, true, 7, 4, &mut ws).unwrap();
/// assert_eq!(a.price.to_bits(), b.price.to_bits()); // threads do not change bits
/// ```
#[derive(Debug, Clone, Default)]
pub struct McWorkspace {
    samples: Vec<f64>,
    control: Vec<f64>,
    raw: Vec<u64>,
}

impl McWorkspace {
    /// Empty workspace. The first call sizes it.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Per-path constants shared by every chunk.
#[derive(Clone, Copy)]
struct PathParams {
    s: f64,
    k: f64,
    disc: f64,
    sign: f64,
    drift: f64,
    vol_sqrt_t: f64,
    antithetic: bool,
    control_variate: bool,
}

impl PathParams {
    #[inline]
    fn terminal(&self, z: f64) -> f64 {
        self.s * (self.drift + self.vol_sqrt_t * z).exp()
    }

    #[inline]
    fn payoff(&self, s_t: f64) -> f64 {
        self.disc * (self.sign * (s_t - self.k)).max(0.0)
    }

    /// Replace each normal in `samples` by its unit sample (path payoff,
    /// or antithetic pair average) and write the matching control.
    /// Returns `false` if any terminal price overflows.
    #[inline]
    fn units(&self, samples: &mut [f64], control: &mut [f64]) -> bool {
        let mut finite = true;
        let disc = self.disc;
        if self.antithetic {
            // Pair (z, -z). Same operations as the stored-mirror layout:
            // average of the two payoffs, average of the two controls.
            if self.control_variate {
                for (x, c) in samples.iter_mut().zip(control.iter_mut()) {
                    let (s1, s2) = (self.terminal(*x), self.terminal(-*x));
                    finite &= s1.is_finite() & s2.is_finite();
                    *x = 0.5 * (self.payoff(s1) + self.payoff(s2));
                    *c = 0.5 * (disc * s1 + disc * s2);
                }
            } else {
                for x in samples.iter_mut() {
                    let (s1, s2) = (self.terminal(*x), self.terminal(-*x));
                    finite &= s1.is_finite() & s2.is_finite();
                    *x = 0.5 * (self.payoff(s1) + self.payoff(s2));
                }
            }
        } else if self.control_variate {
            for (x, c) in samples.iter_mut().zip(control.iter_mut()) {
                let s_t = self.terminal(*x);
                finite &= s_t.is_finite();
                *x = self.payoff(s_t);
                *c = disc * s_t;
            }
        } else {
            for x in samples.iter_mut() {
                let s_t = self.terminal(*x);
                finite &= s_t.is_finite();
                *x = self.payoff(s_t);
            }
        }
        finite
    }
}

/// Fill `out` with Box-Muller normals from the raw stream `raw`, laid out
/// as [`Xoshiro256PlusPlus::fill_standard_normal`] lays them out. `out`
/// must start at an even stream position.
#[inline]
fn normals_from_raw(raw: &[u64], out: &mut [f64]) {
    for (pair, &[u0, u1]) in out.chunks_mut(2).zip(raw.as_chunks::<2>().0) {
        let (z0, z1) = box_muller(uniform_from_bits(u0), uniform_from_bits(u1));
        match pair {
            [a, b] => {
                *a = z0;
                *b = z1;
            }
            [a] => *a = z0,
            _ => {}
        }
    }
}

/// Estimator for validated inputs.
///
/// `threads <= 1` runs on the calling thread and does not allocate once
/// `ws` is sized. `threads > 1` draws the raw stream serially, then runs
/// Box-Muller and the payoffs on a fixed partition inside one
/// `std::thread::scope`. Every reduction runs serially in index order, so
/// the result has the same bits for every thread count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mc_core(
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
    let mut rng = Xoshiro256PlusPlus::new(seed);
    let params = PathParams {
        s,
        k,
        disc: (-r * t).exp(),
        sign: option_type.sign(),
        drift: (r - q - 0.5 * sigma * sigma) * t,
        vol_sqrt_t: sigma * t.sqrt(),
        antithetic,
        control_variate,
    };
    // Fresh normals: one per path, or one per antithetic pair.
    let (units, n_eff) = if antithetic {
        let half = n_paths.div_ceil(2);
        (half, 2 * half)
    } else {
        (n_paths, n_paths)
    };
    ws.samples.clear();
    ws.samples.resize(units, 0.0);
    ws.control.clear();
    ws.control.resize(if control_variate { units } else { 0 }, 0.0);

    let finite = if threads <= 1 {
        rng.fill_standard_normal(&mut ws.samples);
        params.units(&mut ws.samples, &mut ws.control)
    } else {
        ws.raw.clear();
        ws.raw.resize(2 * units.div_ceil(2), 0);
        rng.fill_u64(&mut ws.raw);
        // An even chunk length keeps every chunk on a Box-Muller pair.
        let chunk = units.div_ceil(threads).next_multiple_of(2).max(2);
        let ctrl_chunk = if control_variate { chunk } else { 0 };
        let mut ok = true;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            let mut ctrl_rest: &mut [f64] = &mut ws.control;
            for (smp, raw) in ws.samples.chunks_mut(chunk).zip(ws.raw.chunks(chunk)) {
                let take = ctrl_chunk.min(ctrl_rest.len());
                let (ctrl, rest) = std::mem::take(&mut ctrl_rest).split_at_mut(take);
                ctrl_rest = rest;
                handles.push(scope.spawn(move || {
                    normals_from_raw(raw, smp);
                    params.units(smp, ctrl)
                }));
            }
            for h in handles {
                ok &= h.join().unwrap_or(false);
            }
        });
        ok
    };
    if !finite {
        return Err(ErrorCode::PathOverflow);
    }

    let samples = &mut ws.samples;
    let control = &ws.control;
    if control_variate {
        // Control: discounted terminal stock, known mean S exp(-qT). The
        // beta is fitted on the same units as the estimator (pair
        // averages under antithetic sampling).
        let control_mean = s * (-q * t).exp();
        let n = samples.len() as f64;
        let mean_p = samples.iter().sum::<f64>() / n;
        let mean_c = control.iter().sum::<f64>() / n;
        let mut cov_pc = 0.0;
        let mut var_c = 0.0;
        for (p, c) in samples.iter().zip(control) {
            let dc = c - mean_c;
            cov_pc += (p - mean_p) * dc;
            var_c += dc * dc;
        }
        let beta = if var_c > 0.0 { cov_pc / var_c } else { 0.0 };
        for (p, c) in samples.iter_mut().zip(control) {
            *p -= beta * (c - control_mean);
        }
    }
    let res = summary(samples, n_eff);
    if !(res.price.is_finite() && res.std_error.is_finite()) {
        return Err(ErrorCode::NonFiniteResult);
    }
    Ok(res)
}
