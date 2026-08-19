//! FX stress testing: historical replays, hypothetical scenarios, peg
//! breaks, and reverse stress (mirrors Python `fx_var.stress_testing` / C++
//! `fxvar::stress`, spot/forward factor set — no vol factors here).
//!
//! Stress is the complement to VaR, not a substitute: HS and var-covar are
//! *blind* to risks absent from the estimation window (a pegged currency
//! has no history of breaking — until it does). Every scenario is a joint
//! factor-shock map {per-ccy spot shocks (log returns), rate shifts
//! (absolute)} applied through full revaluation, so forwards feel their
//! rate legs.
//!
//! Historical replay calibrations (close-to-close one-day moves vs USD,
//! sources in `docs/METHODOLOGY.md`):
//!
//! * **GBP flash** — Brexit referendum, 24 Jun 2016: GBPUSD -8.1% (1.4877
//!   -> 1.3679), EURUSD -2.4%, JPY +3.9% (safe haven), and a BoE-easing
//!   rate shift.
//! * **CHF depeg** — SNB floor removal, 15 Jan 2015: CHFUSD +14.9%
//!   close-to-close (intraday +30%+), EURUSD -1.4%. The canonical peg
//!   break: the prior 250 days of USDCHF had no daily move over 1.9%.
//! * **JPY carry unwind**, 7-8 Oct 1998: JPYUSD +11.5% over two days as
//!   USDJPY fell 131 -> 117 (LTCM deleveraging); AUD -4%.
//!
//! Reverse stress: for a linearised book with exposures `w` and factor
//! covariance `Sigma`, the most damaging shock at Mahalanobis radius `k` is
//! `dx* = -k Sigma w / sqrt(w' Sigma w)`, `loss = k sqrt(w' Sigma w)` —
//! closed form, verified against an independent numerical search in tests
//! (tolerance `1e-6`).

use std::collections::HashMap;

use crate::book::{fx_factor, ir_factor, Book, CompiledBook, Market};
use crate::matrix::Matrix;
use crate::rng::Rng;
use crate::{FxVarError, Result};

/// A named stress scenario: factor shocks + description. `shocks` maps
/// factor names to shocks in engine units (`"FX:*"` log returns, `"IR:*"`
/// absolute). Shocks for factors the book does not carry are ignored at
/// run time, so one scenario library serves every book.
#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    /// Human-readable scenario name.
    pub name: String,
    /// Factor -> shock map.
    pub shocks: HashMap<String, f64>,
    /// Longer description / calibration notes.
    pub description: String,
}

/// Convert a simple percentage move to a log return: `ln(1 + pct)`.
///
/// The domain is `pct > -1` (a currency cannot lose more than 100 % of its
/// value). Outside it the map is not a log return at all: `pct = -1`
/// returns `ln(0) = -inf` and `pct < -1` returns `NaN`. Both used to be
/// returned *silently*, so a fat-fingered `-1.0` in a scenario definition
/// produced an infinite shock, an infinite scenario P&L, and a stress
/// report whose worst-case row was `-inf` — or, with `NaN`, a report that
/// sorted arbitrarily and showed the "worst" scenario as harmless. The
/// function therefore returns a [`Result`] and rejects the boundary.
///
/// # Errors
/// [`FxVarError::Invalid`] unless `pct` is finite and `> -1`.
///
/// ```
/// use fx_var_engine::stress::simple_to_log;
/// assert!((simple_to_log(0.0).unwrap()).abs() < 1e-15);
/// assert!((simple_to_log(-0.081).unwrap() - (-0.081f64).ln_1p()).abs() < 1e-15);
/// assert!(simple_to_log(-1.0).is_err());   // ln(0) = -inf
/// assert!(simple_to_log(-1.5).is_err());   // NaN
/// assert!(simple_to_log(f64::NAN).is_err());
/// ```
pub fn simple_to_log(pct: f64) -> Result<f64> {
    if !(pct > -1.0) || !pct.is_finite() {
        return Err(FxVarError::invalid(format!(
            "simple_to_log: a simple move must be finite and > -100%, got {pct}; \
             ln(1 + pct) is -inf at -100% and NaN below it"
        )));
    }
    Ok(pct.ln_1p())
}

/// Internal helper for the *calibrated library constants* below, all of
/// which are compile-time literals inside the domain.
fn lit_log(pct: f64) -> f64 {
    simple_to_log(pct).expect("calibrated scenario constant is inside (-1, inf)")
}

/// Library of calibrated historical FX replay scenarios, keyed
/// `"brexit_2016"`, `"chf_depeg_2015"`, `"jpy_1998"`.
pub fn historical_scenarios() -> HashMap<String, Scenario> {
    let mut lib = HashMap::new();
    lib.insert(
        "brexit_2016".to_string(),
        Scenario {
            name: "GBP flash - Brexit referendum (24 Jun 2016)".to_string(),
            shocks: HashMap::from([
                (fx_factor("GBP").unwrap(), lit_log(-0.081)),
                (fx_factor("EUR").unwrap(), lit_log(-0.024)),
                (fx_factor("JPY").unwrap(), lit_log(0.039)),
                (fx_factor("CHF").unwrap(), lit_log(0.015)),
                (fx_factor("AUD").unwrap(), lit_log(-0.019)),
                (ir_factor("GBP"), -0.0025), // BoE easing repricing
            ]),
            description: "Cable -8.1% in a day; safe havens bid; front-end GBP rates \
                           reprice 25bp lower."
                .to_string(),
        },
    );
    lib.insert(
        "chf_depeg_2015".to_string(),
        Scenario {
            name: "CHF depeg - SNB floor removal (15 Jan 2015)".to_string(),
            shocks: HashMap::from([
                (fx_factor("CHF").unwrap(), lit_log(0.149)),
                (fx_factor("EUR").unwrap(), lit_log(-0.014)),
                (fx_factor("JPY").unwrap(), lit_log(0.012)),
                (ir_factor("CHF"), -0.0050), // SNB cut to -0.75%
            ]),
            description: "CHF +14.9% vs USD close-to-close (intraday >+30%); the \
                           peg-break archetype - invisible to a 250d HS window."
                .to_string(),
        },
    );
    lib.insert(
        "jpy_1998".to_string(),
        Scenario {
            name: "JPY carry unwind (7-8 Oct 1998)".to_string(),
            shocks: HashMap::from([
                (fx_factor("JPY").unwrap(), lit_log(0.115)),
                (fx_factor("AUD").unwrap(), lit_log(-0.040)),
                (fx_factor("NZD").unwrap(), lit_log(-0.040)),
                (fx_factor("CHF").unwrap(), lit_log(0.020)),
            ]),
            description: "USDJPY 131 -> 117 in two sessions as levered carry unwound."
                .to_string(),
        },
    );
    lib
}

/// Hypothetical broad USD move: USD strengthens by `pct` vs every listed
/// currency (`pct = 0.10` means every CCYUSD falls 10% in simple terms;
/// negative `pct` weakens USD). USD itself is skipped.
///
/// # Errors
/// [`FxVarError::Invalid`] for `pct <= -100%`.
pub fn usd_broad_move(ccys: &[String], pct: f64) -> Result<Scenario> {
    // `!(pct > -1.0)` rather than `pct <= -1.0`: the latter is false for
    // NaN, which would produce a NaN shock on every currency in the book.
    if !(pct > -1.0) || !pct.is_finite() {
        return Err(FxVarError::invalid(format!(
            "pct must be finite and > -100%, got {pct}"
        )));
    }
    let name = format!("USD {}{}% broad move", if pct >= 0.0 { "+" } else { "" }, pct * 100.0);
    let mut shocks = HashMap::new();
    for c in ccys {
        let u = c.to_uppercase();
        if u == "USD" {
            continue;
        }
        // USD +pct vs CCY means CCYUSD falls by pct/(1+pct) in simple terms.
        shocks.insert(fx_factor(&u)?, simple_to_log(-pct / (1.0 + pct))?);
    }
    Ok(Scenario { name, shocks, description: "Uniform USD move against all book currencies.".to_string() })
}

/// Peg-break stress add-on for a pegged/managed currency — the mandatory
/// companion to any HS/parametric VaR on a book holding pegged currencies
/// (the engine's peg-blindness warnings point here). `jump` is the simple
/// revaluation vs USD (`-0.30` = 30% devaluation; positive models a
/// CHF-2015-style upward break); `contagion` adds simple-percentage
/// co-moves for other currencies.
///
/// # Errors
/// [`FxVarError::Invalid`] for `jump <= -100%`.
pub fn peg_break_scenario(ccy: &str, jump: f64, contagion: &HashMap<String, f64>) -> Result<Scenario> {
    if !(jump > -1.0) || !jump.is_finite() {
        return Err(FxVarError::invalid(format!(
            "jump must be finite and > -100%, got {jump}"
        )));
    }
    let u = ccy.to_uppercase();
    let mut shocks = HashMap::new();
    shocks.insert(fx_factor(&u)?, simple_to_log(jump)?);
    for (c, &m) in contagion {
        // Contagion moves are validated too: a -100% contagion entry used
        // to become a silent -inf shock on a currency the caller was not
        // even focused on.
        shocks.insert(fx_factor(c)?, simple_to_log(m)?);
    }
    let direction = if jump < 0.0 { "devaluation" } else { "revaluation" };
    let name = format!("{u} peg break ({}{:.1}% {direction})", if jump >= 0.0 { "+" } else { "" }, jump * 100.0);
    let description = format!(
        "Managed-currency regime break: {u} gaps {}{:.1}% vs USD with no intermediate \
         prints; HS/parametric VaR see none of it.",
        if jump >= 0.0 { "+" } else { "" },
        jump * 100.0
    );
    Ok(Scenario { name, shocks, description })
}

/// One row of a stress report.
#[derive(Debug, Clone, PartialEq)]
pub struct StressRow {
    /// Scenario library key.
    pub key: String,
    /// Scenario display name.
    pub name: String,
    /// Base-ccy P&L (loss negative).
    pub pnl: f64,
    /// Scenario description.
    pub description: String,
}

/// Full-revaluation P&L of `book` under each scenario, sorted worst-first.
/// Shocks on factors the book does not carry are dropped.
///
/// # Errors
/// [`FxVarError::Invalid`] for an empty book.
pub fn run_stress(book: &Book, market: &Market, scenarios: &HashMap<String, Scenario>) -> Result<Vec<StressRow>> {
    let compiled = CompiledBook::new(book, market)?;
    let factors: std::collections::HashSet<&String> = compiled.factors().iter().collect();
    let mut rows = Vec::with_capacity(scenarios.len());
    for (key, sc) in scenarios {
        let filtered: HashMap<String, f64> =
            sc.shocks.iter().filter(|(f, _)| factors.contains(f)).map(|(f, &v)| (f.clone(), v)).collect();
        let pnl = compiled.pnl_map(&filtered)?;
        rows.push(StressRow { key: key.clone(), name: sc.name.clone(), pnl, description: sc.description.clone() });
    }
    // `total_cmp`: a total order on every f64, so the report cannot panic
    // on a scenario that revalues to a non-finite P&L (and such a row
    // sorts deterministically rather than arbitrarily).
    rows.sort_by(|a, b| a.pnl.total_cmp(&b.pnl));
    Ok(rows)
}

// ---------------------------------------------------------------------
// Reverse stress
// ---------------------------------------------------------------------

/// Reverse-stress outcome: the worst-case factor shock and its loss.
#[derive(Debug, Clone, PartialEq)]
pub struct ReverseStress {
    /// Shock vector, aligned with the exposure vector.
    pub shocks: Vec<f64>,
    /// Positive loss at the optimum.
    pub loss: f64,
}

fn checked_sigma_p(w: &[f64], cov: &Matrix) -> Result<f64> {
    // Finiteness first: `NaN.max(0.0)` is 0.0, so a NaN quadratic form
    // would be reported as "zero linear risk" rather than as corrupt data.
    if w.iter().any(|v| !v.is_finite()) {
        return Err(FxVarError::invalid("reverse stress: exposures must all be finite"));
    }
    if !cov.all_finite() {
        return Err(FxVarError::invalid("reverse stress: covariance must be finite"));
    }
    let sp2 = cov.quad_form(w)?;
    let sp = sp2.max(0.0).sqrt();
    if !(sp > 0.0) || !sp.is_finite() {
        return Err(FxVarError::invalid("book has zero linear risk; reverse stress undefined"));
    }
    Ok(sp)
}

/// Closed-form reverse stress for a linear book: among all shocks `dx` with
/// Mahalanobis norm `sqrt(dx' Sigma^-1 dx) <= radius`, the loss `-w'dx` is
/// maximised at `dx* = -radius * Sigma w / sqrt(w' Sigma w)` with loss
/// `radius * sqrt(w' Sigma w)`.
///
/// # Errors
/// [`FxVarError::Invalid`] if the book has zero linear risk or
/// `radius <= 0`.
pub fn reverse_stress_linear(exposures: &[f64], cov: &Matrix, radius: f64) -> Result<ReverseStress> {
    if !(radius > 0.0) || !radius.is_finite() {
        return Err(FxVarError::invalid(format!(
            "radius / loss_target must be finite and positive, got {radius}"
        )));
    }
    let sp = checked_sigma_p(exposures, cov)?;
    let sw = cov.matvec(exposures)?;
    let shocks: Vec<f64> = sw.iter().map(|&v| -radius * v / sp).collect();
    Ok(ReverseStress { shocks, loss: radius * sp })
}

/// As [`reverse_stress_linear`] but solving for the radius that produces
/// `loss_target`: `k = loss_target / sqrt(w' Sigma w)`.
///
/// # Errors
/// Same as [`reverse_stress_linear`].
pub fn reverse_stress_for_loss(exposures: &[f64], cov: &Matrix, loss_target: f64) -> Result<ReverseStress> {
    let sp = checked_sigma_p(exposures, cov)?;
    reverse_stress_linear(exposures, cov, loss_target / sp)
}

/// Independent numerical check of the closed form: maximises the linear
/// loss over the Mahalanobis ellipsoid by projected gradient ascent in
/// whitened coordinates (`dx = L y`, `|y| = radius`) from a seeded random
/// start — it never assumes the analytic optimum. Used in tests to confirm
/// [`reverse_stress_linear`] to `1e-6`.
///
/// # Errors
/// [`FxVarError::Invalid`] if `radius <= 0` or the book has zero linear
/// risk; [`FxVarError::Numerical`] if the covariance cannot be factorised.
pub fn reverse_stress_numerical(exposures: &[f64], cov: &Matrix, radius: f64, seed: u64) -> Result<ReverseStress> {
    if !(radius > 0.0) || !radius.is_finite() {
        return Err(FxVarError::invalid(format!(
            "radius must be finite and positive, got {radius}"
        )));
    }
    checked_sigma_p(exposures, cov)?;
    let n = exposures.len();

    let (lower, _) = cov.cholesky_with_jitter(8)?;

    let loss_of = |y: &[f64]| -> f64 {
        let mut loss = 0.0;
        for i in 0..n {
            let li = lower.row(i);
            let mut dxi = 0.0;
            for (j, &lij) in li.iter().enumerate().take(i + 1) {
                dxi += lij * y[j];
            }
            loss -= exposures[i] * dxi;
        }
        loss
    };
    let project = |y: &mut [f64]| {
        let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in y.iter_mut() {
                *v *= radius / norm;
            }
        }
    };

    let mut rng = Rng::new(seed);
    let mut y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    project(&mut y);
    if loss_of(&y) < 0.0 {
        for v in y.iter_mut() {
            *v = -*v;
        }
    }

    let h = 1e-7 * radius;
    let mut step = radius;
    let mut best = loss_of(&y);
    let mut grad = vec![0.0; n];
    for _ in 0..200 {
        let mut trial = y.clone();
        for j in 0..n {
            trial[j] = y[j] + h;
            let up = loss_of(&trial);
            trial[j] = y[j] - h;
            let dn = loss_of(&trial);
            trial[j] = y[j];
            grad[j] = (up - dn) / (2.0 * h);
        }
        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm == 0.0 {
            break;
        }
        let mut improved = false;
        while step > 1e-14 * radius {
            for j in 0..n {
                trial[j] = y[j] + step * radius * grad[j] / gnorm;
            }
            project(&mut trial);
            let cand = loss_of(&trial);
            if cand > best {
                y = trial.clone();
                best = cand;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }

    let mut shocks = vec![0.0; n];
    for i in 0..n {
        let li = lower.row(i);
        let mut dxi = 0.0;
        for (j, &lij) in li.iter().enumerate().take(i + 1) {
            dxi += lij * y[j];
        }
        shocks[i] = dxi;
    }
    Ok(ReverseStress { shocks, loss: best })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::{Position, SpotPosition};

    fn test_market() -> Market {
        Market::new(
            [("EUR", 1.10), ("GBP", 1.27), ("JPY", 0.0090), ("CHF", 1.12), ("HKD", 0.1282)],
            [("USD", 0.050), ("EUR", 0.030), ("GBP", 0.045), ("JPY", 0.001), ("CHF", -0.005)],
        )
        .unwrap()
    }

    #[test]
    fn brexit_scenario_hits_long_cable() {
        let m = test_market();
        let book = Book::new(vec![Position::Spot(SpotPosition::new("GBPUSD", 10e6, None))], "USD").unwrap();
        let lib = historical_scenarios();
        assert!(lib.contains_key("brexit_2016"));
        let rows = run_stress(&book, &m, &lib).unwrap();
        let row = rows.iter().find(|r| r.key == "brexit_2016").unwrap();
        let expect = 10e6 * 1.27 * ((-0.081_f64).ln_1p().exp() - 1.0);
        assert!((row.pnl - expect).abs() < 1e-6 * expect.abs());
        for w in rows.windows(2) {
            assert!(w[0].pnl <= w[1].pnl);
        }
    }

    #[test]
    fn chf_depeg_hits_short_chf() {
        let m = test_market();
        let book = Book::new(vec![Position::Spot(SpotPosition::new("USDCHF", 20e6, None))], "USD").unwrap();
        let lib = historical_scenarios();
        let rows = run_stress(&book, &m, &lib).unwrap();
        let row = rows.iter().find(|r| r.key == "chf_depeg_2015").unwrap();
        assert!(row.pnl < -2e6);
    }

    #[test]
    fn joint_fx_rate_shock_moves_forward_through_both_legs() {
        let m = test_market();
        let fwd = Book::new(vec![Position::Forward(crate::book::ForwardPosition::new("GBPUSD", 10e6, 1.0, None))], "USD").unwrap();
        let lib = historical_scenarios();
        let brexit = &lib["brexit_2016"];
        let cb = CompiledBook::new(&fwd, &m).unwrap();
        let joint = cb.pnl_map(&brexit.shocks).unwrap();
        let mut fx_only = brexit.shocks.clone();
        fx_only.remove(&ir_factor("GBP"));
        let fx_leg = cb.pnl_map(&fx_only).unwrap();
        assert!((joint - fx_leg).abs() > 1e3);
        assert!(joint < 0.0);
    }

    #[test]
    fn usd_broad_move_and_peg_break() {
        let m = test_market();
        let usd10 = usd_broad_move(&["EUR".to_string(), "JPY".to_string(), "USD".to_string()], 0.10).unwrap();
        assert_eq!(usd10.shocks.len(), 2);
        assert!((usd10.shocks["FX:EUR"] - (-0.10_f64 / 1.10).ln_1p()).abs() < 1e-15);
        let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 11e6, None))], "USD").unwrap();
        let cb = CompiledBook::new(&book, &m).unwrap();
        let pnl = cb.pnl_map(&usd10.shocks).unwrap();
        assert!((pnl - 11e6 * 1.10 * (-0.10 / 1.10)).abs() < 1e-6 * pnl.abs());

        let contagion = HashMap::from([("CHF".to_string(), -0.05)]);
        let pb = peg_break_scenario("HKD", -0.30, &contagion).unwrap();
        assert!((pb.shocks["FX:HKD"] - (-0.30_f64).ln_1p()).abs() < 1e-15);
        assert!((pb.shocks["FX:CHF"] - (-0.05_f64).ln_1p()).abs() < 1e-15);
        assert!(peg_break_scenario("HKD", -1.0, &HashMap::new()).is_err());
        assert!(usd_broad_move(&["EUR".to_string()], -1.0).is_err());
    }

    #[test]
    fn reverse_stress_closed_form_loss_and_direction() {
        let w = vec![11e6, -4.5e6, -2.4e6];
        let cov = Matrix::from_rows(&[
            vec![3.6e-5, 1.1e-5, -2.0e-6],
            vec![1.1e-5, 4.9e-5, -1.0e-6],
            vec![-2.0e-6, -1.0e-6, 2.5e-7],
        ])
        .unwrap();
        let sp = cov.quad_form(&w).unwrap().sqrt();
        let k = 3.0;
        let rs = reverse_stress_linear(&w, &cov, k).unwrap();
        assert!((rs.loss - k * sp).abs() < 1e-12 * rs.loss);
        let mut loss = 0.0;
        for i in 0..w.len() {
            loss -= w[i] * rs.shocks[i];
        }
        assert!((loss - rs.loss).abs() < 1e-10 * rs.loss);
        let rs2 = reverse_stress_for_loss(&w, &cov, 1e6).unwrap();
        assert!((rs2.loss - 1e6).abs() < 1e-9 * 1e6);
    }

    #[test]
    fn reverse_stress_numerical_search_confirms_closed_form() {
        let w = vec![11e6, -4.5e6, -2.4e6];
        let cov = Matrix::from_rows(&[
            vec![3.6e-5, 1.1e-5, -2.0e-6],
            vec![1.1e-5, 4.9e-5, -1.0e-6],
            vec![-2.0e-6, -1.0e-6, 2.5e-7],
        ])
        .unwrap();
        let k = 2.5;
        let closed = reverse_stress_linear(&w, &cov, k).unwrap();
        let numeric = reverse_stress_numerical(&w, &cov, k, 3).unwrap();
        assert!((numeric.loss - closed.loss).abs() < 1e-6 * closed.loss);
        for i in 0..w.len() {
            assert!((numeric.shocks[i] - closed.shocks[i]).abs() < 1e-4 * (closed.shocks[i].abs() + 1e-6));
        }
    }

    #[test]
    fn zero_risk_book_is_rejected() {
        let cov = Matrix::from_rows(&[vec![1e-4]]).unwrap();
        assert!(reverse_stress_linear(&[0.0], &cov, 1.0).is_err());
        assert!(reverse_stress_linear(&[1e6], &cov, 0.0).is_err());
        let m = test_market();
        assert!(run_stress(&Book::empty(), &m, &historical_scenarios()).is_err());
    }
}
