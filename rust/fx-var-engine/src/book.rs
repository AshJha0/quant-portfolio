//! Multi-currency FX book: positions, base-currency P&L, triangulation.
//!
//! Mirrors `python/fx/03-var-es-engine` `fx_var.book` and C++
//! `fxvar::Book`/`CompiledBook` (spot + forward + cash subset; options stay
//! in the Python research stack).
//!
//! # Factor representation
//!
//! Every currency is represented by its USD factor: `"FX:CCY"` is the daily
//! log return of the USD price of 1 unit of CCY (the log return of
//! CCYUSD). USD has no FX factor — its USD price is identically 1. A
//! position in a *cross* pair (EURJPY) is decomposed into its two USD legs
//! — long EUR, short JPY — so cross risk is triangulated by construction
//! and the factor set is arbitrage-consistent. `"IR:CCY"` is an absolute
//! shock (decimal p.a.) to the continuously compounded ACT/365 zero rate
//! (flat curve per currency).
//!
//! # Position types
//!
//! * [`CashPosition`] — a currency balance; riskless when denominated in
//!   the book's base currency.
//! * [`SpotPosition`] — long `notional` of the pair's base ccy vs the
//!   quote ccy at `entry_rate` (defaults to the reference market's cross
//!   rate = zero initial value).
//! * [`ForwardPosition`] — outright forward as spot + two deposit legs
//!   (CIP): value in USD is `N e^{-r_f T} S_base - N K e^{-r_d T} S_quote`,
//!   so forward points expose the position to both currencies'
//!   interest-rate factors.
//!
//! # P&L convention
//!
//! P&L is profit (+) / loss (-) in the book's base currency:
//! `PnL = V1_usd / S1_base - V0_usd / S0_base`, with the base currency's
//! own USD price shocked consistently. A pure base-ccy cash balance
//! therefore carries exactly zero risk.
//!
//! # Hot path
//!
//! [`CompiledBook`] flattens the book into a struct-of-arrays leg list (one
//! `exp()` per leg per scenario), which is what historical / Monte Carlo
//! revaluation iterates over. See `src/bin/bench.rs` for throughput.

use std::collections::HashMap;

use crate::returns::ReturnsMatrix;
use crate::{FxVarError, Result};

/// `(base, quote)` legs of a 6-letter pair, upper-cased.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairLegs {
    /// Base currency (e.g. `"EUR"` in `EURUSD`).
    pub base: String,
    /// Quote currency (e.g. `"USD"` in `EURUSD`).
    pub quote: String,
}

/// Split `"EURUSD"` -> `{base: "EUR", quote: "USD"}` (USD per 1 EUR).
///
/// # Errors
/// [`FxVarError::Invalid`] unless `pair` is 6 alphabetic characters with
/// distinct legs.
pub fn split_pair(pair: &str) -> Result<PairLegs> {
    if pair.len() != 6 || !pair.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err(FxVarError::invalid(format!(
            "FX pair must be 6 letters like 'EURUSD', got '{pair}'"
        )));
    }
    let base = pair[..3].to_uppercase();
    let quote = pair[3..].to_uppercase();
    if base == quote {
        return Err(FxVarError::invalid(format!("FX pair has identical legs: '{pair}'")));
    }
    Ok(PairLegs { base, quote })
}

/// Factor name for the log return of CCYUSD.
///
/// # Errors
/// [`FxVarError::Invalid`] for USD (the pivot has no FX factor).
pub fn fx_factor(ccy: &str) -> Result<String> {
    let c = ccy.to_uppercase();
    if c == "USD" {
        return Err(FxVarError::invalid("USD has no FX factor: its USD price is identically 1"));
    }
    Ok(format!("FX:{c}"))
}

/// Factor name for an absolute shock to CCY's cc zero rate (ACT/365).
pub fn ir_factor(ccy: &str) -> String {
    format!("IR:{}", ccy.to_uppercase())
}

/// Point-in-time market snapshot.
///
/// `spot_usd` maps ccy -> USD price of 1 unit (`spot_usd["EUR"] = 1.08`
/// means EURUSD = 1.08); `"USD"` is implied at 1.0 and may be omitted (if
/// present it must equal 1.0). `rates` maps ccy -> continuously compounded
/// zero rate, annualised, ACT/365 (flat curve per currency).
#[derive(Clone, Debug, PartialEq)]
pub struct Market {
    spot_usd: HashMap<String, f64>,
    rates: HashMap<String, f64>,
}

impl Market {
    /// Build a [`Market`] from `(ccy, usd_price)` and `(ccy, rate)` pairs.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if any spot is non-positive/non-finite, or
    /// `spot_usd["USD"]` is present and not 1.0.
    pub fn new<I, J, S1, S2>(spot_usd: I, rates: J) -> Result<Self>
    where
        I: IntoIterator<Item = (S1, f64)>,
        J: IntoIterator<Item = (S2, f64)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        let mut spots: HashMap<String, f64> = HashMap::new();
        for (c, s) in spot_usd {
            let c = c.into().to_uppercase();
            if !s.is_finite() || s <= 0.0 {
                return Err(FxVarError::invalid(format!(
                    "spot_usd[{c}] must be a positive number, got {s}"
                )));
            }
            spots.insert(c, s);
        }
        if let Some(&usd) = spots.get("USD") {
            if (usd - 1.0).abs() > 1e-12 {
                return Err(FxVarError::invalid("spot_usd['USD'] must be 1.0 (USD per USD)"));
            }
        }
        spots.insert("USD".to_string(), 1.0);
        let mut rate_map = HashMap::new();
        for (c, r) in rates {
            rate_map.insert(c.into().to_uppercase(), r);
        }
        Ok(Market { spot_usd: spots, rates: rate_map })
    }

    /// USD price of 1 unit of `ccy` (1.0 for USD).
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if `ccy` is unknown.
    pub fn spot(&self, ccy: &str) -> Result<f64> {
        self.spot_usd
            .get(&ccy.to_uppercase())
            .copied()
            .ok_or_else(|| FxVarError::invalid(format!("no USD spot for currency '{ccy}' in Market")))
    }

    /// cc zero rate for `ccy`.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if `ccy` is unknown.
    pub fn rate(&self, ccy: &str) -> Result<f64> {
        self.rates
            .get(&ccy.to_uppercase())
            .copied()
            .ok_or_else(|| FxVarError::invalid(format!("no interest rate for currency '{ccy}' in Market")))
    }

    /// Cross rate of `pair` (QUOTE per 1 BASE) by USD triangulation.
    pub fn cross(&self, pair: &str) -> Result<f64> {
        let legs = split_pair(pair)?;
        Ok(self.spot(&legs.base)? / self.spot(&legs.quote)?)
    }

    /// CIP forward: `F = X * exp((r_d - r_f) * T)` (QUOTE per BASE).
    pub fn forward(&self, pair: &str, expiry: f64) -> Result<f64> {
        let legs = split_pair(pair)?;
        let x = self.cross(pair)?;
        Ok(x * ((self.rate(&legs.quote)? - self.rate(&legs.base)?) * expiry).exp())
    }
}

/// A cash balance of `amount` units of `ccy`.
#[derive(Clone, Debug, PartialEq)]
pub struct CashPosition {
    /// 3-letter currency code.
    pub ccy: String,
    /// Signed amount in units of `ccy` (negative = short).
    pub amount: f64,
}

impl CashPosition {
    /// Build a cash position of `amount` units of `ccy`.
    pub fn new(ccy: impl Into<String>, amount: f64) -> Self {
        CashPosition { ccy: ccy.into(), amount }
    }
}

/// Spot FX position: long `notional` of BASE ccy vs QUOTE at `entry_rate`
/// (units of the pair's base ccy; negative = short). `entry_rate = None`
/// means struck at the reference market's cross rate (zero initial value).
#[derive(Clone, Debug, PartialEq)]
pub struct SpotPosition {
    /// 6-letter pair, e.g. `"EURUSD"`.
    pub pair: String,
    /// Signed notional in units of the base currency.
    pub notional: f64,
    /// Entry rate (quote per base); `None` resolves to the reference
    /// market's cross rate.
    pub entry_rate: Option<f64>,
}

impl SpotPosition {
    /// Build a spot FX position.
    pub fn new(pair: impl Into<String>, notional: f64, entry_rate: Option<f64>) -> Self {
        SpotPosition { pair: pair.into(), notional, entry_rate }
    }
}

/// Outright FX forward: long `notional` BASE at `strike`, expiry in years.
/// `strike = None` resolves to the ATM CIP forward of the reference market.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardPosition {
    /// 6-letter pair, e.g. `"EURUSD"`.
    pub pair: String,
    /// Signed notional in units of the base currency.
    pub notional: f64,
    /// Expiry in years (>= 0).
    pub expiry: f64,
    /// Strike (quote per base); `None` resolves to the ATM CIP forward.
    pub strike: Option<f64>,
}

impl ForwardPosition {
    /// Build an outright forward position.
    pub fn new(pair: impl Into<String>, notional: f64, expiry: f64, strike: Option<f64>) -> Self {
        ForwardPosition { pair: pair.into(), notional, expiry, strike }
    }
}

/// A book position: cash, spot FX, or outright forward.
#[derive(Clone, Debug, PartialEq)]
pub enum Position {
    /// See [`CashPosition`].
    Cash(CashPosition),
    /// See [`SpotPosition`].
    Spot(SpotPosition),
    /// See [`ForwardPosition`].
    Forward(ForwardPosition),
}

fn validate_position(p: &Position) -> Result<()> {
    match p {
        Position::Cash(c) => {
            if c.ccy.len() != 3 {
                return Err(FxVarError::invalid(format!(
                    "cash ccy must be a 3-letter code, got '{}'",
                    c.ccy
                )));
            }
            Ok(())
        }
        Position::Spot(s) => {
            split_pair(&s.pair)?;
            if let Some(r) = s.entry_rate {
                if r <= 0.0 {
                    return Err(FxVarError::invalid("entry_rate must be > 0"));
                }
            }
            Ok(())
        }
        Position::Forward(fw) => {
            split_pair(&fw.pair)?;
            if fw.expiry < 0.0 {
                return Err(FxVarError::invalid("expiry must be >= 0"));
            }
            if let Some(k) = fw.strike {
                if k <= 0.0 {
                    return Err(FxVarError::invalid("strike must be > 0"));
                }
            }
            Ok(())
        }
    }
}

fn position_currencies(p: &Position) -> Result<(String, String)> {
    match p {
        Position::Cash(c) => {
            let u = c.ccy.to_uppercase();
            Ok((u.clone(), u))
        }
        Position::Spot(s) => {
            let legs = split_pair(&s.pair)?;
            Ok((legs.base, legs.quote))
        }
        Position::Forward(fw) => {
            let legs = split_pair(&fw.pair)?;
            Ok((legs.base, legs.quote))
        }
    }
}

/// A multi-currency book with a designated base (reporting) currency.
#[derive(Clone, Debug, PartialEq)]
pub struct Book {
    positions: Vec<Position>,
    base: String,
}

impl Book {
    /// Build a book from `positions` and a `base` reporting currency.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if any position is malformed.
    pub fn new(positions: Vec<Position>, base: impl Into<String>) -> Result<Self> {
        for p in &positions {
            validate_position(p)?;
        }
        Ok(Book { positions, base: base.into().to_uppercase() })
    }

    /// Empty book with base currency `"USD"`.
    pub fn empty() -> Self {
        Book { positions: Vec::new(), base: "USD".to_string() }
    }

    /// The book's base (reporting) currency.
    pub fn base(&self) -> &str {
        &self.base
    }

    /// This book's positions.
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// True if the book has no positions.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Append a position, validating it first.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if the position is malformed.
    pub fn add(&mut self, p: Position) -> Result<()> {
        validate_position(&p)?;
        self.positions.push(p);
        Ok(())
    }

    /// All currencies the book touches, including the base ccy (sorted).
    pub fn currencies(&self) -> Result<Vec<String>> {
        let mut set = std::collections::BTreeSet::new();
        set.insert(self.base.clone());
        for p in &self.positions {
            let (b, q) = position_currencies(p)?;
            set.insert(b);
            set.insert(q);
        }
        Ok(set.into_iter().collect())
    }

    /// Sorted risk-factor names the book is exposed to: `FX:*` for every
    /// non-USD currency involved (incl. base if not USD), then `IR:*` for
    /// the forwards' leg currencies.
    pub fn factors(&self) -> Result<Vec<String>> {
        let mut fx = std::collections::BTreeSet::new();
        let mut ir = std::collections::BTreeSet::new();
        for c in self.currencies()? {
            if c != "USD" {
                fx.insert(fx_factor(&c)?);
            }
        }
        for p in &self.positions {
            if let Position::Forward(fw) = p {
                let legs = split_pair(&fw.pair)?;
                ir.insert(ir_factor(&legs.base));
                ir.insert(ir_factor(&legs.quote));
            }
        }
        let mut out: Vec<String> = fx.into_iter().collect();
        out.extend(ir);
        Ok(out)
    }
}

/// Struct-of-arrays compiled book: the revaluation hot path.
///
/// Each position is flattened into discounted USD legs with unshocked
/// value `v0 = amount * spot0 * exp(-r0 T)`; a scenario revalues each leg
/// as `v0 * exp(dfx - T * dr)`, one `exp()` per leg. Construction resolves
/// default entry rates / strikes against the reference market.
///
/// # Errors
/// [`FxVarError::Invalid`] for an empty book (a VaR on nothing is a
/// configuration error, not a zero).
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledBook {
    value0: Vec<f64>,
    fx_idx: Vec<i64>,
    ir_idx: Vec<i64>,
    neg_expiry: Vec<f64>,
    factors: Vec<String>,
    base: String,
    base_fx_idx: i64,
    v0_usd: f64,
    s0_base: f64,
}

impl CompiledBook {
    /// Compile `book` against `market`.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] for an empty book or unresolvable market
    /// data (missing spot/rate).
    pub fn new(book: &Book, market: &Market) -> Result<Self> {
        if book.is_empty() {
            return Err(FxVarError::invalid(
                "cannot compile an empty book: add positions before running VaR",
            ));
        }
        let factors = book.factors()?;
        let base = book.base().to_string();
        let s0_base = market.spot(&base)?;

        let factor_index =
            |f: &str, factors: &[String]| -> i64 { factors.iter().position(|x| x == f).map(|i| i as i64).unwrap_or(-1) };
        let base_fx_idx =
            if base == "USD" { -1 } else { factor_index(&fx_factor(&base)?, &factors) };

        let mut cb = CompiledBook {
            value0: Vec::new(),
            fx_idx: Vec::new(),
            ir_idx: Vec::new(),
            neg_expiry: Vec::new(),
            factors,
            base,
            base_fx_idx,
            v0_usd: 0.0,
            s0_base,
        };

        let add_leg = |cb: &mut CompiledBook, amount: f64, ccy: &str, rate0: f64, expiry: f64| -> Result<()> {
            let df = if expiry > 0.0 { (-rate0 * expiry).exp() } else { 1.0 };
            cb.value0.push(amount * market.spot(ccy)? * df);
            cb.fx_idx.push(if ccy == "USD" { -1 } else { factor_index(&fx_factor(ccy)?, &cb.factors) });
            cb.ir_idx.push(if expiry > 0.0 { factor_index(&ir_factor(ccy), &cb.factors) } else { -1 });
            cb.neg_expiry.push(if expiry > 0.0 { -expiry } else { 0.0 });
            Ok(())
        };

        for p in book.positions() {
            match p {
                Position::Cash(c) => {
                    add_leg(&mut cb, c.amount, &c.ccy.to_uppercase(), 0.0, 0.0)?;
                }
                Position::Spot(s) => {
                    let legs = split_pair(&s.pair)?;
                    let x0 = match s.entry_rate {
                        Some(r) => r,
                        None => market.cross(&s.pair)?,
                    };
                    add_leg(&mut cb, s.notional, &legs.base, 0.0, 0.0)?;
                    add_leg(&mut cb, -s.notional * x0, &legs.quote, 0.0, 0.0)?;
                }
                Position::Forward(fw) => {
                    let legs = split_pair(&fw.pair)?;
                    let k = match fw.strike {
                        Some(k) => k,
                        None => market.forward(&fw.pair, fw.expiry)?,
                    };
                    if fw.expiry > 0.0 {
                        add_leg(&mut cb, fw.notional, &legs.base, market.rate(&legs.base)?, fw.expiry)?;
                        add_leg(&mut cb, -fw.notional * k, &legs.quote, market.rate(&legs.quote)?, fw.expiry)?;
                    } else {
                        // Expired forward = spot difference vs strike.
                        add_leg(&mut cb, fw.notional, &legs.base, 0.0, 0.0)?;
                        add_leg(&mut cb, -fw.notional * k, &legs.quote, 0.0, 0.0)?;
                    }
                }
            }
        }
        cb.v0_usd = cb.value0.iter().sum();
        Ok(cb)
    }

    /// Factor order every shock vector must follow.
    pub fn factors(&self) -> &[String] {
        &self.factors
    }

    /// Unshocked book value in USD.
    pub fn value0_usd(&self) -> f64 {
        self.v0_usd
    }

    /// Book value in USD for one scenario; `shocks` has `factors().len()`
    /// entries aligned with [`CompiledBook::factors`] (FX = log returns,
    /// IR = absolute).
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] on a shock-vector length mismatch.
    pub fn value_usd(&self, shocks: &[f64]) -> Result<f64> {
        if shocks.len() != self.factors.len() {
            return Err(FxVarError::invalid(format!(
                "shock vector has {} entries, expected {} (factors().len())",
                shocks.len(),
                self.factors.len()
            )));
        }
        let mut total = 0.0;
        for i in 0..self.value0.len() {
            let mut shift = 0.0;
            if self.fx_idx[i] >= 0 {
                shift += shocks[self.fx_idx[i] as usize];
            }
            if self.ir_idx[i] >= 0 {
                shift += self.neg_expiry[i] * shocks[self.ir_idx[i] as usize];
            }
            total += if shift == 0.0 { self.value0[i] } else { self.value0[i] * shift.exp() };
        }
        Ok(total)
    }

    /// Base-ccy P&L of one scenario (shocks aligned with
    /// [`CompiledBook::factors`]).
    pub fn pnl(&self, shocks: &[f64]) -> Result<f64> {
        let v1 = self.value_usd(shocks)?;
        let s1_base = if self.base_fx_idx >= 0 {
            self.s0_base * shocks[self.base_fx_idx as usize].exp()
        } else {
            self.s0_base
        };
        Ok(v1 / s1_base - self.v0_usd / self.s0_base)
    }

    /// Base-ccy P&L of one scenario given as a factor->shock map. Shocks
    /// for factors the book does not carry are ignored (one scenario
    /// library serves every book).
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if `shocks` contains `"FX:USD"` — USD is the
    /// pivot, shock the other leg(s).
    pub fn pnl_map(&self, shocks: &HashMap<String, f64>) -> Result<f64> {
        if shocks.contains_key("FX:USD") {
            return Err(FxVarError::invalid(
                "shock to 'FX:USD' is not a valid factor: USD is the pivot (its USD price \
                 is identically 1). Shock the other leg(s).",
            ));
        }
        let aligned: Vec<f64> =
            self.factors.iter().map(|f| shocks.get(f).copied().unwrap_or(0.0)).collect();
        self.pnl(&aligned)
    }

    /// Base-ccy P&L of every scenario row. `scenarios` must contain every
    /// factor in [`CompiledBook::factors`] (extra columns are ignored).
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if a required factor column is missing.
    pub fn pnl_scenarios(&self, scenarios: &ReturnsMatrix) -> Result<Vec<f64>> {
        let mut missing = Vec::new();
        let factor_col: Vec<i64> = self
            .factors
            .iter()
            .map(|f| {
                scenarios.column_index(f).map(|j| j as i64).unwrap_or_else(|| {
                    missing.push(f.clone());
                    -1
                })
            })
            .collect();
        if !missing.is_empty() {
            return Err(FxVarError::invalid(format!(
                "scenario matrix is missing required factor columns: {missing:?}"
            )));
        }
        let fx_col: Vec<i64> = self
            .fx_idx
            .iter()
            .map(|&i| if i >= 0 { factor_col[i as usize] } else { -1 })
            .collect();
        let ir_col: Vec<i64> = self
            .ir_idx
            .iter()
            .map(|&i| if i >= 0 { factor_col[i as usize] } else { -1 })
            .collect();
        let base_col = if self.base_fx_idx >= 0 { factor_col[self.base_fx_idx as usize] } else { -1 };

        let n_scen = scenarios.n_obs();
        let n_legs = self.value0.len();
        let v0_base = self.v0_usd / self.s0_base;
        let mut out = vec![0.0; n_scen];
        for s in 0..n_scen {
            let row = scenarios.data.row(s);
            let mut v1 = 0.0;
            for i in 0..n_legs {
                let mut shift = 0.0;
                if fx_col[i] >= 0 {
                    shift += row[fx_col[i] as usize];
                }
                if ir_col[i] >= 0 {
                    shift += self.neg_expiry[i] * row[ir_col[i] as usize];
                }
                v1 += if shift == 0.0 { self.value0[i] } else { self.value0[i] * shift.exp() };
            }
            let s1_base = if base_col >= 0 { self.s0_base * row[base_col as usize].exp() } else { self.s0_base };
            out[s] = v1 / s1_base - v0_base;
        }
        Ok(out)
    }

    /// Delta exposures `dPnL/dfactor` by central finite differences,
    /// aligned with [`CompiledBook::factors`]. Units: base-ccy P&L per unit
    /// factor move — for `FX:*` per unit log return (base-ccy notional
    /// exposure), for `IR:*` per 1.00 of rate. These are the mapping
    /// weights used by the variance-covariance method and the
    /// reverse-stress closed form.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] if `bump <= 0`.
    pub fn linear_exposures(&self, bump: f64) -> Result<Vec<f64>> {
        if !(bump > 0.0) {
            return Err(FxVarError::invalid("bump must be > 0"));
        }
        let n = self.factors.len();
        let mut w = vec![0.0; n];
        let mut shocks = vec![0.0; n];
        for j in 0..n {
            shocks[j] = bump;
            let up = self.pnl(&shocks)?;
            shocks[j] = -bump;
            let dn = self.pnl(&shocks)?;
            shocks[j] = 0.0;
            w[j] = (up - dn) / (2.0 * bump);
        }
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_market() -> Market {
        Market::new(
            [("EUR", 1.10), ("JPY", 0.0090), ("GBP", 1.27), ("CHF", 1.12)],
            [("USD", 0.050), ("EUR", 0.030), ("JPY", 0.001), ("GBP", 0.045)],
        )
        .unwrap()
    }

    #[test]
    fn split_and_factors() {
        let legs = split_pair("eurusd").unwrap();
        assert_eq!(legs.base, "EUR");
        assert_eq!(legs.quote, "USD");
        assert_eq!(fx_factor("jpy").unwrap(), "FX:JPY");
        assert_eq!(ir_factor("usd"), "IR:USD");
        assert!(split_pair("EUR").is_err());
        assert!(split_pair("EUREUR").is_err());
        assert!(split_pair("EUR US").is_err());
        assert!(fx_factor("USD").is_err());
    }

    #[test]
    fn market_spot_cross_forward_and_validation() {
        let m = test_market();
        assert_eq!(m.spot("USD").unwrap(), 1.0);
        assert_eq!(m.spot("EUR").unwrap(), 1.10);
        assert!((m.cross("EURJPY").unwrap() - 1.10 / 0.0090).abs() < 1e-12);
        assert!((m.forward("EURUSD", 1.0).unwrap() - 1.10 * (0.020_f64).exp()).abs() < 1e-12);
        assert!(m.spot("XXX").is_err());
        assert!(m.rate("CHF").is_err());
        assert!(Market::new([("EUR", -1.0)], std::iter::empty::<(&str, f64)>()).is_err());
        assert!(Market::new([("USD", 1.05)], std::iter::empty::<(&str, f64)>()).is_err());
    }

    #[test]
    fn book_factor_enumeration() {
        let book = Book::new(
            vec![
                Position::Spot(SpotPosition::new("EURUSD", 1e6, None)),
                Position::Forward(ForwardPosition::new("USDJPY", 2e6, 0.5, None)),
            ],
            "USD",
        )
        .unwrap();
        let f = book.factors().unwrap();
        assert_eq!(f, vec!["FX:EUR", "FX:JPY", "IR:JPY", "IR:USD"]);
    }

    #[test]
    fn triangulation_identity_eurjpy() {
        let m = test_market();
        let n = 7_000_000.0;
        let cross = Book::new(vec![Position::Spot(SpotPosition::new("EURJPY", n, None))], "USD").unwrap();
        let legs = Book::new(
            vec![
                Position::Spot(SpotPosition::new("EURUSD", n, None)),
                Position::Spot(SpotPosition::new("USDJPY", n * m.cross("EURUSD").unwrap(), None)),
            ],
            "USD",
        )
        .unwrap();
        let cb_cross = CompiledBook::new(&cross, &m).unwrap();
        let cb_legs = CompiledBook::new(&legs, &m).unwrap();
        let mut shocks = HashMap::new();
        shocks.insert("FX:EUR".to_string(), 0.013);
        shocks.insert("FX:JPY".to_string(), -0.021);
        let p_cross = cb_cross.pnl_map(&shocks).unwrap();
        let p_legs = cb_legs.pnl_map(&shocks).unwrap();
        assert!((p_cross - p_legs).abs() < 1e-12 * p_cross.abs() + 1e-12);
        assert!(p_cross.abs() > 1.0);
    }

    #[test]
    fn forward_zero_value_at_inception_and_cip() {
        let m = test_market();
        let atm = Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 5e6, 0.75, None))], "USD").unwrap();
        let cb = CompiledBook::new(&atm, &m).unwrap();
        assert!(cb.value0_usd().abs() < 1e-10 * 5e6);

        let k = 1.08;
        let struck = Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 5e6, 0.75, Some(k)))], "USD").unwrap();
        let cb2 = CompiledBook::new(&struck, &m).unwrap();
        let expect = 5e6 * (-0.030_f64 * 0.75).exp() * 1.10 - 5e6 * k * (-0.050_f64 * 0.75).exp() * 1.0;
        assert!((cb2.value0_usd() - expect).abs() < 1e-10 * expect.abs());
    }

    #[test]
    fn forward_matches_two_leg_deposit_decomposition() {
        let m = test_market();
        let (n, t) = (5e6, 0.5);
        let k = m.forward("EURUSD", t).unwrap();
        let fwd = Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", n, t, Some(k)))], "USD").unwrap();
        let cash = Book::new(
            vec![
                Position::Cash(CashPosition::new("EUR", n * (-0.030_f64 * t).exp())),
                Position::Cash(CashPosition::new("USD", -n * k * (-0.050_f64 * t).exp())),
            ],
            "USD",
        )
        .unwrap();
        let cb_f = CompiledBook::new(&fwd, &m).unwrap();
        let cb_c = CompiledBook::new(&cash, &m).unwrap();
        let mut shocks = HashMap::new();
        shocks.insert("FX:EUR".to_string(), -0.045);
        let pf = cb_f.pnl_map(&shocks).unwrap();
        let pc = cb_c.pnl_map(&shocks).unwrap();
        assert!((pf - pc).abs() < 1e-10 * pf.abs() + 1e-10);
    }

    #[test]
    fn forward_rate_leg_sensitivities() {
        let m = test_market();
        let (n, t) = (5e6, 0.5);
        let k = m.forward("EURUSD", t).unwrap();
        let fwd = Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", n, t, Some(k)))], "USD").unwrap();
        let cb = CompiledBook::new(&fwd, &m).unwrap();
        let factors = cb.factors().to_vec();
        let w = cb.linear_exposures(1e-6).unwrap();
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[1], "IR:EUR");
        assert_eq!(factors[2], "IR:USD");
        let dv_dr_eur = -t * n * (-0.030_f64 * t).exp() * 1.10;
        let dv_dr_usd = t * n * k * (-0.050_f64 * t).exp();
        assert!((w[1] - dv_dr_eur).abs() < 1e-4 * dv_dr_eur.abs());
        assert!((w[2] - dv_dr_usd).abs() < 1e-4 * dv_dr_usd.abs());
    }

    #[test]
    fn base_currency_position_has_zero_risk() {
        let m = test_market();
        let book = Book::new(vec![Position::Cash(CashPosition::new("GBP", 25e6))], "GBP").unwrap();
        let cb = CompiledBook::new(&book, &m).unwrap();
        for shock in [-0.20, -0.05, 0.0, 0.07, 0.30] {
            let mut shocks = HashMap::new();
            shocks.insert("FX:GBP".to_string(), shock);
            let p = cb.pnl_map(&shocks).unwrap();
            assert!(p.abs() < 1e-9 * 25e6);
        }
    }

    #[test]
    fn non_usd_base_pnl_convention() {
        let m = test_market();
        let a = 10e6;
        let book = Book::new(vec![Position::Cash(CashPosition::new("USD", a))], "EUR").unwrap();
        let cb = CompiledBook::new(&book, &m).unwrap();
        let shock: f64 = 0.02;
        let (s0, s1) = (1.10, 1.10 * shock.exp());
        let expect = a / s1 - a / s0;
        let mut shocks = HashMap::new();
        shocks.insert("FX:EUR".to_string(), shock);
        let got = cb.pnl_map(&shocks).unwrap();
        assert!((got - expect).abs() < 1e-9 * expect.abs());
    }

    #[test]
    fn empty_book_throws() {
        let m = test_market();
        assert!(CompiledBook::new(&Book::empty(), &m).is_err());
    }

    #[test]
    fn usd_shock_rejected_and_unknown_factors_ignored() {
        let m = test_market();
        let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 1e6, None))], "USD").unwrap();
        let cb = CompiledBook::new(&book, &m).unwrap();
        let mut shocks = HashMap::new();
        shocks.insert("FX:USD".to_string(), 0.01);
        assert!(cb.pnl_map(&shocks).is_err());
        let mut shocks2 = HashMap::new();
        shocks2.insert("FX:TRY".to_string(), -0.3);
        assert_eq!(cb.pnl_map(&shocks2).unwrap(), 0.0);
    }

    #[test]
    fn position_validation() {
        assert!(Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 1e6, Some(-1.0)))], "USD").is_err());
        assert!(Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 1e6, -0.5, None))], "USD").is_err());
        assert!(Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 1e6, 0.5, Some(0.0)))], "USD").is_err());
        assert!(Book::new(vec![Position::Spot(SpotPosition::new("E2RUSD", 1e6, None))], "USD").is_err());
    }

    #[test]
    fn single_currency_book_spot_exposure_is_notional() {
        let m = test_market();
        let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))], "USD").unwrap();
        let cb = CompiledBook::new(&book, &m).unwrap();
        let w = cb.linear_exposures(1e-6).unwrap();
        assert_eq!(w.len(), 1);
        assert!((w[0] - 10e6 * 1.10).abs() < 1e-3);
    }
}
