//! Rule-based settlement holiday calendars, FX spot dates, spot-date
//! forwards and NDF values.
//!
//! This is the Rust port of `qp_core.holidays` and of the date-based forward
//! and NDF formulas in `fx_rates` (Python). The rules, their sources and the
//! worked examples are documented in `python/core/src/qp_core/holidays.py`;
//! the short version:
//!
//! * USD (Federal Reserve), EUR (TARGET2), GBP (England and Wales bank
//!   holidays with the 2002, 2011, 2012, 2020, 2022 and 2023
//!   proclamations), JPY (national holidays plus 2-3 Jan and 31 Dec), CHF
//!   (Zurich), CAD (Toronto), AUD (Sydney), years 2000-2099.
//! * USD-GOVT (alias SIFMA): the SIFMA US bond-market calendar, for
//!   Treasury schedules. Fed list plus Good Friday (except a first-Friday
//!   Good Friday, an early close), Saturday Jul 4, Juneteenth and Christmas
//!   observed on Friday, one-off closures 2001-09-11/12, 2004-06-11,
//!   2012-10-30, 2018-12-05, 2025-01-09. Early closes are not modelled.
//! * FX spot: T+2, or T+1 for USDCAD, USDTRY, USDRUB, USDPHP; the lag is
//!   counted on the non-USD centres and the result is rolled forward until
//!   both centres and New York are open (Clark 2011, ch. 1; Wystup 2017,
//!   ch. 1).
//! * Forward from the spot date: `F(T) = S * [P_f(T)/P_f(t_s)] / [P_d(T)/P_d(t_s)]`.
//! * NDF settled in the base currency: `N * P_base(T_s) * (1 - K / F)`;
//!   settled in the quote currency: `N * P_quote(T_s) * (F - K)`.
//!
//! Independent check: `tests/holidays_parity.rs` compares every weekday
//! holiday for 2000-2030 with a QuantLib 1.44 extract.

use std::collections::BTreeSet;

/// Earliest supported year.
pub const MIN_YEAR: i32 = 2000;
/// Latest supported year.
pub const MAX_YEAR: i32 = 2099;
/// Currencies with a shipped calendar.
pub const CURRENCIES: [&str; 7] = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"];
/// Every calendar code: the currencies plus `USD-GOVT` (SIFMA bond market).
pub const CALENDAR_CODES: [&str; 8] = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "USD-GOVT"];

/// A proleptic Gregorian calendar date.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    /// Year.
    pub y: i32,
    /// Month, 1-12.
    pub m: u32,
    /// Day of month, 1-31.
    pub d: u32,
}

impl Date {
    /// Build a date; panics on an impossible month or day.
    pub fn new(y: i32, m: u32, d: u32) -> Self {
        assert!(
            (1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m),
            "bad date"
        );
        Date { y, m, d }
    }

    /// Days since 1970-01-01 (Hinnant's `days_from_civil`).
    pub fn serial(self) -> i64 {
        let y = i64::from(self.y) - i64::from(self.m <= 2);
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let m = i64::from(self.m);
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(self.d) - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// Inverse of [`Date::serial`].
    pub fn from_serial(z: i64) -> Self {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        let y = (yoe + era * 400 + i64::from(m <= 2)) as i32;
        Date { y, m, d }
    }

    /// Add calendar days.
    pub fn add_days(self, n: i64) -> Self {
        Date::from_serial(self.serial() + n)
    }

    /// Weekday, Monday = 0 ... Sunday = 6.
    pub fn weekday(self) -> u32 {
        // 1970-01-01 was a Thursday (3).
        ((self.serial() + 3).rem_euclid(7)) as u32
    }

    /// True on Saturday or Sunday.
    pub fn is_weekend(self) -> bool {
        self.weekday() >= 5
    }
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        2 if is_leap(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Gregorian Easter Sunday (anonymous Gregorian algorithm, Meeus 1991, ch. 8).
pub fn easter_sunday(year: i32) -> Date {
    let (a, b, c) = (year % 19, year / 100, year % 100);
    let (d, e) = (b / 4, b % 4);
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let (i, k) = (c / 4, c % 4);
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let n = h + l - 7 * m + 114;
    Date::new(year, (n / 31) as u32, (n % 31 + 1) as u32)
}

fn nth_weekday(y: i32, m: u32, wd: u32, n: i32) -> Date {
    if n > 0 {
        let first = Date::new(y, m, 1);
        let off = (7 + wd - first.weekday()) % 7;
        first.add_days(i64::from(off) + 7 * i64::from(n - 1))
    } else {
        let last = Date::new(y, m, days_in_month(y, m));
        let off = (7 + last.weekday() - wd) % 7;
        last.add_days(-i64::from(off))
    }
}

fn next_weekday(mut d: Date) -> Date {
    while d.is_weekend() {
        d = d.add_days(1);
    }
    d
}

fn xmas_boxing(y: i32, out: &mut BTreeSet<Date>) {
    let c = Date::new(y, 12, 25);
    let b = Date::new(y, 12, 26);
    match (c.weekday(), b.weekday()) {
        (5, _) => {
            out.insert(Date::new(y, 12, 27));
            out.insert(Date::new(y, 12, 28));
        }
        (6, _) => {
            out.insert(b);
            out.insert(Date::new(y, 12, 27));
        }
        (_, 5) => {
            out.insert(c);
            out.insert(Date::new(y, 12, 28));
        }
        _ => {
            out.insert(c);
            out.insert(b);
        }
    }
}

fn usd(y: i32, out: &mut BTreeSet<Date>) {
    let mut fixed = vec![
        Date::new(y, 1, 1),
        Date::new(y, 7, 4),
        Date::new(y, 11, 11),
        Date::new(y, 12, 25),
    ];
    if y >= 2022 {
        fixed.push(Date::new(y, 6, 19));
    }
    for d in fixed {
        match d.weekday() {
            5 => {}
            6 => {
                out.insert(d.add_days(1));
            }
            _ => {
                out.insert(d);
            }
        }
    }
    for d in [
        nth_weekday(y, 1, 0, 3),
        nth_weekday(y, 2, 0, 3),
        nth_weekday(y, 5, 0, -1),
        nth_weekday(y, 9, 0, 1),
        nth_weekday(y, 10, 0, 2),
        nth_weekday(y, 11, 3, 4),
    ] {
        out.insert(d);
    }
}

const GOVT_SPECIAL: [(i32, u32, u32); 6] = [
    (2001, 9, 11),
    (2001, 9, 12),
    (2004, 6, 11),
    (2012, 10, 30),
    (2018, 12, 5),
    (2025, 1, 9),
];

fn usd_govt(y: i32, out: &mut BTreeSet<Date>) {
    for d in [
        nth_weekday(y, 1, 0, 3),
        nth_weekday(y, 2, 0, 3),
        nth_weekday(y, 5, 0, -1),
        nth_weekday(y, 9, 0, 1),
        nth_weekday(y, 10, 0, 2),
        nth_weekday(y, 11, 3, 4),
    ] {
        out.insert(d);
    }
    // New Year and Veterans Day: Sunday -> Monday, Saturday not moved.
    for d in [Date::new(y, 1, 1), Date::new(y, 11, 11)] {
        match d.weekday() {
            5 => {}
            6 => {
                out.insert(d.add_days(1));
            }
            _ => {
                out.insert(d);
            }
        }
    }
    let mut fixed = vec![Date::new(y, 7, 4), Date::new(y, 12, 25)];
    if y >= 2022 {
        fixed.push(Date::new(y, 6, 19));
    }
    for d in fixed {
        out.insert(match d.weekday() {
            5 => d.add_days(-1),
            6 => d.add_days(1),
            _ => d,
        });
    }
    let good_friday = easter_sunday(y).add_days(-2);
    if good_friday.d > 7 {
        out.insert(good_friday);
    }
    for (yy, m, d) in GOVT_SPECIAL {
        if yy == y {
            out.insert(Date::new(yy, m, d));
        }
    }
}

fn eur(y: i32, out: &mut BTreeSet<Date>) {
    let e = easter_sunday(y);
    for d in [
        Date::new(y, 1, 1),
        e.add_days(-2),
        e.add_days(1),
        Date::new(y, 5, 1),
        Date::new(y, 12, 25),
        Date::new(y, 12, 26),
    ] {
        out.insert(d);
    }
    if y == 2000 || y == 2001 {
        out.insert(Date::new(y, 12, 31));
    }
}

fn gbp(y: i32, out: &mut BTreeSet<Date>) {
    let e = easter_sunday(y);
    out.insert(next_weekday(Date::new(y, 1, 1)));
    out.insert(e.add_days(-2));
    out.insert(e.add_days(1));
    out.insert(nth_weekday(y, 8, 0, -1));
    xmas_boxing(y, out);
    out.insert(if y == 2020 {
        Date::new(2020, 5, 8)
    } else {
        nth_weekday(y, 5, 0, 1)
    });
    match y {
        2002 => {
            out.insert(Date::new(2002, 6, 3));
            out.insert(Date::new(2002, 6, 4));
        }
        2012 => {
            out.insert(Date::new(2012, 6, 4));
            out.insert(Date::new(2012, 6, 5));
        }
        2022 => {
            out.insert(Date::new(2022, 6, 2));
            out.insert(Date::new(2022, 6, 3));
            out.insert(Date::new(2022, 9, 19));
        }
        _ => {
            out.insert(nth_weekday(y, 5, 0, -1));
        }
    }
    if y == 2011 {
        out.insert(Date::new(2011, 4, 29));
    }
    if y == 2023 {
        out.insert(Date::new(2023, 5, 8));
    }
}

fn equinox(y: i32, base: f64) -> u32 {
    ((base + 0.242_194 * f64::from(y - 1980)).floor() as i32 - (y - 1980) / 4) as u32
}

fn jpy(y: i32, out: &mut BTreeSet<Date>) {
    let mut nat: BTreeSet<Date> = [
        Date::new(y, 1, 1),
        Date::new(y, 2, 11),
        Date::new(y, 4, 29),
        Date::new(y, 5, 3),
        Date::new(y, 5, 4),
        Date::new(y, 5, 5),
        Date::new(y, 11, 3),
        Date::new(y, 11, 23),
        nth_weekday(y, 1, 0, 2),
        if y >= 2003 {
            nth_weekday(y, 9, 0, 3)
        } else {
            Date::new(y, 9, 15)
        },
        Date::new(y, 3, equinox(y, 20.8431)),
        Date::new(y, 9, equinox(y, 23.2488)),
    ]
    .into_iter()
    .collect();
    if y >= 2020 {
        nat.insert(Date::new(y, 2, 23));
    } else if (1989..=2018).contains(&y) {
        nat.insert(Date::new(y, 12, 23));
    }
    if y == 2019 {
        for (m, d) in [(4, 30), (5, 1), (5, 2), (10, 22)] {
            nat.insert(Date::new(2019, m, d));
        }
    }
    match y {
        2020 => {
            for (m, d) in [(7, 23), (7, 24), (8, 10)] {
                nat.insert(Date::new(2020, m, d));
            }
        }
        2021 => {
            for (m, d) in [(7, 22), (7, 23), (8, 8)] {
                nat.insert(Date::new(2021, m, d));
            }
        }
        _ => {
            nat.insert(if y >= 2003 {
                nth_weekday(y, 7, 0, 3)
            } else {
                Date::new(y, 7, 20)
            });
            nat.insert(nth_weekday(y, 10, 0, 2));
            if y >= 2016 {
                nat.insert(Date::new(y, 8, 11));
            }
        }
    }
    // Citizens' holiday: a non-Sunday day between two national holidays.
    // Same single pass, in date order, as the Python generator.
    let sorted: Vec<Date> = nat.iter().copied().collect();
    for d in sorted {
        let mid = d.add_days(1);
        if !nat.contains(&mid) && nat.contains(&mid.add_days(1)) && mid.weekday() != 6 {
            nat.insert(mid);
        }
    }
    // Substitute holiday: a Sunday holiday moves to the next non-holiday day.
    let mut all = nat.clone();
    for d in nat.iter().copied() {
        if d.weekday() == 6 {
            let mut s = d.add_days(1);
            while all.contains(&s) {
                s = s.add_days(1);
            }
            all.insert(s);
        }
    }
    out.extend(all);
    for (m, d) in [(1, 2), (1, 3), (12, 31)] {
        out.insert(Date::new(y, m, d));
    }
}

fn chf(y: i32, out: &mut BTreeSet<Date>) {
    let e = easter_sunday(y);
    for d in [
        Date::new(y, 1, 1),
        Date::new(y, 1, 2),
        e.add_days(-2),
        e.add_days(1),
        Date::new(y, 5, 1),
        e.add_days(39),
        e.add_days(50),
        Date::new(y, 8, 1),
        Date::new(y, 12, 25),
        Date::new(y, 12, 26),
    ] {
        out.insert(d);
    }
}

fn cad(y: i32, out: &mut BTreeSet<Date>) {
    let e = easter_sunday(y);
    let may24 = Date::new(y, 5, 24);
    for d in [
        next_weekday(Date::new(y, 1, 1)),
        e.add_days(-2),
        may24.add_days(-i64::from(may24.weekday())),
        next_weekday(Date::new(y, 7, 1)),
        nth_weekday(y, 8, 0, 1),
        nth_weekday(y, 9, 0, 1),
        nth_weekday(y, 10, 0, 2),
        next_weekday(Date::new(y, 11, 11)),
    ] {
        out.insert(d);
    }
    if y >= 2008 {
        out.insert(nth_weekday(y, 2, 0, 3));
    }
    if y >= 2021 {
        out.insert(next_weekday(Date::new(y, 9, 30)));
    }
    xmas_boxing(y, out);
}

fn aud(y: i32, out: &mut BTreeSet<Date>) {
    let e = easter_sunday(y);
    for d in [
        next_weekday(Date::new(y, 1, 1)),
        next_weekday(Date::new(y, 1, 26)),
        e.add_days(-2),
        e.add_days(1),
        Date::new(y, 4, 25),
        nth_weekday(y, 6, 0, 2),
        nth_weekday(y, 8, 0, 1),
        nth_weekday(y, 10, 0, 1),
    ] {
        out.insert(d);
    }
    if y == 2022 {
        out.insert(Date::new(2022, 9, 22));
    }
    xmas_boxing(y, out);
}

/// Error for an unknown currency or a year outside `[MIN_YEAR, MAX_YEAR]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalendarError(pub String);

impl std::fmt::Display for CalendarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CalendarError {}

/// Weekday settlement holidays of `ccy` in `year`.
pub fn holidays(ccy: &str, year: i32) -> Result<BTreeSet<Date>, CalendarError> {
    if !(MIN_YEAR..=MAX_YEAR).contains(&year) {
        return Err(CalendarError(format!(
            "year must be in [{MIN_YEAR}, {MAX_YEAR}], got {year}"
        )));
    }
    let mut out = BTreeSet::new();
    match ccy {
        "USD" => usd(year, &mut out),
        "EUR" | "TARGET" => eur(year, &mut out),
        "GBP" => gbp(year, &mut out),
        "JPY" => jpy(year, &mut out),
        "CHF" => chf(year, &mut out),
        "CAD" => cad(year, &mut out),
        "AUD" => aud(year, &mut out),
        "USD-GOVT" | "SIFMA" => usd_govt(year, &mut out),
        _ => return Err(CalendarError(format!("no calendar for {ccy:?}"))),
    }
    out.retain(|d| d.y == year && !d.is_weekend());
    Ok(out)
}

/// True when `d` is a weekday open in every centre of `centres`
/// (an empty slice means weekends only).
pub fn is_business_day(d: Date, centres: &[&str]) -> Result<bool, CalendarError> {
    if d.is_weekend() {
        return Ok(false);
    }
    for c in centres {
        if holidays(c, d.y)?.contains(&d) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn roll_forward(mut d: Date, centres: &[&str]) -> Result<Date, CalendarError> {
    while !is_business_day(d, centres)? {
        d = d.add_days(1);
    }
    Ok(d)
}

/// Spot lag in business days: 1 for USDCAD, USDTRY, USDRUB, USDPHP (either
/// order), else 2.
pub fn spot_lag(base: &str, quote: &str) -> u32 {
    let t1 = ["CAD", "TRY", "RUB", "PHP"];
    if (base == "USD" && t1.contains(&quote)) || (quote == "USD" && t1.contains(&base)) {
        1
    } else {
        2
    }
}

/// FX spot date: lag counted on the non-USD centres, then rolled forward
/// until both centres and USD are open. Currencies with no shipped
/// calendar count weekends only.
pub fn fx_spot_date(trade: Date, base: &str, quote: &str) -> Result<Date, CalendarError> {
    let shipped = |c: &&str| CURRENCIES.contains(c);
    let non_usd: Vec<&str> = [base, quote]
        .into_iter()
        .filter(|c| *c != "USD")
        .filter(shipped)
        .collect();
    // Count on the shipped non-USD centres; none shipped means weekends only.
    let count = non_usd;
    let mut d = trade;
    for _ in 0..spot_lag(base, quote) {
        d = roll_forward(d.add_days(1), &count)?;
    }
    let mut all: Vec<&str> = [base, quote, "USD"].into_iter().filter(shipped).collect();
    all.dedup();
    roll_forward(d, &all)
}

/// ACT/365F year fraction.
pub fn yf365(a: Date, b: Date) -> f64 {
    (b.serial() - a.serial()) as f64 / 365.0
}

/// Forward from the spot date with flat continuously compounded rates:
/// `S * exp((r_d - r_f) * (T - t_s))`.
pub fn forward_flat(spot: f64, t: f64, t_spot: f64, r_d: f64, r_f: f64) -> f64 {
    spot * ((r_d - r_f) * (t - t_spot)).exp()
}

/// Forward from the spot date with discount factors:
/// `S * [P_f(T)/P_f(t_s)] / [P_d(T)/P_d(t_s)]`.
pub fn forward_from_dfs(spot: f64, pd_t: f64, pf_t: f64, pd_ts: f64, pf_ts: f64) -> f64 {
    spot * (pf_t / pf_ts) / (pd_t / pd_ts)
}

/// NDF present value in the settlement currency. `forward` is the forward
/// to the settlement date, `df_settle` the settlement currency's discount
/// factor to it; `base_settled` selects `N (1 - K/F)` over `N (F - K)`.
pub fn ndf_value(
    notional_base: f64,
    strike: f64,
    forward: f64,
    df_settle: f64,
    base_settled: bool,
) -> f64 {
    if base_settled {
        notional_base * df_settle * (1.0 - strike / forward)
    } else {
        notional_base * df_settle * (forward - strike)
    }
}

/// Value today, in quote currency, of buying `n` base units at `k` for
/// value at time `t` (one FX swap leg or an outright):
/// `n * (s_today * P_f(t) - k * P_d(t))`, with `s_today = S P_d(t_s)/P_f(t_s)`.
pub fn fx_leg_value(n: f64, k: f64, s_today: f64, pf_t: f64, pd_t: f64) -> f64 {
    n * (s_today * pf_t - k * pd_t)
}

/// Far rate that makes an FX swap worth zero: `F(T_f) - V_near / (n_f P_d(T_f))`.
pub fn fx_swap_far_rate(forward_far: f64, v_near: f64, n_far: f64, pd_far: f64) -> f64 {
    forward_far - v_near / (n_far * pd_far)
}

/// Value of a cross-currency swap's initial exchange on the spot date:
/// `sign * P_d(t_s) * (N_quote - S * N_base)` (`sign` +1 when receiving the
/// base leg, so paying base and receiving quote at the start).
pub fn xccy_initial_exchange(sign: f64, n_quote: f64, n_base: f64, spot: f64, pd_ts: f64) -> f64 {
    sign * pd_ts * (n_quote - spot * n_base)
}
