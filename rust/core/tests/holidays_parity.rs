//! Parity of the Rust calendar port with an independent oracle (QuantLib
//! 1.44 holiday lists, 2000-2030) and with the Python spot-date table.

use qp_core::holidays::{
    CALENDAR_CODES, CURRENCIES, Date, easter_sunday, forward_flat, forward_from_dfs, fx_leg_value,
    fx_spot_date, fx_swap_far_rate, holidays, is_business_day, ndf_value, spot_lag,
    xccy_initial_exchange, yf365,
};
use std::collections::{BTreeMap, BTreeSet};

fn parse(s: &str) -> Date {
    let p: Vec<&str> = s.split('-').collect();
    Date::new(
        p[0].parse().unwrap(),
        p[1].parse().unwrap(),
        p[2].parse().unwrap(),
    )
}

#[test]
fn matches_quantlib_2000_2030() {
    let text = include_str!("data/ql_holidays_2000_2030.txt");
    let mut want: BTreeMap<(String, i32), BTreeSet<Date>> = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let (ccy, d) = line.split_once(' ').unwrap();
        let d = parse(d);
        want.entry((ccy.to_string(), d.y)).or_default().insert(d);
    }
    // SIFMA closures that QuantLib 1.44 does not list (see holidays.py).
    let not_in_ql = [
        Date::new(2001, 9, 11),
        Date::new(2001, 9, 12),
        Date::new(2025, 1, 9),
    ];
    assert_eq!(CALENDAR_CODES.len(), CURRENCIES.len() + 1);
    for ccy in CALENDAR_CODES {
        for y in 2000..=2030 {
            let got = holidays(ccy, y).unwrap();
            let mut exp = want.get(&(ccy.to_string(), y)).cloned().unwrap_or_default();
            if ccy == "USD-GOVT" {
                exp.extend(not_in_ql.iter().filter(|d| d.y == y));
            }
            assert_eq!(got, exp, "{ccy} {y}");
        }
    }
}

#[test]
fn usd_govt_named_days() {
    let govt = ["USD-GOVT"];
    // Good Friday 2025 closed for bonds, open at the Fed.
    assert!(!is_business_day(Date::new(2025, 4, 18), &govt).unwrap());
    assert!(is_business_day(Date::new(2025, 4, 18), &["USD"]).unwrap());
    // First-Friday Good Friday 2026: early close, business day.
    assert!(is_business_day(Date::new(2026, 4, 3), &govt).unwrap());
    // Saturday Christmas 2027 observed Friday.
    assert!(!is_business_day(Date::new(2027, 12, 24), &govt).unwrap());
    assert_eq!(holidays("SIFMA", 2025), holidays("USD-GOVT", 2025));
}

#[test]
fn dates_and_easter() {
    let d = Date::new(2024, 2, 29);
    assert_eq!(Date::from_serial(d.serial()), d);
    assert_eq!(Date::new(1970, 1, 1).serial(), 0);
    assert_eq!(Date::new(2025, 6, 5).weekday(), 3);
    assert_eq!(easter_sunday(2025), Date::new(2025, 4, 20));
    assert_eq!(easter_sunday(2038), Date::new(2038, 4, 25));
    assert!(holidays("USD", 1999).is_err());
    assert!(holidays("NZD", 2025).is_err());
}

#[test]
fn spot_table() {
    // Same table as python/fx/04-fixed-income/tests/test_fx_spot_forwards.py
    let table = [
        ("2025-06-05", "EURUSD", "2025-06-09"),
        ("2025-04-30", "EURUSD", "2025-05-05"),
        ("2025-07-02", "EURUSD", "2025-07-07"),
        ("2025-07-03", "EURUSD", "2025-07-07"),
        ("2025-07-02", "EURJPY", "2025-07-07"),
        ("2025-07-02", "GBPJPY", "2025-07-07"),
        ("2025-06-06", "USDCAD", "2025-06-09"),
        ("2025-06-30", "USDCAD", "2025-07-02"),
        ("2025-04-16", "GBPUSD", "2025-04-22"),
        ("2025-04-17", "GBPUSD", "2025-04-23"),
        ("2025-04-17", "EURUSD", "2025-04-23"),
        ("2025-12-23", "EURUSD", "2025-12-29"),
        ("2025-12-24", "GBPUSD", "2025-12-30"),
        ("2025-12-30", "USDJPY", "2026-01-06"),
        ("2025-12-31", "USDJPY", "2026-01-06"),
        ("2025-11-26", "EURUSD", "2025-11-28"),
        ("2025-11-25", "EURUSD", "2025-11-28"),
        ("2025-05-01", "USDJPY", "2025-05-07"),
        ("2025-05-02", "USDJPY", "2025-05-08"),
        ("2026-01-29", "EURUSD", "2026-02-02"),
        ("2026-02-26", "USDCHF", "2026-03-02"),
        ("2026-03-27", "AUDUSD", "2026-03-31"),
        ("2025-04-24", "AUDUSD", "2025-04-29"),
        ("2025-08-29", "GBPUSD", "2025-09-02"),
        ("2025-05-22", "EURCHF", "2025-05-27"),
        ("2025-05-28", "EURCHF", "2025-06-02"),
        ("2025-09-26", "USDCAD", "2025-09-29"),
        ("2025-10-10", "USDCAD", "2025-10-14"),
        ("2026-06-18", "EURUSD", "2026-06-22"),
        ("2025-01-17", "USDJPY", "2025-01-21"),
        ("2025-10-31", "EURUSD", "2025-11-04"),
        ("2025-06-17", "USDTRY", "2025-06-18"),
    ];
    for (t, pair, want) in table {
        let got = fx_spot_date(parse(t), &pair[..3], &pair[3..]).unwrap();
        assert_eq!(got, parse(want), "{t} {pair}");
    }
    assert_eq!(spot_lag("USD", "CAD"), 1);
    assert_eq!(spot_lag("CAD", "USD"), 1);
    assert_eq!(spot_lag("EUR", "CAD"), 2);
}

#[test]
fn forward_and_ndf_closed_forms() {
    let (s, rd, rf) = (1.0850_f64, 0.045, 0.025);
    let val = Date::new(2025, 6, 5);
    let ts = yf365(val, fx_spot_date(val, "EUR", "USD").unwrap());
    assert!((ts - 4.0 / 365.0).abs() < 1e-15);
    let f = forward_flat(s, 1.0, ts, rd, rf);
    let g = forward_from_dfs(
        s,
        (-rd * 1.0_f64).exp(),
        (-rf * 1.0_f64).exp(),
        (-rd * ts).exp(),
        (-rf * ts).exp(),
    );
    assert!((f - g).abs() < 1e-14);
    assert!((f - 1.106_675_868_250_858).abs() < 1e-12);
    // NDF from the Python worked example: USDBRL 10m at 5.55, settle 8 Dec 2025.
    let t = yf365(val, Date::new(2025, 12, 8));
    let fwd = forward_flat(5.40, t, ts, 0.11, 0.045);
    let v = ndf_value(10e6, 5.55, fwd, (-0.045 * t).exp(), true);
    assert!((v - 48_861.053_538_993_1).abs() < 1e-6, "{v}");
    let q = ndf_value(10e6, 5.6, fwd, (-0.11 * t).exp(), false);
    assert!((q - 10e6 * (-0.11 * t).exp() * (fwd - 5.6)).abs() < 1e-9);
}

#[test]
fn fx_swap_and_xccy_closed_forms() {
    // Same numbers as python/fx/04-fixed-income/tests/test_fx_swaps_dated.py
    // and test_xccy_initial_exchange.py (EURUSD 1.0850, flat 4.5% / 2.5%,
    // valued Thu 5-Jun-2025, spot Mon 9-Jun).
    let (s, rd, rf) = (1.0850_f64, 0.045_f64, 0.025_f64);
    let val = Date::new(2025, 6, 5);
    let ts = yf365(val, fx_spot_date(val, "EUR", "USD").unwrap());
    let s_today = s * (-rd * ts).exp() / (-rf * ts).exp();
    // tom-next points: (S - F(tom)) * 1e4
    let f_tom = forward_flat(s, 1.0 / 365.0, ts, rd, rf);
    assert!(((s - f_tom) * 1e4 - 1.7834150577589192).abs() < 1e-10);
    // off-market swap legs, near 9-Jun-2025 at 1.08, far 9-Jun-2026 at 1.10
    let tn = yf365(val, Date::new(2025, 6, 9));
    let tf = yf365(val, Date::new(2026, 6, 9));
    let near = fx_leg_value(10e6, 1.08, s_today, (-rf * tn).exp(), (-rd * tn).exp());
    let far = fx_leg_value(-10e6, 1.10, s_today, (-rf * tf).exp(), (-rd * tf).exp());
    assert!((near - 49975.348544692984).abs() < 1e-6, "{near}");
    assert!((far + 66107.63627727056).abs() < 1e-6, "{far}");
    // far rate for an off-market near leg at 1.07, far 9-Dec-2025
    let t9 = yf365(val, Date::new(2025, 12, 9));
    let v_near = fx_leg_value(10e6, 1.07, s_today, (-rf * tn).exp(), (-rd * tn).exp());
    let kf = fx_swap_far_rate(
        forward_flat(s, t9, ts, rd, rf),
        v_near,
        10e6,
        (-rd * t9).exp(),
    );
    assert!((kf - 1.0805921852368006).abs() < 1e-13, "{kf}");
    // xccy initial exchange, 150k off market
    let x = xccy_initial_exchange(1.0, 11.0e6, 10e6, s, (-rd * ts).exp());
    assert!((x - 149926.0456340816).abs() < 1e-6, "{x}");
}
