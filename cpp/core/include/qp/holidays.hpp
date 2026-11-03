// Rule-based settlement holiday calendars, FX spot dates, spot-date
// forwards and NDF values.
//
// C++ port of python/core/src/qp_core/holidays.py and of the date-based
// forward and NDF formulas in fx_rates (Python); the Rust port is
// rust/core/src/holidays.rs. Rules, sources and worked examples are in the
// Python module docstring. In short:
//
//   * USD (Federal Reserve), EUR (TARGET2), GBP (England and Wales bank
//     holidays with the 2002, 2011, 2012, 2020, 2022 and 2023
//     proclamations), JPY, CHF, CAD, AUD; years 2000-2099.
//   * FX spot: T+2, or T+1 for USDCAD, USDTRY, USDRUB, USDPHP. The lag is
//     counted on the shipped non-USD centres, then the date is rolled
//     forward until both centres and New York are open (Clark 2011, ch. 1;
//     Wystup 2017, ch. 1).
//   * Forward from the spot date: F = S [P_f(T)/P_f(t_s)] / [P_d(T)/P_d(t_s)].
//   * NDF: base-settled N P_base(T_s) (1 - K/F); quote-settled N P_quote(T_s) (F - K).
//
// Independent check: tests/test_holidays.cpp compares every weekday holiday
// 2000-2030 with a QuantLib 1.44 extract shared with the Rust tests.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace qp::holidays {

using Date = std::chrono::sys_days;

inline constexpr int kMinYear = 2000;
inline constexpr int kMaxYear = 2099;
inline constexpr std::array<std::string_view, 7> kCurrencies{"USD", "EUR", "GBP", "JPY",
                                                             "CHF", "CAD", "AUD"};

inline Date ymd(int y, unsigned m, unsigned d) {
    return Date{std::chrono::year{y} / std::chrono::month{m} / std::chrono::day{d}};
}

/// Monday = 0 ... Sunday = 6.
inline unsigned weekday(Date d) { return std::chrono::weekday{d}.iso_encoding() - 1; }
inline bool is_weekend(Date d) { return weekday(d) >= 5; }
inline int year_of(Date d) { return int(std::chrono::year_month_day{d}.year()); }
inline Date add_days(Date d, int n) { return d + std::chrono::days{n}; }

/// Gregorian Easter Sunday (Meeus 1991, ch. 8).
inline Date easter_sunday(int year) {
    const int a = year % 19, b = year / 100, c = year % 100;
    const int d = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3;
    const int h = (19 * a + b - d - g + 15) % 30;
    const int i = c / 4, k = c % 4;
    const int l = (32 + 2 * e + 2 * i - h - k) % 7;
    const int m = (a + 11 * h + 22 * l) / 451;
    const int n = h + l - 7 * m + 114;
    return ymd(year, unsigned(n / 31), unsigned(n % 31 + 1));
}

namespace detail {

inline Date nth_weekday(int y, unsigned m, unsigned wd, int n) {
    if (n > 0) {
        const Date first = ymd(y, m, 1);
        const unsigned off = (7 + wd - weekday(first)) % 7;
        return add_days(first, int(off) + 7 * (n - 1));
    }
    const Date last = Date{std::chrono::year{y} / std::chrono::month{m} / std::chrono::last};
    const unsigned off = (7 + weekday(last) - wd) % 7;
    return add_days(last, -int(off));
}

inline Date next_weekday(Date d) {
    while (is_weekend(d)) d = add_days(d, 1);
    return d;
}

inline void xmas_boxing(int y, std::set<Date>& out) {
    const Date c = ymd(y, 12, 25), b = ymd(y, 12, 26);
    if (weekday(c) == 5) {
        out.insert({ymd(y, 12, 27), ymd(y, 12, 28)});
    } else if (weekday(c) == 6) {
        out.insert({b, ymd(y, 12, 27)});
    } else if (weekday(b) == 5) {
        out.insert({c, ymd(y, 12, 28)});
    } else {
        out.insert({c, b});
    }
}

inline void usd(int y, std::set<Date>& out) {
    std::vector<Date> fixed{ymd(y, 1, 1), ymd(y, 7, 4), ymd(y, 11, 11), ymd(y, 12, 25)};
    if (y >= 2022) fixed.push_back(ymd(y, 6, 19));
    for (Date d : fixed) {
        if (weekday(d) == 5) continue;
        out.insert(weekday(d) == 6 ? add_days(d, 1) : d);
    }
    out.insert({nth_weekday(y, 1, 0, 3), nth_weekday(y, 2, 0, 3), nth_weekday(y, 5, 0, -1),
                nth_weekday(y, 9, 0, 1), nth_weekday(y, 10, 0, 2), nth_weekday(y, 11, 3, 4)});
}

inline void eur(int y, std::set<Date>& out) {
    const Date e = easter_sunday(y);
    out.insert({ymd(y, 1, 1), add_days(e, -2), add_days(e, 1), ymd(y, 5, 1), ymd(y, 12, 25),
                ymd(y, 12, 26)});
    if (y == 2000 || y == 2001) out.insert(ymd(y, 12, 31));
}

inline void gbp(int y, std::set<Date>& out) {
    const Date e = easter_sunday(y);
    out.insert(
        {next_weekday(ymd(y, 1, 1)), add_days(e, -2), add_days(e, 1), nth_weekday(y, 8, 0, -1)});
    xmas_boxing(y, out);
    out.insert(y == 2020 ? ymd(2020, 5, 8) : nth_weekday(y, 5, 0, 1));
    if (y == 2002) {
        out.insert({ymd(2002, 6, 3), ymd(2002, 6, 4)});
    } else if (y == 2012) {
        out.insert({ymd(2012, 6, 4), ymd(2012, 6, 5)});
    } else if (y == 2022) {
        out.insert({ymd(2022, 6, 2), ymd(2022, 6, 3), ymd(2022, 9, 19)});
    } else {
        out.insert(nth_weekday(y, 5, 0, -1));
    }
    if (y == 2011) out.insert(ymd(2011, 4, 29));
    if (y == 2023) out.insert(ymd(2023, 5, 8));
}

inline unsigned equinox(int y, double base) {
    return unsigned(int(std::floor(base + 0.242194 * (y - 1980))) - (y - 1980) / 4);
}

inline void jpy(int y, std::set<Date>& out) {
    std::set<Date> nat{ymd(y, 1, 1),
                       ymd(y, 2, 11),
                       ymd(y, 4, 29),
                       ymd(y, 5, 3),
                       ymd(y, 5, 4),
                       ymd(y, 5, 5),
                       ymd(y, 11, 3),
                       ymd(y, 11, 23),
                       nth_weekday(y, 1, 0, 2),
                       y >= 2003 ? nth_weekday(y, 9, 0, 3) : ymd(y, 9, 15),
                       ymd(y, 3, equinox(y, 20.8431)),
                       ymd(y, 9, equinox(y, 23.2488))};
    if (y >= 2020) {
        nat.insert(ymd(y, 2, 23));
    } else if (y >= 1989 && y <= 2018) {
        nat.insert(ymd(y, 12, 23));
    }
    if (y == 2019)
        nat.insert({ymd(2019, 4, 30), ymd(2019, 5, 1), ymd(2019, 5, 2), ymd(2019, 10, 22)});
    if (y == 2020) {
        nat.insert({ymd(2020, 7, 23), ymd(2020, 7, 24), ymd(2020, 8, 10)});
    } else if (y == 2021) {
        nat.insert({ymd(2021, 7, 22), ymd(2021, 7, 23), ymd(2021, 8, 8)});
    } else {
        nat.insert(y >= 2003 ? nth_weekday(y, 7, 0, 3) : ymd(y, 7, 20));
        nat.insert(nth_weekday(y, 10, 0, 2));
        if (y >= 2016) nat.insert(ymd(y, 8, 11));
    }
    // Citizens' holiday: a non-Sunday day between two national holidays.
    const std::vector<Date> sorted(nat.begin(), nat.end());
    for (Date d : sorted) {
        const Date mid = add_days(d, 1);
        if (!nat.contains(mid) && nat.contains(add_days(mid, 1)) && weekday(mid) != 6)
            nat.insert(mid);
    }
    // Substitute holiday: a Sunday holiday moves to the next non-holiday day.
    std::set<Date> all = nat;
    for (Date d : nat) {
        if (weekday(d) == 6) {
            Date s = add_days(d, 1);
            while (all.contains(s)) s = add_days(s, 1);
            all.insert(s);
        }
    }
    out.insert(all.begin(), all.end());
    out.insert({ymd(y, 1, 2), ymd(y, 1, 3), ymd(y, 12, 31)});
}

inline void chf(int y, std::set<Date>& out) {
    const Date e = easter_sunday(y);
    out.insert({ymd(y, 1, 1), ymd(y, 1, 2), add_days(e, -2), add_days(e, 1), ymd(y, 5, 1),
                add_days(e, 39), add_days(e, 50), ymd(y, 8, 1), ymd(y, 12, 25), ymd(y, 12, 26)});
}

inline void cad(int y, std::set<Date>& out) {
    const Date e = easter_sunday(y);
    const Date may24 = ymd(y, 5, 24);
    out.insert({next_weekday(ymd(y, 1, 1)), add_days(e, -2), add_days(may24, -int(weekday(may24))),
                next_weekday(ymd(y, 7, 1)), nth_weekday(y, 8, 0, 1), nth_weekday(y, 9, 0, 1),
                nth_weekday(y, 10, 0, 2), next_weekday(ymd(y, 11, 11))});
    if (y >= 2008) out.insert(nth_weekday(y, 2, 0, 3));
    if (y >= 2021) out.insert(next_weekday(ymd(y, 9, 30)));
    xmas_boxing(y, out);
}

inline void aud(int y, std::set<Date>& out) {
    const Date e = easter_sunday(y);
    out.insert({next_weekday(ymd(y, 1, 1)), next_weekday(ymd(y, 1, 26)), add_days(e, -2),
                add_days(e, 1), ymd(y, 4, 25), nth_weekday(y, 6, 0, 2), nth_weekday(y, 8, 0, 1),
                nth_weekday(y, 10, 0, 1)});
    if (y == 2022) out.insert(ymd(2022, 9, 22));
    xmas_boxing(y, out);
}

inline bool shipped(std::string_view c) {
    return std::find(kCurrencies.begin(), kCurrencies.end(), c) != kCurrencies.end();
}

}  // namespace detail

/// Weekday settlement holidays of `ccy` in `year`. Throws std::invalid_argument
/// on an unknown currency or a year outside [kMinYear, kMaxYear].
inline std::set<Date> holidays(std::string_view ccy, int year) {
    if (year < kMinYear || year > kMaxYear) throw std::invalid_argument("year out of range");
    std::set<Date> out;
    if (ccy == "USD") {
        detail::usd(year, out);
    } else if (ccy == "EUR" || ccy == "TARGET") {
        detail::eur(year, out);
    } else if (ccy == "GBP") {
        detail::gbp(year, out);
    } else if (ccy == "JPY") {
        detail::jpy(year, out);
    } else if (ccy == "CHF") {
        detail::chf(year, out);
    } else if (ccy == "CAD") {
        detail::cad(year, out);
    } else if (ccy == "AUD") {
        detail::aud(year, out);
    } else {
        throw std::invalid_argument("no calendar for this currency");
    }
    std::erase_if(out, [year](Date d) { return year_of(d) != year || is_weekend(d); });
    return out;
}

/// True when `d` is a weekday open in every centre (empty = weekends only).
inline bool is_business_day(Date d, const std::vector<std::string_view>& centres) {
    if (is_weekend(d)) return false;
    for (auto c : centres)
        if (holidays(c, year_of(d)).contains(d)) return false;
    return true;
}

inline Date roll_forward(Date d, const std::vector<std::string_view>& centres) {
    while (!is_business_day(d, centres)) d = add_days(d, 1);
    return d;
}

/// 1 for USDCAD, USDTRY, USDRUB, USDPHP (either order), else 2.
inline int spot_lag(std::string_view base, std::string_view quote) {
    constexpr std::array<std::string_view, 4> t1{"CAD", "TRY", "RUB", "PHP"};
    auto in = [&](std::string_view c) { return std::find(t1.begin(), t1.end(), c) != t1.end(); };
    return ((base == "USD" && in(quote)) || (quote == "USD" && in(base))) ? 1 : 2;
}

/// FX spot date (see the header comment for the rule).
inline Date fx_spot_date(Date trade, std::string_view base, std::string_view quote) {
    std::vector<std::string_view> count;
    for (auto c : {base, quote})
        if (c != "USD" && detail::shipped(c)) count.push_back(c);
    Date d = trade;
    for (int i = 0; i < spot_lag(base, quote); ++i) d = roll_forward(add_days(d, 1), count);
    std::vector<std::string_view> all;
    for (auto c : {base, quote, std::string_view{"USD"}})
        if (detail::shipped(c) && std::find(all.begin(), all.end(), c) == all.end())
            all.push_back(c);
    return roll_forward(d, all);
}

/// ACT/365F year fraction.
inline double yf365(Date a, Date b) { return double((b - a).count()) / 365.0; }

/// S exp((r_d - r_f)(T - t_s)).
inline double forward_flat(double spot, double t, double t_spot, double r_d, double r_f) {
    return spot * std::exp((r_d - r_f) * (t - t_spot));
}

/// S [P_f(T)/P_f(t_s)] / [P_d(T)/P_d(t_s)].
inline double forward_from_dfs(double spot, double pd_t, double pf_t, double pd_ts, double pf_ts) {
    return spot * (pf_t / pf_ts) / (pd_t / pd_ts);
}

/// NDF value in the settlement currency.
inline double ndf_value(double notional_base, double strike, double forward, double df_settle,
                        bool base_settled) {
    return base_settled ? notional_base * df_settle * (1.0 - strike / forward)
                        : notional_base * df_settle * (forward - strike);
}

}  // namespace qp::holidays
