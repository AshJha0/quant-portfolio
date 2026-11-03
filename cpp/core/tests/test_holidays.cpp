// Parity of the C++ calendar port with an independent oracle (QuantLib
// 1.44 holiday lists 2000-2030, the same extract the Rust tests use) and
// with the Python spot-date table.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "qp/holidays.hpp"

namespace h = qp::holidays;

namespace {

h::Date parse(const std::string& s) {
    return h::ymd(std::stoi(s.substr(0, 4)), unsigned(std::stoi(s.substr(5, 2))),
                  unsigned(std::stoi(s.substr(8, 2))));
}

}  // namespace

TEST(Holidays, MatchesQuantLib2000To2030) {
    std::ifstream in(QP_HOLIDAY_FIXTURE);
    ASSERT_TRUE(in.good()) << QP_HOLIDAY_FIXTURE;
    std::map<std::pair<std::string, int>, std::set<h::Date>> want;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto sp = line.find(' ');
        const h::Date d = parse(line.substr(sp + 1, 10));
        want[{line.substr(0, sp), h::year_of(d)}].insert(d);
    }
    // SIFMA closures QuantLib 1.44 does not list (see holidays.py).
    for (h::Date d : {h::ymd(2001, 9, 11), h::ymd(2001, 9, 12), h::ymd(2025, 1, 9)})
        want[{"USD-GOVT", h::year_of(d)}].insert(d);
    for (auto ccy : h::kCalendarCodes) {
        for (int y = 2000; y <= 2030; ++y) {
            EXPECT_EQ(h::holidays(ccy, y), (want[{std::string(ccy), y}])) << ccy << " " << y;
        }
    }
}

TEST(Holidays, UsdGovtNamedDays) {
    const std::vector<std::string_view> govt{"USD-GOVT"};
    const std::vector<std::string_view> usd{"USD"};
    EXPECT_FALSE(h::is_business_day(h::ymd(2025, 4, 18), govt));  // Good Friday
    EXPECT_TRUE(h::is_business_day(h::ymd(2025, 4, 18), usd));
    EXPECT_TRUE(h::is_business_day(h::ymd(2026, 4, 3), govt));     // payrolls: early close
    EXPECT_FALSE(h::is_business_day(h::ymd(2027, 12, 24), govt));  // Sat Christmas
    EXPECT_EQ(h::holidays("SIFMA", 2025), h::holidays("USD-GOVT", 2025));
}

TEST(Holidays, DatesAndErrors) {
    EXPECT_EQ(h::easter_sunday(2025), h::ymd(2025, 4, 20));
    EXPECT_EQ(h::easter_sunday(2038), h::ymd(2038, 4, 25));
    EXPECT_EQ(h::weekday(h::ymd(2025, 6, 5)), 3u);
    EXPECT_THROW(h::holidays("USD", 1999), std::invalid_argument);
    EXPECT_THROW(h::holidays("NZD", 2025), std::invalid_argument);
}

TEST(Holidays, SpotTable) {
    // Same table as python/fx/04-fixed-income/tests/test_fx_spot_forwards.py
    const char* table[][3] = {
        {"2025-06-05", "EURUSD", "2025-06-09"}, {"2025-04-30", "EURUSD", "2025-05-05"},
        {"2025-07-02", "EURUSD", "2025-07-07"}, {"2025-07-03", "EURUSD", "2025-07-07"},
        {"2025-07-02", "EURJPY", "2025-07-07"}, {"2025-07-02", "GBPJPY", "2025-07-07"},
        {"2025-06-06", "USDCAD", "2025-06-09"}, {"2025-06-30", "USDCAD", "2025-07-02"},
        {"2025-04-16", "GBPUSD", "2025-04-22"}, {"2025-04-17", "GBPUSD", "2025-04-23"},
        {"2025-04-17", "EURUSD", "2025-04-23"}, {"2025-12-23", "EURUSD", "2025-12-29"},
        {"2025-12-24", "GBPUSD", "2025-12-30"}, {"2025-12-30", "USDJPY", "2026-01-06"},
        {"2025-12-31", "USDJPY", "2026-01-06"}, {"2025-11-26", "EURUSD", "2025-11-28"},
        {"2025-11-25", "EURUSD", "2025-11-28"}, {"2025-05-01", "USDJPY", "2025-05-07"},
        {"2025-05-02", "USDJPY", "2025-05-08"}, {"2026-01-29", "EURUSD", "2026-02-02"},
        {"2026-02-26", "USDCHF", "2026-03-02"}, {"2026-03-27", "AUDUSD", "2026-03-31"},
        {"2025-04-24", "AUDUSD", "2025-04-29"}, {"2025-08-29", "GBPUSD", "2025-09-02"},
        {"2025-05-22", "EURCHF", "2025-05-27"}, {"2025-05-28", "EURCHF", "2025-06-02"},
        {"2025-09-26", "USDCAD", "2025-09-29"}, {"2025-10-10", "USDCAD", "2025-10-14"},
        {"2026-06-18", "EURUSD", "2026-06-22"}, {"2025-01-17", "USDJPY", "2025-01-21"},
        {"2025-10-31", "EURUSD", "2025-11-04"}, {"2025-06-17", "USDTRY", "2025-06-18"},
    };
    for (const auto& row : table) {
        const std::string pair = row[1];
        EXPECT_EQ(h::fx_spot_date(parse(row[0]), pair.substr(0, 3), pair.substr(3, 3)),
                  parse(row[2]))
            << row[0] << " " << pair;
    }
    EXPECT_EQ(h::spot_lag("USD", "CAD"), 1);
    EXPECT_EQ(h::spot_lag("EUR", "CAD"), 2);
}

TEST(Holidays, ForwardAndNdfClosedForms) {
    const double s = 1.0850, rd = 0.045, rf = 0.025;
    const h::Date val = h::ymd(2025, 6, 5);
    const double ts = h::yf365(val, h::fx_spot_date(val, "EUR", "USD"));
    EXPECT_NEAR(ts, 4.0 / 365.0, 1e-15);
    const double f = h::forward_flat(s, 1.0, ts, rd, rf);
    EXPECT_NEAR(f,
                h::forward_from_dfs(s, std::exp(-rd), std::exp(-rf), std::exp(-rd * ts),
                                    std::exp(-rf * ts)),
                1e-14);
    EXPECT_NEAR(f, 1.106675868250858, 1e-12);
    const double t = h::yf365(val, h::ymd(2025, 12, 8));
    const double fwd = h::forward_flat(5.40, t, ts, 0.11, 0.045);
    EXPECT_NEAR(h::ndf_value(10e6, 5.55, fwd, std::exp(-0.045 * t), true), 48861.0535389931, 1e-6);
}

TEST(Holidays, FxSwapAndXccyClosedForms) {
    // Same numbers as the Python tests test_fx_swaps_dated.py and
    // test_xccy_initial_exchange.py (EURUSD 1.0850, flat 4.5% / 2.5%).
    const double s = 1.0850, rd = 0.045, rf = 0.025;
    const h::Date val = h::ymd(2025, 6, 5);
    const double ts = h::yf365(val, h::fx_spot_date(val, "EUR", "USD"));
    const double s_today = s * std::exp(-rd * ts) / std::exp(-rf * ts);
    const double f_tom = h::forward_flat(s, 1.0 / 365.0, ts, rd, rf);
    EXPECT_NEAR((s - f_tom) * 1e4, 1.7834150577589192, 1e-10);
    const double tn = h::yf365(val, h::ymd(2025, 6, 9));
    const double tf = h::yf365(val, h::ymd(2026, 6, 9));
    EXPECT_NEAR(h::fx_leg_value(10e6, 1.08, s_today, std::exp(-rf * tn), std::exp(-rd * tn)),
                49975.348544692984, 1e-6);
    EXPECT_NEAR(h::fx_leg_value(-10e6, 1.10, s_today, std::exp(-rf * tf), std::exp(-rd * tf)),
                -66107.63627727056, 1e-6);
    const double t9 = h::yf365(val, h::ymd(2025, 12, 9));
    const double v_near =
        h::fx_leg_value(10e6, 1.07, s_today, std::exp(-rf * tn), std::exp(-rd * tn));
    EXPECT_NEAR(
        h::fx_swap_far_rate(h::forward_flat(s, t9, ts, rd, rf), v_near, 10e6, std::exp(-rd * t9)),
        1.0805921852368006, 1e-13);
    EXPECT_NEAR(h::xccy_initial_exchange(1.0, 11.0e6, 10e6, s, std::exp(-rd * ts)),
                149926.0456340816, 1e-6);
}
