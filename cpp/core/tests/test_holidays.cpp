// Parity of the C++ calendar port with an independent oracle (QuantLib
// 1.44 holiday lists 2000-2030, the same extract the Rust tests use) and
// with the Python spot-date table.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>

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
        const h::Date d = parse(line.substr(4, 10));
        want[{line.substr(0, 3), h::year_of(d)}].insert(d);
    }
    for (auto ccy : h::kCurrencies) {
        for (int y = 2000; y <= 2030; ++y) {
            EXPECT_EQ(h::holidays(ccy, y), (want[{std::string(ccy), y}])) << ccy << " " << y;
        }
    }
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
