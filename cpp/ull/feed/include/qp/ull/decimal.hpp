// Exact decimal strings to fixed-point integers.
//
// Exchanges send prices and sizes as decimal strings ("27123.45",
// "0.00150000"). Parsing them through double would make "0.1" + "0.2" a
// different level from "0.3". We keep 8 decimal places (1e-8 units, the
// finest increment Coinbase and Binance spot publish) in an int64, which
// holds values up to about 9.2e10 at that scale. More than 8 non-zero
// fractional digits, or a value that overflows, throws instead of rounding.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qp::ull {

inline constexpr int kDecimals = 8;
inline constexpr std::int64_t kScale = 100'000'000;

struct DecimalError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

inline std::int64_t parse_fixed(std::string_view s) {
    if (s.empty()) throw DecimalError("empty decimal");
    bool neg = false;
    std::size_t i = 0;
    if (s[0] == '-') {
        neg = true;
        i = 1;
    }
    if (i >= s.size()) throw DecimalError("bad decimal");
    constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
    std::int64_t ip = 0;
    bool any = false;
    for (; i < s.size() && s[i] != '.'; ++i) {
        char c = s[i];
        if (c < '0' || c > '9') throw DecimalError("bad decimal: " + std::string(s));
        if (ip > (kMax / kScale - 9) / 10)
            throw DecimalError("decimal overflow: " + std::string(s));
        ip = ip * 10 + (c - '0');
        any = true;
    }
    std::int64_t fp = 0;
    int nd = 0;
    if (i < s.size()) {  // at '.'
        ++i;
        if (i >= s.size()) throw DecimalError("bad decimal: " + std::string(s));
        for (; i < s.size(); ++i) {
            char c = s[i];
            if (c < '0' || c > '9') throw DecimalError("bad decimal: " + std::string(s));
            any = true;
            if (nd < kDecimals) {
                fp = fp * 10 + (c - '0');
                ++nd;
            } else if (c != '0')
                throw DecimalError("more than 8 decimals: " + std::string(s));
        }
    }
    if (!any) throw DecimalError("bad decimal: " + std::string(s));
    for (; nd < kDecimals; ++nd) fp *= 10;
    std::int64_t v = ip * kScale + fp;
    return neg ? -v : v;
}

inline std::string format_fixed(std::int64_t v) {
    bool neg = v < 0;
    std::uint64_t u = neg ? std::uint64_t(0) - std::uint64_t(v) : std::uint64_t(v);
    std::string f = std::to_string(u % std::uint64_t(kScale));
    f.insert(0, std::size_t(kDecimals) - f.size(), '0');
    while (!f.empty() && f.back() == '0') f.pop_back();
    std::string o = (neg ? "-" : "") + std::to_string(u / std::uint64_t(kScale));
    return f.empty() ? o : o + "." + f;
}

}  // namespace qp::ull
