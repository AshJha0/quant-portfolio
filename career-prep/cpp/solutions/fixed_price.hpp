// C18. Fixed-point price: an integer count of ticks of 0.0001.
//
// Binary floating point cannot hold 0.1 exactly, so sums of prices drift
// and equality tests on prices fail. Venues quote on a tick grid, so an
// integer count of ticks is exact. Arithmetic is overflow-checked with the
// GCC/Clang builtins.
#pragma once
#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace cp {

class Price {
public:
    static constexpr std::int64_t kScale = 10000;  // 4 decimal places

    static constexpr Price from_ticks(std::int64_t t) { return Price(t); }

    // Parses "[-]digits[.digits]" with at most 4 decimals; nullopt otherwise.
    static std::optional<Price> parse(std::string_view s) {
        const bool neg = !s.empty() && s[0] == '-';
        if (neg) s.remove_prefix(1);
        std::int64_t whole = 0, frac = 0;
        int frac_digits = -1;  // -1: no '.' seen yet
        bool any_digit = false;
        for (char c : s) {
            if (c == '.') {
                if (frac_digits >= 0) return std::nullopt;
                frac_digits = 0;
                continue;
            }
            if (c < '0' || c > '9') return std::nullopt;
            any_digit = true;
            const int d = c - '0';
            if (frac_digits >= 0) {
                if (++frac_digits > 4) return std::nullopt;
                frac = frac * 10 + d;
            } else if (__builtin_mul_overflow(whole, 10, &whole) || __builtin_add_overflow(whole, d, &whole)) {
                return std::nullopt;
            }
        }
        if (!any_digit) return std::nullopt;
        for (int i = frac_digits < 0 ? 0 : frac_digits; i < 4; ++i) frac *= 10;
        std::int64_t t = 0;
        if (__builtin_mul_overflow(whole, kScale, &t) || __builtin_add_overflow(t, frac, &t)) return std::nullopt;
        return Price(neg ? -t : t);
    }

    constexpr std::int64_t ticks() const { return t_; }

    std::optional<Price> checked_add(Price o) const {
        std::int64_t r = 0;
        if (__builtin_add_overflow(t_, o.t_, &r)) return std::nullopt;
        return Price(r);
    }
    std::optional<Price> checked_mul(std::int64_t qty) const {
        std::int64_t r = 0;
        if (__builtin_mul_overflow(t_, qty, &r)) return std::nullopt;
        return Price(r);
    }

    std::string str() const {
        const std::uint64_t mag = t_ < 0 ? 0 - static_cast<std::uint64_t>(t_) : static_cast<std::uint64_t>(t_);
        const std::string frac = std::to_string(mag % kScale);
        return (t_ < 0 ? "-" : "") + std::to_string(mag / kScale) + "." + std::string(4 - frac.size(), '0') + frac;
    }

    friend constexpr bool operator==(Price, Price) = default;
    friend constexpr auto operator<=>(Price, Price) = default;

private:
    constexpr explicit Price(std::int64_t t) : t_(t) {}
    std::int64_t t_;
};

}  // namespace cp
