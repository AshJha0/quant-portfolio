/// \file black_scholes.hpp
/// \brief Black–Scholes–Merton pricing for European equity options.
///
/// C++ twin of the Python reference module
/// `python/equity/01-options-pricing/src/eq_options/black_scholes.py`.
/// Conventions (identical to the Python reference):
///  - rates `r` and dividend yields `q` are continuously compounded,
///    annualised (ACT/365F);
///  - `T` is time to expiry in years; `sigma` is annualised log-return vol;
///  - prices are in the same currency units as `S` and `K`.
///
/// Edge-case policy (documented + unit tested, matches Python exactly):
///  - `T == 0`     -> intrinsic value `max(S - K, 0)` / `max(K - S, 0)`;
///  - `sigma == 0` -> discounted intrinsic on the forward,
///                    `exp(-rT) * max(±(F - K), 0)` with `F = S exp((r-q)T)`;
///  - `K == 0`     -> call is a forward on the stock `S exp(-qT)`; put is 0;
///  - `S == 0`     -> call is 0; put is `K exp(-rT)`;
///  - negative `S`, `K`, `T` or `sigma`, and NaN/Inf in any input (including
///    `r` and `q`), throw std::invalid_argument. Negative *finite* `r` and
///    `q` are fully supported.

#ifndef EQOPT_BLACK_SCHOLES_HPP
#define EQOPT_BLACK_SCHOLES_HPP

#include <cmath>
#include <utility>

#include "eqopt/error.hpp"

namespace eqopt {

/// Option payoff direction.
enum class OptionType { Call, Put };

/// \brief Standard normal CDF via erfc for tail stability.
///
/// `Phi(x) = 0.5 * erfc(-x / sqrt(2))` keeps full relative accuracy in the
/// left tail (down to ~1e-300), where the naive `0.5 * (1 + erf(...))`
/// form loses all precision below ~x = -6.
/// \param x Evaluation point.
/// \return `P(Z <= x)` for `Z ~ N(0, 1)`.
inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * 0.7071067811865475244008443621048490393);
}

/// \brief Standard normal PDF `exp(-x^2/2) / sqrt(2*pi)`.
/// \param x Evaluation point.
/// \return Density value.
inline double norm_pdf(double x) noexcept {
    constexpr double inv_sqrt_2pi = 0.3989422804014326779399460599343819;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

/// \brief Validate common Black–Scholes inputs.
///
/// \param S     Spot price, must satisfy `S >= 0` and be finite.
/// \param K     Strike price, must satisfy `K >= 0` and be finite.
/// \param T     Time to expiry in years (ACT/365F), `T >= 0`, finite.
/// \param sigma Annualised log-return volatility, `sigma >= 0`, finite.
/// \throws std::invalid_argument if any constraint is violated (negative,
///         NaN or +/-Inf).
void validate_inputs(double S, double K, double T, double sigma);

/// \brief Validate rate-like inputs: `r` and `q` may be negative but must
///        be finite (NaN/Inf rejected).
/// \param r Continuously compounded annualised risk-free rate.
/// \param q Continuously compounded annualised dividend yield.
/// \throws std::invalid_argument if `r` or `q` is NaN or +/-Inf.
void validate_rates(double r, double q = 0.0);

/// \brief Intrinsic (exercise-now) value `max(S - K, 0)` or `max(K - S, 0)`.
/// \param S,K  Spot and strike, currency units.
/// \param type Payoff direction.
/// \return Intrinsic value in currency units.
inline double intrinsic_value(double S, double K,
                              OptionType type = OptionType::Call) noexcept {
    return type == OptionType::Call ? std::fmax(S - K, 0.0)
                                    : std::fmax(K - S, 0.0);
}

/// \brief Equity forward `F = S * exp((r - q) * T)`.
/// \param S Spot price.
/// \param T Time to delivery in years (ACT/365F).
/// \param r Continuously compounded annualised risk-free rate.
/// \param q Continuously compounded annualised dividend yield.
/// \return Forward price in currency units.
inline double forward_price(double S, double T, double r,
                            double q = 0.0) noexcept {
    return S * std::exp((r - q) * T);
}

/// \brief Black–Scholes `d1` and `d2` terms.
///
/// `d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))`,
/// `d2 = d1 - sigma sqrt(T)`.
/// \param S,K   Spot and strike, strictly positive.
/// \param T     Time to expiry in years, strictly positive.
/// \param r     Continuously compounded annualised risk-free rate.
/// \param sigma Annualised volatility, strictly positive.
/// \param q     Continuously compounded annualised dividend yield.
/// \return Pair `(d1, d2)`.
/// \throws std::invalid_argument if `S`, `K`, `T` or `sigma` is not
///         strictly positive (or any input is negative/NaN).
std::pair<double, double> d1_d2(double S, double K, double T, double r,
                                double sigma, double q = 0.0);

/// \brief Black–Scholes–Merton price of a European option with dividend yield.
///
/// \param S     Spot price (currency units), `S >= 0`.
/// \param K     Strike price (currency units), `K >= 0`.
/// \param T     Time to expiry in years (ACT/365F), `T >= 0`.
/// \param r     Continuously compounded annualised risk-free rate
///              (negative rates supported).
/// \param sigma Annualised log-return volatility, `sigma >= 0`.
/// \param q     Continuously compounded annualised dividend yield.
/// \param type  Option payoff direction.
/// \return Present value of the option in currency units.
/// \throws std::invalid_argument if `S`, `K`, `T` or `sigma` is negative
///         or NaN.
/// \note `T == 0` returns intrinsic; `sigma == 0` returns the discounted
///       intrinsic on the forward `exp(-rT) max(±(F - K), 0)`: identical
///       to the Python reference edge-case policy.
double bs_price(double S, double K, double T, double r, double sigma,
                double q = 0.0, OptionType type = OptionType::Call);

/// \brief Return `value`, or throw std::invalid_argument naming `what` if it
/// is not finite (a discount factor or forward overflowed for finite inputs).
double require_finite_result(double value, const char* what);

// ------------------------------------------------------------------------
// noexcept kernel API. Same conventions, edge cases and arithmetic as the
// throwing functions above (which are thin wrappers over these), so the
// returned values are bit-for-bit the same. No allocation, no exceptions.
// ------------------------------------------------------------------------

/// \brief Check `x >= 0` and finite; on failure the error names `field`.
[[nodiscard]] inline Result<void> try_check_nonneg(double x, const char* field) noexcept {
    if (!std::isfinite(x) || x < 0.0) [[unlikely]] {
        return std::unexpected(Error{Errc::NegativeOrNonFinite, field, x});
    }
    return {};
}

#if defined(__GNUC__) || defined(__clang__)
#define EQOPT_COLD [[gnu::cold, gnu::noinline]]
#elif defined(_MSC_VER)
#define EQOPT_COLD __declspec(noinline)
#else
#define EQOPT_COLD
#endif

namespace detail {
/// `std::isfinite(x) && x >= 0` as one comparison pair (false for NaN).
inline bool nonneg_finite(double x) noexcept { return x >= 0.0 && x <= 1.7976931348623157e308; }
/// Out-of-line builders for the error values, so the inlined success path
/// carries only the comparisons.
EQOPT_COLD Error input_error(double S, double K, double T, double sigma) noexcept;
EQOPT_COLD Error rate_error(double r, double q) noexcept;
}  // namespace detail

/// \brief noexcept form of validate_inputs().
[[nodiscard]] inline Result<void> try_validate_inputs(double S, double K, double T,
                                                      double sigma) noexcept {
    if (detail::nonneg_finite(S) && detail::nonneg_finite(K) && detail::nonneg_finite(T) &&
        detail::nonneg_finite(sigma)) [[likely]] {
        return {};
    }
    return std::unexpected(detail::input_error(S, K, T, sigma));
}

/// \brief noexcept form of validate_rates().
[[nodiscard]] inline Result<void> try_validate_rates(double r, double q = 0.0) noexcept {
    if (std::fabs(r) <= 1.7976931348623157e308 && std::fabs(q) <= 1.7976931348623157e308)
        [[likely]] {
        return {};
    }
    return std::unexpected(detail::rate_error(r, q));
}

/// \brief noexcept form of require_finite_result().
[[nodiscard]] inline Result<double> try_finite(double value, const char* what) noexcept {
    if (!std::isfinite(value)) [[unlikely]] {
        return std::unexpected(Error{Errc::NonFiniteResult, what});
    }
    return value;
}

/// \brief `(d1, d2)` without validation; caller guarantees S, K, T, sigma > 0.
inline std::pair<double, double> d1_d2_unchecked(double S, double K, double T, double r,
                                                 double sigma, double q) noexcept {
    const double sqrt_t = std::sqrt(T);
    const double d1 =
        (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t);
    return {d1, d1 - sigma * sqrt_t};
}

/// \brief noexcept form of d1_d2().
[[nodiscard]] inline Result<std::pair<double, double>> try_d1_d2(
    double S, double K, double T, double r, double sigma, double q = 0.0) noexcept {
    if (auto e = try_validate_inputs(S, K, T, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (auto e = try_validate_rates(r, q); !e) [[unlikely]] return std::unexpected(e.error());
    if (S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0) [[unlikely]] {
        return std::unexpected(Error{Errc::NotStrictlyPositive, "", S, K, T, sigma});
    }
    return d1_d2_unchecked(S, K, T, r, sigma, q);
}

/// \brief Black-Scholes-Merton price on inputs the caller has validated
///        (finite; S, K, T, sigma >= 0; r, q finite). Returns the raw value,
///        which is non-finite if a discount factor overflows.
inline double bs_price_unchecked(double S, double K, double T, double r, double sigma,
                                 double q, OptionType type) noexcept {
    if (T == 0.0) [[unlikely]] return intrinsic_value(S, K, type);
    const double disc_s = S * std::exp(-q * T);
    const double disc_k = K * std::exp(-r * T);
    const bool call = type == OptionType::Call;
    if (K == 0.0) [[unlikely]] return call ? disc_s : 0.0;
    if (S == 0.0) [[unlikely]] return call ? 0.0 : disc_k;
    if (sigma == 0.0) [[unlikely]] {
        const double sign = call ? 1.0 : -1.0;
        return std::fmax(sign * (disc_s - disc_k), 0.0);
    }
    const auto [d1, d2] = d1_d2_unchecked(S, K, T, r, sigma, q);
    return call ? disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
                : disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1);
}

/// \brief noexcept form of bs_price(): the same value, or the error the
///        throwing form raises.
[[nodiscard]] inline Result<double> try_bs_price(double S, double K, double T, double r,
                                                 double sigma, double q = 0.0,
                                                 OptionType type = OptionType::Call) noexcept {
    if (auto e = try_validate_inputs(S, K, T, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (auto e = try_validate_rates(r, q); !e) [[unlikely]] return std::unexpected(e.error());
    return try_finite(bs_price_unchecked(S, K, T, r, sigma, q, type), "bs_price");
}

}  // namespace eqopt

#endif  // EQOPT_BLACK_SCHOLES_HPP
