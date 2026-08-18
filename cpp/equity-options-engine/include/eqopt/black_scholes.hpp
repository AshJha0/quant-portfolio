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
///  - negative `S`, `K`, `T` or `sigma` (or NaN) throw std::invalid_argument.
///    Negative `r` and `q` are fully supported.

#ifndef EQOPT_BLACK_SCHOLES_HPP
#define EQOPT_BLACK_SCHOLES_HPP

#include <cmath>
#include <utility>

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
/// \param S     Spot price, must satisfy `S >= 0` and not NaN.
/// \param K     Strike price, must satisfy `K >= 0` and not NaN.
/// \param T     Time to expiry in years (ACT/365F), `T >= 0`, not NaN.
/// \param sigma Annualised log-return volatility, `sigma >= 0`, not NaN.
/// \throws std::invalid_argument if any constraint is violated.
void validate_inputs(double S, double K, double T, double sigma);

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
///       intrinsic on the forward `exp(-rT) max(±(F - K), 0)` — identical
///       to the Python reference edge-case policy.
double bs_price(double S, double K, double T, double r, double sigma,
                double q = 0.0, OptionType type = OptionType::Call);

}  // namespace eqopt

#endif  // EQOPT_BLACK_SCHOLES_HPP
