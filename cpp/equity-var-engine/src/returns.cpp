#include "eqvar/returns.hpp"

#include "eqvar/kernels.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace eqvar {

namespace {

void check_prices(std::span<const double> prices, const char* who) {
    if (prices.size() < 2) {
        throw std::invalid_argument(std::string(who) + ": need at least 2 prices, got " +
                                    std::to_string(prices.size()));
    }
    for (double p : prices) {
        if (!(p > 0.0) || !std::isfinite(p)) {
            throw std::invalid_argument(std::string(who) + ": prices must be positive and finite");
        }
    }
}

}  // namespace

std::vector<double> log_returns(std::span<const double> prices) {
    check_prices(prices, "log_returns");
    std::vector<double> r(prices.size() - 1);
    for (std::size_t t = 1; t < prices.size(); ++t) r[t - 1] = std::log(prices[t] / prices[t - 1]);
    return r;
}

std::vector<double> simple_returns(std::span<const double> prices) {
    check_prices(prices, "simple_returns");
    std::vector<double> r(prices.size() - 1);
    for (std::size_t t = 1; t < prices.size(); ++t) r[t - 1] = prices[t] / prices[t - 1] - 1.0;
    return r;
}

std::vector<double> portfolio_pnl(const Matrix& returns, std::span<const double> exposures) {
    validate_exposures(exposures, "portfolio_pnl");
    if (returns.cols() != exposures.size()) {
        throw std::invalid_argument("portfolio_pnl: panel has " + std::to_string(returns.cols()) +
                                    " factors, portfolio has " + std::to_string(exposures.size()) +
                                    " exposures");
    }
    for (std::size_t i = 0; i < returns.size(); ++i) {
        if (!std::isfinite(returns.data()[i])) {
            throw std::invalid_argument("portfolio_pnl: returns contain NaN or inf");
        }
    }
    return matvec(returns, exposures);
}

void validate_pnl(std::span<const double> pnl, std::size_t min_obs) {
    value_or_throw(try_validate_pnl(pnl, min_obs));
}

void validate_alpha(double alpha) { value_or_throw(try_validate_alpha(alpha)); }

void validate_exposures(std::span<const double> exposures, const char* who) {
    value_or_throw(try_validate_exposures(exposures, who));
}

}  // namespace eqvar
