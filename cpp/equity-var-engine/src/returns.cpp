#include "eqvar/returns.hpp"

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
    if (exposures.empty()) {
        throw std::invalid_argument(
            "portfolio_pnl: empty portfolio (no exposures); nothing to revalue");
    }
    if (returns.cols() != exposures.size()) {
        throw std::invalid_argument("portfolio_pnl: panel has " + std::to_string(returns.cols()) +
                                    " factors, portfolio has " + std::to_string(exposures.size()) +
                                    " exposures");
    }
    return matvec(returns, exposures);
}

void validate_pnl(std::span<const double> pnl, std::size_t min_obs) {
    if (pnl.size() < min_obs) {
        throw std::invalid_argument("need at least " + std::to_string(min_obs) +
                                    " P&L observations, got " + std::to_string(pnl.size()) +
                                    "; empirical tail quantiles are meaningless on shorter samples");
    }
    for (double v : pnl) {
        if (!std::isfinite(v)) throw std::invalid_argument("pnl contains NaN or infinite values");
    }
}

void validate_alpha(double alpha) {
    if (!(alpha > 0.0 && alpha < 0.5)) {
        throw std::invalid_argument("alpha must be in (0, 0.5) (tail probability), got " +
                                    std::to_string(alpha));
    }
}

}  // namespace eqvar
