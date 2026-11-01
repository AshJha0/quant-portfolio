#include "eqvar/expected_shortfall.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "eqvar/historical.hpp"
#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace eqvar {

double expected_shortfall(std::span<const double> pnl, double alpha) {
    validate_alpha(alpha);
    validate_pnl(pnl, 10);
    return var_es_order_statistic(pnl, alpha).es;
}

namespace {

void check_scale(double sigma, double mean, const char* who) {
    if (!(sigma >= 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument(std::string(who) + ": sigma must be finite and >= 0, got " +
                                    std::to_string(sigma));
    }
    if (!std::isfinite(mean)) {
        throw std::invalid_argument(std::string(who) + ": mean must be finite");
    }
}

}  // namespace

double normal_es(double sigma, double alpha, double mean) {
    validate_alpha(alpha);
    check_scale(sigma, mean, "normal_es");
    const double z = normal_ppf(alpha);
    return sigma * normal_pdf(z) / alpha - mean;
}

double student_t_es(double sigma, double alpha, double df, double mean) {
    validate_alpha(alpha);
    check_scale(sigma, mean, "student_t_es");
    if (!(df > 2.0) || !std::isfinite(df)) {
        throw std::invalid_argument("student_t_es: df must be > 2 for finite variance, got " +
                                    std::to_string(df));
    }
    const double q = student_t_ppf(alpha, df);
    const double es_std = student_t_pdf(q, df) * (df + q * q) / ((df - 1.0) * alpha);
    return sigma * es_std * std::sqrt((df - 2.0) / df) - mean;
}

}  // namespace eqvar
