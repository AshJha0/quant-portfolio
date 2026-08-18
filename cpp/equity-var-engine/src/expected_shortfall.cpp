#include "eqvar/expected_shortfall.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace eqvar {

double expected_shortfall(std::span<const double> pnl, double alpha) {
    validate_alpha(alpha);
    validate_pnl(pnl, 10);
    std::vector<double> sorted(pnl.begin(), pnl.end());
    std::sort(sorted.begin(), sorted.end());
    const double an = alpha * static_cast<double>(sorted.size());
    const std::size_t k = static_cast<std::size_t>(std::floor(an));
    double tail_sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) tail_sum += sorted[i];
    const double frac = an - static_cast<double>(k);
    if (frac > 0.0 && k < sorted.size()) tail_sum += frac * sorted[k];
    return -tail_sum / an;
}

double normal_es(double sigma, double alpha, double mean) {
    validate_alpha(alpha);
    if (sigma < 0.0) {
        throw std::invalid_argument("normal_es: sigma must be >= 0, got " + std::to_string(sigma));
    }
    const double z = normal_ppf(alpha);
    return sigma * normal_pdf(z) / alpha - mean;
}

double student_t_es(double sigma, double alpha, double df, double mean) {
    validate_alpha(alpha);
    if (sigma < 0.0) {
        throw std::invalid_argument("student_t_es: sigma must be >= 0, got " +
                                    std::to_string(sigma));
    }
    if (df <= 2.0) {
        throw std::invalid_argument("student_t_es: df must be > 2 for finite variance, got " +
                                    std::to_string(df));
    }
    const double q = student_t_ppf(alpha, df);
    const double es_std = student_t_pdf(q, df) * (df + q * q) / ((df - 1.0) * alpha);
    return sigma * es_std * std::sqrt((df - 2.0) / df) - mean;
}

}  // namespace eqvar
