#include "eqvar/parametric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace eqvar {

double portfolio_sigma(std::span<const double> exposures, const Matrix& cov) {
    if (exposures.empty()) {
        throw std::invalid_argument("portfolio_sigma: empty portfolio (no exposures)");
    }
    if (!cov.square() || cov.rows() != exposures.size()) {
        throw std::invalid_argument("portfolio_sigma: covariance shape does not match " +
                                    std::to_string(exposures.size()) + " exposures");
    }
    const double var = quad_form(exposures, cov);
    double wmax = 0.0;
    for (double w : exposures) wmax = std::max(wmax, std::abs(w));
    if (var < -1e-10 * std::max(1.0, wmax * wmax)) {
        throw std::invalid_argument(
            "portfolio_sigma: covariance matrix is not positive semi-definite (w'Sw < 0)");
    }
    return std::sqrt(std::max(var, 0.0));
}

double parametric_var(std::span<const double> exposures, const Matrix& cov, double alpha,
                      Dist dist, double df, double mean, int horizon_days) {
    validate_alpha(alpha);
    if (horizon_days < 1) {
        throw std::invalid_argument("parametric_var: horizon_days must be >= 1, got " +
                                    std::to_string(horizon_days));
    }
    const double sigma =
        portfolio_sigma(exposures, cov) * std::sqrt(static_cast<double>(horizon_days));
    const double mu = mean * static_cast<double>(horizon_days);
    double z;
    if (dist == Dist::Normal) {
        z = normal_ppf(alpha);
    } else {
        if (df <= 2.0) {
            throw std::invalid_argument(
                "parametric_var: Student-t df must be > 2 for finite variance, got " +
                std::to_string(df));
        }
        // Variance-matched t: quantile rescaled to unit variance.
        z = student_t_ppf(alpha, df) * std::sqrt((df - 2.0) / df);
    }
    return -(mu + z * sigma);
}

double cornish_fisher_z(double z, double skew, double excess_kurt) {
    return z + (z * z - 1.0) * skew / 6.0 + (z * z * z - 3.0 * z) * excess_kurt / 24.0 -
           (2.0 * z * z * z - 5.0 * z) * skew * skew / 36.0;
}

bool cornish_fisher_domain_ok(double skew, double excess_kurt, double z_range, int n_grid) {
    // dz_cf/dz = 1 + zS/3 + (3z^2-3)K/24 - (6z^2-5)S^2/36 must stay > 0.
    for (int i = 0; i < n_grid; ++i) {
        const double z = -z_range + 2.0 * z_range * static_cast<double>(i) /
                                        static_cast<double>(n_grid - 1);
        const double deriv = 1.0 + z * skew / 3.0 + (3.0 * z * z - 3.0) * excess_kurt / 24.0 -
                             (6.0 * z * z - 5.0) * skew * skew / 36.0;
        if (!(deriv > 0.0)) return false;
    }
    return true;
}

double cornish_fisher_var(double sigma, double alpha, double skew, double excess_kurt,
                          double mean, bool check_domain) {
    validate_alpha(alpha);
    if (sigma < 0.0) {
        throw std::invalid_argument("cornish_fisher_var: sigma must be >= 0, got " +
                                    std::to_string(sigma));
    }
    if (check_domain && !cornish_fisher_domain_ok(skew, excess_kurt)) {
        throw std::invalid_argument(
            "cornish_fisher_var: expansion is non-monotone for skew=" + std::to_string(skew) +
            ", excess_kurt=" + std::to_string(excess_kurt) +
            "; outside its validity region the 'quantile' is not a quantile. "
            "Use historical or MC VaR instead.");
    }
    const double z = cornish_fisher_z(normal_ppf(alpha), skew, excess_kurt);
    return -(mean + z * sigma);
}

}  // namespace eqvar
