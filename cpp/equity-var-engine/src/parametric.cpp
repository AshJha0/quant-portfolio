#include "eqvar/parametric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace eqvar {

double portfolio_sigma(std::span<const double> exposures, const Matrix& cov) {
    validate_exposures(exposures, "portfolio_sigma");
    if (!cov.square() || cov.rows() != exposures.size()) {
        throw std::invalid_argument("portfolio_sigma: covariance shape does not match " +
                                    std::to_string(exposures.size()) + " exposures");
    }
    for (double w : exposures) {
        if (!std::isfinite(w)) {
            throw std::invalid_argument("portfolio_sigma: exposures must be finite");
        }
    }
    const double var = quad_form(exposures, cov);
    // NaN fails every ordered comparison below, so a non-finite covariance
    // would otherwise return a NaN sigma and poison the whole VaR report.
    if (!std::isfinite(var)) {
        throw std::invalid_argument(
            "portfolio_sigma: w'Sw is not finite (covariance contains NaN/Inf "
            "or overflows)");
    }
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
    if (!std::isfinite(mean)) throw std::invalid_argument("parametric_var: mean must be finite");
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
        if (!(df > 2.0) || !std::isfinite(df)) {
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
    if (!std::isfinite(z) || !std::isfinite(skew) || !std::isfinite(excess_kurt)) {
        throw std::invalid_argument("cornish_fisher_z: inputs must be finite");
    }
    return z + (z * z - 1.0) * skew / 6.0 + (z * z * z - 3.0 * z) * excess_kurt / 24.0 -
           (2.0 * z * z * z - 5.0 * z) * skew * skew / 36.0;
}

namespace {

// Minimum of g'(z) = A z^2 + B z + C on [-r, r] for the CF map
// g(z) = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36:
// A = K/8 - S^2/6, B = S/3, C = 1 - K/8 + 5 S^2/36. The minimum is at the
// vertex -B/(2A) when A > 0 and the vertex lies inside, else at an endpoint.
double cf_min_derivative(double skew, double excess_kurt, double r) {
    const double a = excess_kurt / 8.0 - skew * skew / 6.0;
    const double b = skew / 3.0;
    const double c = 1.0 - excess_kurt / 8.0 + 5.0 * skew * skew / 36.0;
    const auto g = [&](double z) { return a * z * z + b * z + c; };
    double m = std::min(g(-r), g(r));
    if (a > 0.0) {
        const double z_star = -b / (2.0 * a);
        if (z_star >= -r && z_star <= r) m = std::min(m, g(z_star));
    }
    return m;
}

}  // namespace

bool cornish_fisher_valid(double skew, double excess_kurt, double alpha) {
    if (!std::isfinite(skew) || !std::isfinite(excess_kurt) || !std::isfinite(alpha)) {
        throw std::invalid_argument("cornish_fisher_valid: skew, excess_kurt and alpha must be finite");
    }
    validate_alpha(alpha);
    const double z_a = normal_ppf(1.0 - alpha);
    return cf_min_derivative(skew, excess_kurt, z_a) > 0.0;
}

bool cornish_fisher_domain_ok(double skew, double excess_kurt, double z_range, int n_grid) {
    if (!(z_range > 0.0) || !std::isfinite(z_range)) {
        throw std::invalid_argument("cornish_fisher_domain_ok: z_range must be finite and > 0");
    }
    if (n_grid < 2) throw std::invalid_argument("cornish_fisher_domain_ok: n_grid must be >= 2");
    if (!std::isfinite(skew) || !std::isfinite(excess_kurt)) {
        throw std::invalid_argument(
            "cornish_fisher_domain_ok: skew and excess_kurt must be finite");
    }
    return cf_min_derivative(skew, excess_kurt, z_range) > 0.0;
}

double cornish_fisher_var(double sigma, double alpha, double skew, double excess_kurt,
                          double mean, bool check_domain) {
    validate_alpha(alpha);
    if (!(sigma >= 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("cornish_fisher_var: sigma must be finite and >= 0, got " +
                                    std::to_string(sigma));
    }
    if (!std::isfinite(mean)) throw std::invalid_argument("cornish_fisher_var: mean must be finite");
    // Non-finite moments throw inside cornish_fisher_valid even when the
    // domain check is off.
    const bool valid = cornish_fisher_valid(skew, excess_kurt, alpha);
    if (check_domain && !valid) {
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
