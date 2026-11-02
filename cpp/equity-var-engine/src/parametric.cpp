#include "eqvar/parametric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "eqvar/kernels.hpp"
#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace eqvar {

double portfolio_sigma(std::span<const double> exposures, const Matrix& cov) {
    return value_or_throw(try_portfolio_sigma(exposures, cov));
}

double parametric_var(std::span<const double> exposures, const Matrix& cov, double alpha, Dist dist,
                      double df, double mean, int horizon_days) {
    return value_or_throw(try_parametric_var(exposures, cov, alpha, dist, df, mean, horizon_days));
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
        throw std::invalid_argument(
            "cornish_fisher_valid: skew, excess_kurt and alpha must be finite");
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

double cornish_fisher_var(double sigma, double alpha, double skew, double excess_kurt, double mean,
                          bool check_domain) {
    validate_alpha(alpha);
    if (!(sigma >= 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("cornish_fisher_var: sigma must be finite and >= 0, got " +
                                    std::to_string(sigma));
    }
    if (!std::isfinite(mean))
        throw std::invalid_argument("cornish_fisher_var: mean must be finite");
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
