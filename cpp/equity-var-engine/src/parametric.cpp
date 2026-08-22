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
    if (!(z_range > 0.0) || !std::isfinite(z_range)) {
        throw std::invalid_argument("cornish_fisher_domain_ok: z_range must be finite and > 0");
    }
    if (n_grid < 2) {
        // Retained for API compatibility; the check below is closed-form
        // and no longer samples a grid, but n_grid keeps its validation.
        throw std::invalid_argument("cornish_fisher_domain_ok: n_grid must be >= 2");
    }
    if (!std::isfinite(skew) || !std::isfinite(excess_kurt)) {
        throw std::invalid_argument(
            "cornish_fisher_domain_ok: skew and excess_kurt must be finite");
    }
    // dz_cf/dz = 1 + zS/3 + (3z^2-3)K/24 - (6z^2-5)S^2/36 must stay > 0 on
    // [-z_range, z_range]. This derivative is itself a quadratic in z,
    // g(z) = A z^2 + B z + C with A = K/8 - S^2/6, B = S/3,
    // C = 1 - K/8 + 5 S^2/36, so its exact minimum on the interval is
    // known in closed form: the vertex -B/(2A) when g is convex (A > 0)
    // and that point lies in range, else an interval endpoint (a
    // concave/linear g, A <= 0, always attains its minimum at an
    // endpoint). This replaces a fixed-resolution grid scan, which can
    // miss a thin sub-grid dip below zero (verified against a closed-form
    // counterexample: skew=0.122, excess_kurt=-0.427 is non-monotone on
    // |z| <= 4 but an 801-point grid there reports it as monotone).
    const double a = excess_kurt / 8.0 - skew * skew / 6.0;
    const double b = skew / 3.0;
    const double c = 1.0 - excess_kurt / 8.0 + 5.0 * skew * skew / 36.0;
    const auto g = [&](double z) { return a * z * z + b * z + c; };
    double m;
    if (a > 0.0) {
        const double z_star = -b / (2.0 * a);
        m = (z_star >= -z_range && z_star <= z_range) ? g(z_star)
                                                       : std::min(g(-z_range), g(z_range));
    } else {
        m = std::min(g(-z_range), g(z_range));
    }
    return m > 0.0;
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
