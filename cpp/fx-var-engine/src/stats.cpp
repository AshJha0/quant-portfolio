#include "fxvar/stats.hpp"

// Wrappers over qp/var_stats.hpp, the checked layer shared with eqvar
// (plan W20, Round 8). That layer uses the equity solvers and tolerances;
// the FX numbers moved by at most 2.2e-12 relative (norm_ppf near 1),
// see MIGRATION.md.

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "qp/normal.hpp"
#include "qp/var_stats.hpp"

namespace fxvar {

double validate_alpha(double alpha) {
  if (!(alpha > 0.0 && alpha < 0.5)) {
    std::ostringstream msg;
    msg << "alpha is the tail probability (e.g. 0.01 for 99 %); must be in "
           "(0, 0.5), got "
        << alpha;
    throw std::invalid_argument(msg.str());
  }
  return alpha;
}

double validate_horizon(double horizon_days) {
  if (!(horizon_days > 0.0) || !std::isfinite(horizon_days)) {
    std::ostringstream msg;
    msg << "horizon_days must be > 0, got " << horizon_days;
    throw std::invalid_argument(msg.str());
  }
  return horizon_days;
}

double log_gamma(double x) { return qp::var::log_gamma(x); }
double norm_pdf(double x) { return qp::norm_pdf(x); }
double norm_cdf(double x) { return qp::norm_cdf(x); }
double norm_ppf(double p) { return qp::var::normal_ppf(p); }
double reg_lower_gamma(double a, double x) { return qp::var::regularized_gamma_p(a, x); }
double reg_upper_gamma(double a, double x) { return qp::var::regularized_gamma_q(a, x); }
double chi2_sf(double x, double df) { return qp::var::chi2_sf(x, df); }
double reg_inc_beta(double a, double b, double x) { return qp::var::betainc_reg(a, b, x); }
double t_pdf(double x, double df) { return qp::var::student_t_pdf(x, df); }
double t_cdf(double x, double df) { return qp::var::student_t_cdf(x, df); }
double t_ppf(double p, double df) { return qp::var::student_t_ppf(p, df); }
double binom_cdf(int k, int n, double p) { return qp::var::binomial_cdf(k, n, p); }

Moments sample_moments(const std::vector<double>& x) {
  const qp::var::Moments m = qp::var::moments(x);
  return Moments{m.mean, m.stdev, m.skewness, m.excess_kurtosis};
}

double sample_std(const std::vector<double>& x) { return sample_moments(x).stdev; }

}  // namespace fxvar
