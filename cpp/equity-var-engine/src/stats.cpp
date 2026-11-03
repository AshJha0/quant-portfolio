#include "eqvar/stats.hpp"

// Wrappers over qp/var_stats.hpp, the checked layer shared with fxvar
// (plan W20, Round 8). The numbers did not change: the shared layer is this
// file's former code.

#include <stdexcept>

#include "qp/normal.hpp"
#include "qp/var_stats.hpp"

namespace eqvar {

double normal_pdf(double x) noexcept { return qp::norm_pdf(x); }
double normal_cdf(double x) noexcept { return qp::norm_cdf(x); }
double normal_ppf(double p) { return qp::var::normal_ppf(p); }
double log_gamma(double x) { return qp::var::log_gamma(x); }
double betainc_reg(double a, double b, double x) { return qp::var::betainc_reg(a, b, x); }
double regularized_gamma_p(double a, double x) { return qp::var::regularized_gamma_p(a, x); }
double regularized_gamma_q(double a, double x) { return qp::var::regularized_gamma_q(a, x); }
double chi2_sf(double x, double df) { return qp::var::chi2_sf(x, df); }

double binomial_cdf(int k, int n, double p) {
    if (n < 1) throw std::invalid_argument("binomial_cdf: n must be >= 1");
    return qp::var::binomial_cdf(k, n, p);
}

double student_t_pdf(double x, double df) { return qp::var::student_t_pdf(x, df); }
double student_t_cdf(double x, double df) { return qp::var::student_t_cdf(x, df); }
double student_t_ppf(double p, double df) { return qp::var::student_t_ppf(p, df); }
double mean(std::span<const double> x) { return qp::var::mean(x); }

double stdev(std::span<const double> x) {
    if (x.size() < 2) throw std::invalid_argument("stdev: need at least 2 observations");
    return qp::var::moments(x).stdev;
}

double skewness(std::span<const double> x) {
    if (x.size() < 3) throw std::invalid_argument("skewness: need at least 3 observations");
    return qp::var::moments(x).skewness;
}

double excess_kurtosis(std::span<const double> x) {
    if (x.size() < 4) throw std::invalid_argument("excess_kurtosis: need at least 4 observations");
    return qp::var::moments(x).excess_kurtosis;
}

}  // namespace eqvar
