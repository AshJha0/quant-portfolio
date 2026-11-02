#include "fxvar/parametric.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/kernels.hpp"
#include "fxvar/stats.hpp"

namespace fxvar {

double portfolio_sigma(const std::vector<double>& exposures, const Matrix& cov) {
  return value_or_throw(try_portfolio_sigma(exposures, cov));
}

VarEs var_covar(const std::vector<double>& exposures, const Matrix& cov, double alpha,
                double horizon_days, TailDist dist, double df, double mean) {
  return value_or_throw(try_var_covar(exposures, cov, alpha, horizon_days, dist, df, mean));
}

ParametricResult parametric_var(const Book& book, const Market& market,
                                const ReturnsMatrix& returns, const ParametricOptions& opts) {
  validate_alpha(opts.alpha);
  validate_horizon(opts.horizon_days);
  const CompiledBook compiled(book, market);  // throws on empty book

  ParametricResult res;
  res.alpha = opts.alpha;
  res.horizon_days = opts.horizon_days;
  res.dist = opts.dist;
  res.factors = compiled.factors();
  if (res.factors.empty()) return res;  // pure base-ccy cash: zero risk

  validate_returns(returns, res.factors, opts.min_obs);
  const ReturnsMatrix rets = returns.select(res.factors);

  if (opts.warn_pegs) {
    res.flagged_peg_factors = flag_peg_factors(rets);
    for (const auto& f : res.flagged_peg_factors) {
      std::ostringstream msg;
      msg << "peg blindness: factor " << f << " has daily vol < " << kPegVolThreshold
          << " (pegged/managed currency). Historical and parametric VaR are "
             "blind to peg-break risk; add the peg-break stress add-on "
             "(fxvar::peg_break_scenario).";
      res.warnings.push_back(msg.str());
    }
  }

  const FactorCov cov =
      (opts.cov_method == CovMethod::kSample) ? sample_cov(rets) : ewma_cov(rets, opts.ewma_lambda);
  res.exposures = compiled.linear_exposures();
  const VarEs ve =
      var_covar(res.exposures, cov.cov, opts.alpha, opts.horizon_days, opts.dist, opts.df);
  res.var = ve.var;
  res.es = ve.es;
  res.sigma = portfolio_sigma(res.exposures, cov.cov);
  return res;
}

// ------------------------------------------------------- Cornish-Fisher
double cornish_fisher_z(double z, double skew, double excess_kurtosis) {
  if (!std::isfinite(z) || !std::isfinite(skew) || !std::isfinite(excess_kurtosis))
    throw std::invalid_argument("cornish_fisher_z: inputs must be finite");
  const double z2 = z * z;
  const double z3 = z2 * z;
  return z + (z2 - 1.0) * skew / 6.0 + (z3 - 3.0 * z) * excess_kurtosis / 24.0 -
         (2.0 * z3 - 5.0 * z) * skew * skew / 36.0;
}

namespace {

// Minimum of g'(z) = A z^2 + B z + C on [-r, r] for the CF map, with
// A = K/8 - S^2/6, B = S/3, C = 1 - K/8 + 5 S^2/36: the vertex -B/(2A) when
// A > 0 and it lies inside, else an endpoint.
double cf_min_derivative(double skew, double k, double r) {
  const double a = k / 8.0 - skew * skew / 6.0;
  const double b = skew / 3.0;
  const double c = 1.0 - k / 8.0 + 5.0 * skew * skew / 36.0;
  const auto g = [&](double z) { return a * z * z + b * z + c; };
  double m = std::min(g(-r), g(r));
  if (a > 0.0) {
    const double z_star = -b / (2.0 * a);
    if (z_star >= -r && z_star <= r) m = std::min(m, g(z_star));
  }
  return m;
}

}  // namespace

bool cornish_fisher_valid(double skew, double excess_kurtosis, double alpha) {
  if (!std::isfinite(skew) || !std::isfinite(excess_kurtosis) || !std::isfinite(alpha))
    throw std::invalid_argument(
        "cornish_fisher_valid: skew, excess_kurtosis and alpha must be finite");
  validate_alpha(alpha);
  return cf_min_derivative(skew, excess_kurtosis, norm_ppf(1.0 - alpha)) > 0.0;
}

bool cornish_fisher_domain_ok(double skew, double excess_kurtosis, double z_range, int n_grid) {
  if (n_grid < 3) throw std::invalid_argument("n_grid must be >= 3");
  if (!(z_range > 0.0) || !std::isfinite(z_range))
    throw std::invalid_argument("z_range must be finite and > 0");
  if (!std::isfinite(skew) || !std::isfinite(excess_kurtosis))
    throw std::invalid_argument("skew and excess_kurtosis must be finite");
  return cf_min_derivative(skew, excess_kurtosis, z_range) > 0.0;
}

double cornish_fisher_var(double sigma, double skew, double excess_kurtosis, double alpha,
                          double mean, double horizon_days, bool check_domain) {
  validate_alpha(alpha);
  validate_horizon(horizon_days);
  if (!(sigma >= 0.0) || !std::isfinite(sigma))
    throw std::invalid_argument("sigma must be finite and >= 0");
  if (!std::isfinite(mean)) throw std::invalid_argument("mean must be finite");
  const bool valid = cornish_fisher_valid(skew, excess_kurtosis, alpha);
  if (check_domain && !valid) {
    std::ostringstream msg;
    msg << "Cornish-Fisher expansion is non-monotone for skew=" << skew
        << ", excess_kurtosis=" << excess_kurtosis
        << ": outside validity domain; fall back to historical or t VaR "
           "(set check_domain=false to force)";
    throw std::invalid_argument(msg.str());
  }
  // Loss quantile: lower tail of the P&L distribution.
  const double z = norm_ppf(alpha);  // lower-tail P&L quantile
  const double zcf = cornish_fisher_z(z, skew, excess_kurtosis);
  const double scale = std::sqrt(horizon_days);
  return -(mean * horizon_days) - sigma * scale * zcf;
}

}  // namespace fxvar
