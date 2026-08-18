#include "fxvar/parametric.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/stats.hpp"

namespace fxvar {

double portfolio_sigma(const std::vector<double>& exposures, const Matrix& cov) {
  const double var = quad_form(exposures, cov);
  if (var < -1e-12)
    throw std::invalid_argument(
        "portfolio_sigma: covariance matrix is not positive semi-definite");
  return std::sqrt(std::max(var, 0.0));
}

VarEs var_covar(const std::vector<double>& exposures, const Matrix& cov,
                double alpha, double horizon_days, TailDist dist, double df,
                double mean) {
  validate_alpha(alpha);
  validate_horizon(horizon_days);
  const double sig1 = portfolio_sigma(exposures, cov);
  const double scale = std::sqrt(horizon_days);
  const double sig = sig1 * scale;
  const double mu = mean * horizon_days;
  if (dist == TailDist::kNormal)
    return {normal_var(sig, alpha, mu), normal_es(sig, alpha, mu)};
  return {t_var(sig, alpha, df, mu), t_es(sig, alpha, df, mu)};
}

ParametricResult parametric_var(const Book& book, const Market& market,
                                const ReturnsMatrix& returns,
                                const ParametricOptions& opts) {
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
      msg << "peg blindness: factor " << f << " has daily vol < "
          << kPegVolThreshold
          << " (pegged/managed currency). Historical and parametric VaR are "
             "blind to peg-break risk; add the peg-break stress add-on "
             "(fxvar::peg_break_scenario).";
      res.warnings.push_back(msg.str());
    }
  }

  const FactorCov cov = (opts.cov_method == CovMethod::kSample)
                            ? sample_cov(rets)
                            : ewma_cov(rets, opts.ewma_lambda);
  res.exposures = compiled.linear_exposures();
  const VarEs ve = var_covar(res.exposures, cov.cov, opts.alpha,
                             opts.horizon_days, opts.dist, opts.df);
  res.var = ve.var;
  res.es = ve.es;
  res.sigma = portfolio_sigma(res.exposures, cov.cov);
  return res;
}

// ------------------------------------------------------- Cornish-Fisher
double cornish_fisher_z(double z, double skew, double excess_kurtosis) {
  const double z2 = z * z;
  const double z3 = z2 * z;
  return z + (z2 - 1.0) * skew / 6.0 + (z3 - 3.0 * z) * excess_kurtosis / 24.0 -
         (2.0 * z3 - 5.0 * z) * skew * skew / 36.0;
}

bool cornish_fisher_domain_ok(double skew, double excess_kurtosis,
                              double z_range, int n_grid) {
  if (n_grid < 3) throw std::invalid_argument("n_grid must be >= 3");
  double prev = cornish_fisher_z(-z_range, skew, excess_kurtosis);
  for (int i = 1; i < n_grid; ++i) {
    const double z = -z_range + 2.0 * z_range * static_cast<double>(i) /
                                    static_cast<double>(n_grid - 1);
    const double cur = cornish_fisher_z(z, skew, excess_kurtosis);
    if (!(cur > prev)) return false;
    prev = cur;
  }
  return true;
}

double cornish_fisher_var(double sigma, double skew, double excess_kurtosis,
                          double alpha, double mean, double horizon_days,
                          bool check_domain) {
  validate_alpha(alpha);
  validate_horizon(horizon_days);
  if (sigma < 0.0) throw std::invalid_argument("sigma must be >= 0");
  if (check_domain && !cornish_fisher_domain_ok(skew, excess_kurtosis)) {
    std::ostringstream msg;
    msg << "Cornish-Fisher expansion is non-monotone for skew=" << skew
        << ", excess_kurtosis=" << excess_kurtosis
        << ": outside validity domain; fall back to historical or t VaR "
           "(set check_domain=false to force)";
    throw std::invalid_argument(msg.str());
  }
  // Loss quantile: lower tail of the P&L distribution.
  const double z = norm_ppf(1.0 - alpha);
  const double zcf = cornish_fisher_z(z, skew, excess_kurtosis);
  const double scale = std::sqrt(horizon_days);
  return -(mean * horizon_days) - sigma * scale * zcf;
}

}  // namespace fxvar
