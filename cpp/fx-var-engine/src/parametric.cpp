#include "fxvar/parametric.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/stats.hpp"

namespace fxvar {

double portfolio_sigma(const std::vector<double>& exposures, const Matrix& cov) {
  for (double w : exposures)
    if (!std::isfinite(w))
      throw std::invalid_argument("portfolio_sigma: exposures must be finite");
  const double var = quad_form(exposures, cov);
  // NaN loses every ordered comparison, so an unguarded non-finite input
  // would return a NaN sigma and quietly poison the whole VaR report.
  if (!std::isfinite(var))
    throw std::invalid_argument(
        "portfolio_sigma: w'Sw is not finite (covariance contains NaN/Inf or "
        "the exposures overflow)");
  // The PSD test must be RELATIVE to the size of the terms being summed.
  // A hedged multi-billion book against a rank-deficient (single-driver)
  // covariance has true variance 0 but a quadratic form that rounds to
  // ~1e-16 of |w|^2 |Sigma| - which an absolute -1e-12 threshold flags as
  // "not positive semi-definite" on perfectly good market data.
  double wmax = 0.0;
  for (double w : exposures) wmax = std::max(wmax, std::abs(w));
  double cmax = 0.0;
  for (std::size_t i = 0; i < cov.rows(); ++i)
    for (std::size_t j = 0; j < cov.cols(); ++j)
      cmax = std::max(cmax, std::abs(cov(i, j)));
  const double tol = 1e-10 * std::max(1.0, wmax * wmax * cmax);
  if (var < -tol)
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
  // n_grid is retained for API compatibility; the check below is
  // closed-form and no longer samples a grid, but n_grid keeps its
  // validation so a caller-supplied resolution is still sanity-checked.
  if (n_grid < 3) throw std::invalid_argument("n_grid must be >= 3");
  if (!(z_range > 0.0) || !std::isfinite(z_range)) {
    throw std::invalid_argument("z_range must be finite and > 0");
  }
  // dz_cf/dz = 1 + zS/3 + (3z^2-3)K/24 - (6z^2-5)S^2/36 must stay > 0 on
  // [-z_range, z_range]. It is a quadratic in z, g(z) = A z^2 + B z + C
  // with A = K/8 - S^2/6, B = S/3, C = 1 - K/8 + 5 S^2/36, whose exact
  // minimum on the interval is known in closed form: the vertex -B/(2A)
  // when g is convex (A > 0) and that point lies in range, else an
  // interval endpoint (a concave/linear g, A <= 0, always attains its
  // minimum at an endpoint). This replaces a finite-difference scan over
  // z_cf values on a fixed grid, which can miss a thin sub-grid dip below
  // zero: skew=0.122, excess_kurtosis=-0.427 is non-monotone on |z| <= 4
  // (minimum derivative ~ -9e-4 near z ~ 3.1) but the 801-point grid
  // previously used here reported it as monotone.
  if (!std::isfinite(skew) || !std::isfinite(excess_kurtosis)) return false;
  const double a = excess_kurtosis / 8.0 - skew * skew / 6.0;
  const double b = skew / 3.0;
  const double c = 1.0 - excess_kurtosis / 8.0 + 5.0 * skew * skew / 36.0;
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
