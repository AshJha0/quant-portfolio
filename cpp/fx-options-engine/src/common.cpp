#include "fxopt/common.hpp"

#include <cmath>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

namespace fxopt {

// Acklam's rational approximation to the inverse normal CDF (~1.15e-9
// relative), polished with two Halley iterations against the erfc-based
// norm_cdf to full double precision (~1e-15 absolute in z).
double norm_ppf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("norm_ppf requires p in (0, 1), got " +
                                    std::format("{}", p));
    }
    return norm_ppf_unchecked(p);
}

double norm_ppf_unchecked(double p) noexcept {
    static constexpr double a[6] = {
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    static constexpr double b[5] = {
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01,  -1.328068155288572e+01};
    static constexpr double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    static constexpr double d[4] = {
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00};
    constexpr double p_low = 0.02425;

    double z;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
             c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - p_low) {
        const double q = p - 0.5;
        const double r = q * q;
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
             a[5]) *
            q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r +
             1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
              c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    // Halley refinement: z_{n+1} = z - e/(f - e z / 2) with e = (N(z)-p)/n(z).
    for (int i = 0; i < 2; ++i) {
        const double e = norm_cdf(z) - p;
        const double u = e / norm_pdf(z);
        z -= u / (1.0 + 0.5 * z * u);
    }
    return z;
}

std::string message(const Error& e) {
    switch (e.code) {
        case Errc::NonFinite:
            return std::string(e.field) + " must be finite, got " + std::format("{}", e.a);
        case Errc::SpotNonPositive:
            return "Spot S must be positive, got " + std::format("{}", e.a);
        case Errc::StrikeNonPositive:
            return "Strike K must be positive, got " + std::format("{}", e.a);
        case Errc::ForwardNonPositive:
            return "Forward F must be positive, got " + std::format("{}", e.a);
        case Errc::NegativeExpiry:
            return "Time to expiry T must be non-negative, got " + std::format("{}", e.a);
        case Errc::NegativeVol:
            return "Volatility sigma must be non-negative, got " + std::format("{}", e.a);
        case Errc::NonFiniteResult:
            return std::string(e.field) +
                   ": result is not representable as a finite double (a discount factor or "
                   "forward overflows)";
        case Errc::D1Undefined:
            return "d1 undefined for sigma*sqrt(T)=" + std::format("{}", e.a) +
                   "; need sigma>0, T>0";
        case Errc::GreeksDomain:
            return "analytic_greeks requires T > 0 and sigma > 0";
        case Errc::BadSteps:
            return "steps must be a positive integer, got " +
                   std::format("{}", static_cast<int>(e.a));
        case Errc::ProbabilityOutOfRange:
            return std::format("risk-neutral probability {} outside [0, 1]; increase steps "
                               "(dt too large for |r_d - r_f| vs sigma)",
                               e.a);
        case Errc::WorkspaceTooSmall:
            return std::format("{}: workspace holds {} doubles, needs {}", e.field,
                               static_cast<std::int64_t>(e.a), static_cast<std::int64_t>(e.b));
        case Errc::IvExpired:
            return "implied_vol requires T > 0";
        case Errc::IvOutsideBounds:
            return std::format("price {} outside no-arbitrage bounds [{}, {}]", e.a, e.b, e.c);
        case Errc::IvNearUpperBound:
            return std::format(
                "price {} is within {} of the sigma->inf bound {}; no finite implied "
                "volatility is identifiable",
                e.a, e.b, e.c);
        case Errc::AtBound:
            return "price is within rounding of a no-arbitrage bound";
        case Errc::NoConvergence:
            return "implied vol: no convergence in 100 iterations";
        case Errc::McExpired:
            return "mc_price requires T > 0";
        case Errc::TooFewPaths:
            return std::format(
                "n_paths = {} gives {} independent sample(s); at least {} are needed "
                "for a standard error",
                static_cast<std::int64_t>(e.a), static_cast<std::int64_t>(e.b),
                static_cast<std::int64_t>(e.c));
        case Errc::McOverflow:
            return "simulated spots overflow a double; drift or vol too large";
        case Errc::BadThreads:
            return "threads must be >= 1";
        case Errc::SizeMismatch:
            return std::format("{}: span lengths differ ({} vs {})", e.field,
                               static_cast<std::int64_t>(e.a), static_cast<std::int64_t>(e.b));
    }
    return "unknown error";
}

void throw_error(const Error& e) {
    if (e.code == Errc::NoConvergence) throw std::runtime_error(message(e));
    throw std::invalid_argument(message(e));
}

}  // namespace fxopt
