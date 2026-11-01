#include "fxopt/common.hpp"

#include <cmath>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

namespace fxopt {

// Acklam's rational approximation polished with two Halley steps
// (qp::norm_ppf_unchecked<2>), about 1e-15 absolute in z.
double norm_ppf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("norm_ppf requires p in (0, 1), got " +
                                    std::format("{}", p));
    }
    return norm_ppf_unchecked(p);
}

double norm_ppf_unchecked(double p) noexcept { return qp::norm_ppf_unchecked<2>(p); }

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
