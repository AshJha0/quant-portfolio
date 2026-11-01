#include "eqopt/black_scholes.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace eqopt {

std::string message(const Error& e) {
    std::ostringstream oss;
    switch (e.code) {
        case Errc::NegativeOrNonFinite:
            oss << e.field << " must be finite and >= 0, got " << e.a;
            break;
        case Errc::NonFiniteRate:
            oss << "r and q must be finite (NaN/Inf rejected), got r=" << e.a
                << ", q=" << e.b;
            break;
        case Errc::NotStrictlyPositive:
            oss << "d1/d2 require strictly positive S, K, T and sigma; got S=" << e.a
                << ", K=" << e.b << ", T=" << e.c << ", sigma=" << e.d;
            break;
        case Errc::B76NotStrictlyPositive:
            oss << "b76_d1_d2 requires strictly positive F, K, T, sigma; got F=" << e.a
                << ", K=" << e.b << ", T=" << e.c << ", sigma=" << e.d;
            break;
        case Errc::NonFiniteResult:
            oss << e.field
                << ": result is not representable as a finite double (a discount "
                   "factor or forward overflows)";
            break;
        case Errc::BadSteps:
            oss << "n_steps must be >= 1, got " << static_cast<long long>(e.a);
            break;
        case Errc::ProbabilityOutOfRange:
            oss << "risk-neutral probability p=" << e.a
                << " outside (0, 1); increase n_steps or check r, q, sigma";
            break;
        case Errc::WorkspaceTooSmall:
            oss << e.field << ": workspace holds " << static_cast<long long>(e.a)
                << " doubles, needs " << static_cast<long long>(e.b);
            break;
        case Errc::PriceNotFinite:
            oss << "price must be finite";
            break;
        case Errc::Expired:
            oss << "implied_vol requires T > 0 (option already expired)";
            break;
        case Errc::NonPositiveSpotStrike:
            oss << "implied_vol requires S > 0 and K > 0";
            break;
        case Errc::BelowLowerBound:
            oss << "price " << e.a << " is at or below the sigma->0 arbitrage bound " << e.b
                << "; implied vol is undefined";
            break;
        case Errc::AboveUpperBound:
            oss << "price " << e.a << " is at or above the sigma->inf bound " << e.b
                << "; implied vol is undefined";
            break;
        case Errc::AtBound:
            oss << "price is within rounding of a no-arbitrage bound";
            break;
        case Errc::NoConvergence:
            oss << "implied vol: no convergence in 100 iterations";
            break;
        case Errc::TooFewPaths:
            oss << "n_paths gives too few independent samples for a standard error "
                   "(need 2, or 3 with the control variate; an antithetic pair is one)";
            break;
        case Errc::BadThreads:
            oss << "threads must be >= 1";
            break;
        case Errc::McNotFinite:
            oss << "Monte Carlo estimate is not finite (simulated prices overflow)";
            break;
        case Errc::SizeMismatch:
            oss << e.field << ": span lengths differ (" << static_cast<long long>(e.a)
                << " vs " << static_cast<long long>(e.b) << ")";
            break;
    }
    return oss.str();
}

void throw_error(const Error& e) {
    if (e.code == Errc::NoConvergence) throw std::runtime_error(message(e));
    throw std::invalid_argument(message(e));
}

void validate_inputs(double S, double K, double T, double sigma) {
    if (auto e = try_validate_inputs(S, K, T, sigma); !e) throw_error(e.error());
}

void validate_rates(double r, double q) {
    if (auto e = try_validate_rates(r, q); !e) throw_error(e.error());
}

std::pair<double, double> d1_d2(double S, double K, double T, double r,
                                double sigma, double q) {
    return value_or_throw(try_d1_d2(S, K, T, r, sigma, q));
}

double bs_price(double S, double K, double T, double r, double sigma,
                double q, OptionType type) {
    return value_or_throw(try_bs_price(S, K, T, r, sigma, q, type));
}

double require_finite_result(double value, const char* what) {
    return value_or_throw(try_finite(value, what));
}

}  // namespace eqopt
