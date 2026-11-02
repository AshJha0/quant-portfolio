// qp/constants.hpp: numeric constants shared by the engines.
//
// Each constant is the double nearest the exact value. The engines used to
// spell these with different numbers of digits; every spelling rounds to the
// same double, so sharing them changes no result.

#pragma once

#include <limits>
#include <numbers>

namespace qp {

inline constexpr double kSqrt2 = std::numbers::sqrt2;                         // 1.4142135623730951
inline constexpr double kInvSqrt2 = 0.7071067811865475244008443621048490393;  // 1/sqrt(2)
inline constexpr double kInvSqrt2Pi = 0.3989422804014326779399460599343819;   // 1/sqrt(2 pi)
inline constexpr double kSqrt2Pi = 2.5066282746310002;                        // sqrt(2 pi)
inline constexpr double kLnSqrt2Pi = 0.9189385332046728;                      // ln sqrt(2 pi)
inline constexpr double kPi = std::numbers::pi;
inline constexpr double kEps = std::numeric_limits<double>::epsilon();

/// 2^-53: the spacing of a 53-bit uniform on [0, 1).
inline constexpr double kTwoPowMinus53 = 0x1.0p-53;

}  // namespace qp
