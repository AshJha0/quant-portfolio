// eqvar/error.hpp: error values for the noexcept kernel API.
//
// Every try_* kernel is noexcept and returns std::expected<T, Error>. An
// Error is a code, the name of the calling function (a string literal) and
// a few numbers, so building one never allocates. message() formats the
// text only when asked, which is what the throwing wrappers do: they keep
// the exception types and messages of the original API.

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "qp/expected.hpp"
#include <type_traits>
#include <utility>

namespace eqvar {

/// What went wrong. Each code maps to one exception message.
enum class Errc : std::uint8_t {
    BadAlpha,             ///< alpha (a) outside (0, 0.5).
    EmptySample,          ///< who: empty sample.
    NonFiniteInput,       ///< who: input contains NaN or inf.
    TooFewPnl,            ///< need at least n1 P&L observations, got n2.
    PnlNotFinite,         ///< pnl contains NaN or infinite values.
    BadQuantile,          ///< quantile_linear: q (a) outside [0, 1].
    NoPositions,          ///< who: book has no positions.
    ExposuresNotFinite,   ///< who: exposures must be finite.
    MatvecShape,          ///< vector size n1 vs n2 columns.
    QuadFormShape,        ///< quad_form: shape vs vector size n1.
    SigmaShape,           ///< portfolio_sigma: shape vs n1 exposures.
    SigmaNotFinite,       ///< portfolio_sigma: w'Sw not finite.
    NotPsd,               ///< portfolio_sigma: w'Sw < 0.
    MeanNotFinite,        ///< who: mean must be finite.
    BadHorizon,           ///< parametric_var: horizon_days (n1 as signed) < 1.
    BadDf,                ///< parametric_var: df (a) <= 2 or not finite.
    CovNotSquare,         ///< who: covariance must be square and non-empty.
    CovNotFinite,         ///< who: covariance contains NaN or infinite entries.
    CovNegativeDiag,      ///< who: negative variance on the diagonal.
    CovAsymmetric,        ///< who: covariance matrix must be symmetric.
    CholZeroVar,          ///< cholesky: zero variances with non-zero covariances.
    CholIndefinite,       ///< cholesky: past the jitter cap (runtime_error).
    WorkspaceTooSmall,    ///< who: workspace holds n1, needs n2.
    McShape,              ///< monte_carlo_var: covariance shape does not match.
    McNoPaths,            ///< simulate_factor_returns: n_paths must be >= 1.
    McBadDf,              ///< simulate_factor_returns: df (a) <= 2.
    McReturnsNotFinite,   ///< portfolio_pnl: returns contain NaN or inf.
    McTooFewScenarios,    ///< mc_tail_metrics: need 100 scenarios, got n1.
    McPnlNotFinite,       ///< who: pnl contains NaN or inf.
    BootTooFewObs,        ///< mc_bootstrap_se: need 10 observations, got n1.
    BootTooFewReps,       ///< mc_bootstrap_se: n_boot must be >= 2.
};

/// A kernel error. Trivially copyable, never allocates.
struct Error {
    Errc code;
    const char* who = "";  ///< static string, never owned
    double a = 0.0;
    std::size_t n1 = 0, n2 = 0;
};

/// Result of a noexcept kernel.
template <class T>
using Result = qp::Result<T, Error>;

/// Human-readable message, identical to the exception text of the throwing
/// API. Allocates; call it on the error path only.
std::string message(const Error& e);

/// Throw what the original API threw for `e`: std::runtime_error for
/// CholZeroVar and CholIndefinite, std::invalid_argument otherwise.
[[noreturn]] void throw_error(const Error& e);

/// Unwrap a result or throw (qp/expected.hpp). Used by the throwing wrappers.
using qp::value_or_throw;

}  // namespace eqvar
