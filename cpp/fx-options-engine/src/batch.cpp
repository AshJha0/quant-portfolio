#include "fxopt/batch.hpp"

#include <cfloat>
#include <cmath>

namespace fxopt {

namespace {

constexpr double kMax = DBL_MAX;

bool same_length(const GkBatchInputs& in, std::size_t n) noexcept {
    return in.S.size() == n && in.K.size() == n && in.T.size() == n && in.r_d.size() == n &&
           in.r_f.size() == n && in.sigma.size() == n && in.type.size() == n;
}

/// Validity flag for element i. Every field is loaded unconditionally and
/// the comparisons are combined with bitwise and on ints, so the loop has no
/// control flow and vectorises. `|x| <= DBL_MAX` is std::isfinite.
inline int ok_price_inputs(const GkBatchInputs& in, std::size_t i) noexcept {
    const double S = in.S[i], K = in.K[i], T = in.T[i], v = in.sigma[i];
    const double rd = in.r_d[i], rf = in.r_f[i];
    return static_cast<int>(S > 0.0) & static_cast<int>(S <= kMax) & static_cast<int>(K > 0.0) &
           static_cast<int>(K <= kMax) & static_cast<int>(T >= 0.0) & static_cast<int>(T <= kMax) &
           static_cast<int>(v >= 0.0) & static_cast<int>(v <= kMax) &
           static_cast<int>(std::fabs(rd) <= kMax) & static_cast<int>(std::fabs(rf) <= kMax);
}

/// Greeks domain: also T > 0, sigma > 0 and sigma*sqrt(T) > kMinVol,
/// computed exactly as try_d1() does (sqrt is correctly rounded).
inline int ok_greeks_inputs(const GkBatchInputs& in, std::size_t i) noexcept {
    const double T = in.T[i], v = in.sigma[i];
    return ok_price_inputs(in, i) & static_cast<int>(T > 0.0) & static_cast<int>(v > 0.0) &
           static_cast<int>(v * std::sqrt(T) > kMinVol);
}

/// Vectorisable pass: nonzero if any element is invalid.
template <bool Greeks>
unsigned any_bad(const GkBatchInputs& in, std::size_t n) noexcept {
    unsigned bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int ok = Greeks ? ok_greeks_inputs(in, i) : ok_price_inputs(in, i);
        bad |= static_cast<unsigned>(ok ^ 1);
    }
    return bad;
}

}  // namespace

std::expected<void, BatchError> gk_price_batch(const GkBatchInputs& in,
                                               std::span<double> out) noexcept {
    const std::size_t n = out.size();
    if (!same_length(in, n)) [[unlikely]] {
        return std::unexpected(
            BatchError{0, Error{Errc::SizeMismatch, "gk_price_batch",
                                static_cast<double>(in.S.size()), static_cast<double>(n)}});
    }
    if (any_bad<false>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r =
                try_validate_inputs(in.S[i], in.K[i], in.T[i], in.r_d[i], in.r_f[i], in.sigma[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = gk_price_unchecked(in.S[i], in.K[i], in.T[i], in.r_d[i], in.r_f[i], in.sigma[i],
                                    in.type[i]);
    }
    unsigned non_finite = 0;
    for (std::size_t i = 0; i < n; ++i) {
        non_finite |= static_cast<unsigned>(!(std::fabs(out[i]) <= kMax));
    }
    if (non_finite != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(out[i])) {
                return std::unexpected(BatchError{i, Error{Errc::NonFiniteResult, "gk_price"}});
            }
        }
    }
    return {};
}

std::expected<void, BatchError> gk_greeks_batch(const GkBatchInputs& in,
                                                const GkGreeksBatchOutputs& out) noexcept {
    const std::size_t n = out.price.size();
    const bool sizes = same_length(in, n) && out.delta_spot.size() == n &&
                       out.delta_forward.size() == n && out.gamma.size() == n &&
                       out.vega.size() == n && out.theta.size() == n &&
                       out.rho_domestic.size() == n && out.rho_foreign.size() == n &&
                       out.vanna.size() == n && out.volga.size() == n;
    if (!sizes) [[unlikely]] {
        return std::unexpected(
            BatchError{0, Error{Errc::SizeMismatch, "gk_greeks_batch",
                                static_cast<double>(in.S.size()), static_cast<double>(n)}});
    }
    if (any_bad<true>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r = try_analytic_greeks(in.S[i], in.K[i], in.T[i], in.r_d[i], in.r_f[i],
                                               in.sigma[i], in.type[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        const GreeksResult g = analytic_greeks_unchecked(in.S[i], in.K[i], in.T[i], in.r_d[i],
                                                         in.r_f[i], in.sigma[i], in.type[i]);
        out.price[i] = g.price;
        out.delta_spot[i] = g.delta_spot;
        out.delta_forward[i] = g.delta_forward;
        out.gamma[i] = g.gamma;
        out.vega[i] = g.vega;
        out.theta[i] = g.theta;
        out.rho_domestic[i] = g.rho_domestic;
        out.rho_foreign[i] = g.rho_foreign;
        out.vanna[i] = g.vanna;
        out.volga[i] = g.volga;
    }
    return {};
}

}  // namespace fxopt
