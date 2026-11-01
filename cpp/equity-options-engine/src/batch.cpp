#include "eqopt/batch.hpp"

#include <cfloat>
#include <cmath>

namespace eqopt {

namespace {

constexpr double kMax = DBL_MAX;

/// Lengths of every input span equal `n`.
bool same_length(const BsBatchInputs& in, std::size_t n) noexcept {
    return in.S.size() == n && in.K.size() == n && in.T.size() == n && in.r.size() == n &&
           in.sigma.size() == n && in.q.size() == n && in.type.size() == n;
}

/// Validity flag for element i. Every field is loaded unconditionally and
/// the comparisons are combined with bitwise and on ints, so the loop has no
/// control flow and vectorises. `x >= 0 && x <= DBL_MAX` is false for NaN,
/// +/-Inf and negatives, the set try_check_nonneg rejects; `|x| <= DBL_MAX`
/// is std::isfinite.
inline int ok_price_inputs(const BsBatchInputs& in, std::size_t i) noexcept {
    const double S = in.S[i], K = in.K[i], T = in.T[i], v = in.sigma[i];
    const double r = in.r[i], q = in.q[i];
    return static_cast<int>(S >= 0.0) & static_cast<int>(S <= kMax) &
           static_cast<int>(K >= 0.0) & static_cast<int>(K <= kMax) &
           static_cast<int>(T >= 0.0) & static_cast<int>(T <= kMax) &
           static_cast<int>(v >= 0.0) & static_cast<int>(v <= kMax) &
           static_cast<int>(std::fabs(r) <= kMax) & static_cast<int>(std::fabs(q) <= kMax);
}

inline int ok_greeks_inputs(const BsBatchInputs& in, std::size_t i) noexcept {
    const double S = in.S[i], K = in.K[i], T = in.T[i], v = in.sigma[i];
    const double r = in.r[i], q = in.q[i];
    return static_cast<int>(S > 0.0) & static_cast<int>(S <= kMax) &
           static_cast<int>(K > 0.0) & static_cast<int>(K <= kMax) &
           static_cast<int>(T > 0.0) & static_cast<int>(T <= kMax) &
           static_cast<int>(v > 0.0) & static_cast<int>(v <= kMax) &
           static_cast<int>(std::fabs(r) <= kMax) & static_cast<int>(std::fabs(q) <= kMax);
}

/// Vectorisable pass: count invalid elements.
template <bool Greeks>
std::size_t count_bad(const BsBatchInputs& in, std::size_t n) noexcept {
    unsigned bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int ok = Greeks ? ok_greeks_inputs(in, i) : ok_price_inputs(in, i);
        bad |= static_cast<unsigned>(ok ^ 1);
    }
    return bad;
}

}  // namespace

std::expected<void, BatchError> bs_price_batch(const BsBatchInputs& in,
                                               std::span<double> out) noexcept {
    const std::size_t n = out.size();
    if (!same_length(in, n)) [[unlikely]] {
        return std::unexpected(BatchError{
            0, Error{Errc::SizeMismatch, "bs_price_batch", static_cast<double>(in.S.size()),
                     static_cast<double>(n)}});
    }
    if (count_bad<false>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r = try_bs_price(in.S[i], in.K[i], in.T[i], in.r[i], in.sigma[i],
                                        in.q[i], in.type[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bs_price_unchecked(in.S[i], in.K[i], in.T[i], in.r[i], in.sigma[i], in.q[i],
                                    in.type[i]);
    }
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < n; ++i) non_finite += std::fabs(out[i]) <= kMax ? 0U : 1U;
    if (non_finite != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(out[i])) {
                return std::unexpected(BatchError{i, Error{Errc::NonFiniteResult, "bs_price"}});
            }
        }
    }
    return {};
}

std::expected<void, BatchError> bs_greeks_batch(const BsBatchInputs& in,
                                                const BsGreeksBatchOutputs& out) noexcept {
    const std::size_t n = out.price.size();
    const bool sizes = same_length(in, n) && out.delta.size() == n && out.gamma.size() == n &&
                       out.vega.size() == n && out.theta.size() == n && out.rho.size() == n &&
                       out.vanna.size() == n && out.volga.size() == n;
    if (!sizes) [[unlikely]] {
        return std::unexpected(BatchError{
            0, Error{Errc::SizeMismatch, "bs_greeks_batch", static_cast<double>(in.S.size()),
                     static_cast<double>(n)}});
    }
    if (count_bad<true>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r = try_bs_greeks(in.S[i], in.K[i], in.T[i], in.r[i], in.sigma[i],
                                         in.q[i], in.type[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        const BSGreeks g = bs_greeks_unchecked(in.S[i], in.K[i], in.T[i], in.r[i],
                                               in.sigma[i], in.q[i], in.type[i]);
        out.price[i] = g.price;
        out.delta[i] = g.delta;
        out.gamma[i] = g.gamma;
        out.vega[i] = g.vega;
        out.theta[i] = g.theta;
        out.rho[i] = g.rho;
        out.vanna[i] = g.vanna;
        out.volga[i] = g.volga;
    }
    return {};
}

}  // namespace eqopt
