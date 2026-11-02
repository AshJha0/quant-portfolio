#include "eqopt/batch.hpp"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "qp/vmath.hpp"

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
    return static_cast<int>(S >= 0.0) & static_cast<int>(S <= kMax) & static_cast<int>(K >= 0.0) &
           static_cast<int>(K <= kMax) & static_cast<int>(T >= 0.0) & static_cast<int>(T <= kMax) &
           static_cast<int>(v >= 0.0) & static_cast<int>(v <= kMax) &
           static_cast<int>(std::fabs(r) <= kMax) & static_cast<int>(std::fabs(q) <= kMax);
}

inline int ok_greeks_inputs(const BsBatchInputs& in, std::size_t i) noexcept {
    const double S = in.S[i], K = in.K[i], T = in.T[i], v = in.sigma[i];
    const double r = in.r[i], q = in.q[i];
    return static_cast<int>(S > 0.0) & static_cast<int>(S <= kMax) & static_cast<int>(K > 0.0) &
           static_cast<int>(K <= kMax) & static_cast<int>(T > 0.0) & static_cast<int>(T <= kMax) &
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

/// `mask ? a : b` for an all-ones or all-zeros mask, as bit operations so
/// that GCC does not turn it into a branch (which blocks vectorisation).
inline double blend(std::uint64_t mask, double a, double b) noexcept {
    return std::bit_cast<double>((std::bit_cast<std::uint64_t>(a) & mask) |
                                 (std::bit_cast<std::uint64_t>(b) & ~mask));
}

/// Vectorised pass over one block: writes the fast price for every element
/// and slow[i] = 1 where the scalar formula is needed. Returns non-zero if
/// any slow flag is set. On x86-64 Linux with GCC or Clang the function is
/// also compiled for AVX2 and the loader picks that clone on CPUs that have
/// it (target_clones / ifunc). AVX2 alone does not enable FMA, so both
/// clones do the same IEEE operations and return the same bits.
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target_clones("avx2", "default")))
#endif
unsigned fast_block(const double* S, const double* K, const double* T, const double* r,
                    const double* v, const double* q, const OptionType* ty, double* o,
                    std::uint8_t* slow, std::size_t m) noexcept {
constexpr double kRsqrt2 = 0x1.6a09e667f3bcdp-1;
unsigned any_slow = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const double Si = S[i], Ki = K[i], Ti = T[i], ri = r[i], vi = v[i], qi = q[i];
        const double phi = 1.0 - 2.0 * static_cast<double>(ty[i] != OptionType::Call);
        const double ratio = Si / Ki;
        // Strictly positive inputs and a normal, finite S / K.
        const int ok = static_cast<int>(Ti > 0.0) & static_cast<int>(vi > 0.0) &
                       static_cast<int>(ratio >= 0x1p-1022) & static_cast<int>(ratio <= kMax);
        const std::uint64_t mask = std::uint64_t{0} - static_cast<std::uint64_t>(ok);
        const double Ts = blend(mask, Ti, 1.0);
        const double vs = blend(mask, vi, 1.0);
        const double xs = blend(mask, ratio, 1.0);
        const double sqrt_t = std::sqrt(Ts);
        const double vst = vs * sqrt_t;
        const double d1 = (qp::vm::detail::log1(xs) + (ri - qi + 0.5 * vs * vs) * Ts) / vst;
        const double d2 = d1 - vst;
        const double disc_s = Si * qp::vm::detail::exp1(-qi * Ts);
        const double disc_k = Ki * qp::vm::detail::exp1(-ri * Ts);
        const double n1 = 0.5 * qp::vm::detail::erfc1(-phi * d1 * kRsqrt2);
        const double n2 = 0.5 * qp::vm::detail::erfc1(-phi * d2 * kRsqrt2);
        o[i] = phi * (disc_s * n1 - disc_k * n2);
        slow[i] = static_cast<std::uint8_t>(ok ^ 1);
        any_slow |= static_cast<unsigned>(ok ^ 1);
    }
    return any_slow;
}

/// Full validation pass; on failure, the first bad index and the error the
/// scalar try_bs_price reports for it.
std::expected<void, BatchError> check_price_inputs(const BsBatchInputs& in,
                                                   std::size_t n) noexcept {
    if (count_bad<false>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r =
                try_bs_price(in.S[i], in.K[i], in.T[i], in.r[i], in.sigma[i], in.q[i], in.type[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    return {};
}

/// First non-finite output, if any, as a NonFiniteResult error.
std::expected<void, BatchError> check_finite_outputs(std::span<const double> out) noexcept {
    std::size_t non_finite = 0;
    for (double x : out) non_finite += std::fabs(x) <= kMax ? 0U : 1U;
    if (non_finite != 0) [[unlikely]] {
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (!std::isfinite(out[i])) {
                return std::unexpected(BatchError{i, Error{Errc::NonFiniteResult, "bs_price"}});
            }
        }
    }
    return {};
}

}  // namespace

std::expected<void, BatchError> bs_price_batch(const BsBatchInputs& in,
                                               std::span<double> out) noexcept {
    const std::size_t n = out.size();
    if (!same_length(in, n)) [[unlikely]] {
        return std::unexpected(
            BatchError{0, Error{Errc::SizeMismatch, "bs_price_batch",
                                static_cast<double>(in.S.size()), static_cast<double>(n)}});
    }
    if (auto e = check_price_inputs(in, n); !e) [[unlikely]]
        return e;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bs_price_unchecked(in.S[i], in.K[i], in.T[i], in.r[i], in.sigma[i], in.q[i],
                                    in.type[i]);
    }
    return check_finite_outputs(out);
}

std::expected<void, BatchError> bs_price_batch_fast(const BsBatchInputs& in,
                                                    std::span<double> out) noexcept {
    const std::size_t n = out.size();
    if (!same_length(in, n)) [[unlikely]] {
        return std::unexpected(
            BatchError{0, Error{Errc::SizeMismatch, "bs_price_batch_fast",
                                static_cast<double>(in.S.size()), static_cast<double>(n)}});
    }
    if (auto e = check_price_inputs(in, n); !e) [[unlikely]]
        return e;
    // Blocks of kBlock: one branch-free pass that the compiler vectorises,
    // then a scalar pass over the few elements the fast formula does not
    // cover (T, K, S or sigma zero, or S / K outside the normal range).
    constexpr std::size_t kBlock = 256;
    std::uint8_t slow[kBlock];
    for (std::size_t b = 0; b < n; b += kBlock) {
        const std::size_t m = std::min(kBlock, n - b);
        const double* S = in.S.data() + b;
        const double* K = in.K.data() + b;
        const double* T = in.T.data() + b;
        const double* r = in.r.data() + b;
        const double* v = in.sigma.data() + b;
        const double* q = in.q.data() + b;
        const OptionType* ty = in.type.data() + b;
        double* o = out.data() + b;
        const unsigned any_slow = fast_block(S, K, T, r, v, q, ty, o, slow, m);
        if (any_slow != 0) [[unlikely]] {
            for (std::size_t i = 0; i < m; ++i) {
                if (slow[i] != 0) o[i] = bs_price_unchecked(S[i], K[i], T[i], r[i], v[i], q[i], ty[i]);
            }
        }
    }
    return check_finite_outputs(out);
}

std::expected<void, BatchError> bs_greeks_batch(const BsBatchInputs& in,
                                                const BsGreeksBatchOutputs& out) noexcept {
    const std::size_t n = out.price.size();
    const bool sizes = same_length(in, n) && out.delta.size() == n && out.gamma.size() == n &&
                       out.vega.size() == n && out.theta.size() == n && out.rho.size() == n &&
                       out.vanna.size() == n && out.volga.size() == n;
    if (!sizes) [[unlikely]] {
        return std::unexpected(
            BatchError{0, Error{Errc::SizeMismatch, "bs_greeks_batch",
                                static_cast<double>(in.S.size()), static_cast<double>(n)}});
    }
    if (count_bad<true>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r =
                try_bs_greeks(in.S[i], in.K[i], in.T[i], in.r[i], in.sigma[i], in.q[i], in.type[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        const BSGreeks g = bs_greeks_unchecked(in.S[i], in.K[i], in.T[i], in.r[i], in.sigma[i],
                                               in.q[i], in.type[i]);
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
