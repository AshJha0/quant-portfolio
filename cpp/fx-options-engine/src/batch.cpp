#include "fxopt/batch.hpp"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "qp/vmath.hpp"

namespace fxopt {

#if defined(_MSC_VER)
#define QP_RESTRICT __restrict
#else
#define QP_RESTRICT __restrict__
#endif

// Kernel attributes: an AVX2 clone on x86-64 Linux (GCC, Clang); on MSVC the
// kernels are unused (the _fast entry points take the exact path there).
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
#define QP_KERNEL_ATTR __attribute__((target_clones("avx2", "default")))
#elif defined(_MSC_VER) && !defined(__clang__)
#define QP_KERNEL_ATTR [[maybe_unused]]
#else
#define QP_KERNEL_ATTR
#endif

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

/// `mask ? a : b` for an all-ones or all-zeros mask, as bit operations so
/// that GCC does not turn it into a branch (which blocks vectorisation).
[[maybe_unused]] inline double blend(std::uint64_t mask, double a, double b) noexcept {
    return std::bit_cast<double>((std::bit_cast<std::uint64_t>(a) & mask) |
                                 (std::bit_cast<std::uint64_t>(b) & ~mask));
}

/// Vectorised pass over one block: the fast price for every element and
/// slow[i] = 1 where the scalar formula is needed; returns non-zero if any
/// slow flag is set. On x86-64 Linux with GCC or Clang it is also compiled
/// for AVX2 and the loader picks that clone where the CPU has it. AVX2 alone
/// does not enable FMA, so both clones return the same bits.
QP_KERNEL_ATTR
unsigned fast_block(const double* S, const double* K, const double* T, const double* rd,
                    const double* rf, const double* v, const OptionType* ty, double* o,
                    std::uint8_t* slow, std::size_t m) noexcept {
    constexpr double kRsqrt2 = 0x1.6a09e667f3bcdp-1;
    unsigned any_slow = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const double Si = S[i], Ki = K[i], Ti = T[i], rdi = rd[i], rfi = rf[i], vi = v[i];
        const double p = 1.0 - 2.0 * static_cast<double>(ty[i] != OptionType::Call);
        const double ratio = Si / Ki;
        const double vst_raw = vi * std::sqrt(Ti);
        const int ok = static_cast<int>(vst_raw > kMinVol) & static_cast<int>(ratio >= 0x1p-1022) &
                       static_cast<int>(ratio <= kMax);
        const std::uint64_t mask = std::uint64_t{0} - static_cast<std::uint64_t>(ok);
        const double Ts = blend(mask, Ti, 1.0);
        const double vs = blend(mask, vi, 1.0);
        const double xs = blend(mask, ratio, 1.0);
        const double vst = vs * std::sqrt(Ts);
        const double d1 = (qp::vm::detail::log1(xs) + (rdi - rfi + 0.5 * vs * vs) * Ts) / vst;
        const double d2 = d1 - vst;
        const double disc_s = Si * qp::vm::detail::exp1(-rfi * Ts);
        const double disc_k = Ki * qp::vm::detail::exp1(-rdi * Ts);
        const double n1 = 0.5 * qp::vm::detail::erfc1(-p * d1 * kRsqrt2);
        const double n2 = 0.5 * qp::vm::detail::erfc1(-p * d2 * kRsqrt2);
        o[i] = p * (disc_s * n1 - disc_k * n2);
        slow[i] = static_cast<std::uint8_t>(ok ^ 1);
        any_slow |= static_cast<unsigned>(ok ^ 1);
    }
    return any_slow;
}

/// Vectorised Greeks over one block. Validation has already required S, K,
/// T, sigma > 0 and sigma sqrt(T) > kMinVol; slow[i] = 1 only where S / K
/// is outside the normal range.
QP_KERNEL_ATTR
// Every pointer is restrict-qualified: with 7 inputs and up to 11
// outputs GCC would need more run-time alias checks than it allows
// (--param vect-max-version-for-alias-checks=10) and leaves the loop scalar.
unsigned fast_greeks_block(const double* QP_RESTRICT S, const double* QP_RESTRICT K,
                           const double* QP_RESTRICT T, const double* QP_RESTRICT rd,
                           const double* QP_RESTRICT rf, const double* QP_RESTRICT v,
                           const OptionType* ty, double* QP_RESTRICT o0, double* QP_RESTRICT o1,
                           double* QP_RESTRICT o2, double* QP_RESTRICT o3, double* QP_RESTRICT o4,
                           double* QP_RESTRICT o5, double* QP_RESTRICT o6, double* QP_RESTRICT o7,
                           double* QP_RESTRICT o8, double* QP_RESTRICT o9, std::uint8_t* slow,
                           std::size_t m) noexcept {
    constexpr double kRsqrt2 = 0x1.6a09e667f3bcdp-1;
    constexpr double kInvSqrt2Pi = 0x1.9884533d43651p-2;
    unsigned any_slow = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const double Si = S[i], Ki = K[i], Ti = T[i], rdi = rd[i], rfi = rf[i], vi = v[i];
        const double p = 1.0 - 2.0 * static_cast<double>(ty[i] != OptionType::Call);
        const double ratio = Si / Ki;
        const int ok = static_cast<int>(ratio >= 0x1p-1022) & static_cast<int>(ratio <= kMax);
        const std::uint64_t mask = std::uint64_t{0} - static_cast<std::uint64_t>(ok);
        const double xs = blend(mask, ratio, 1.0);
        const double sqrt_t = std::sqrt(Ti);
        const double vst = vi * sqrt_t;
        const double d1 = (qp::vm::detail::log1(xs) + (rdi - rfi + 0.5 * vi * vi) * Ti) / vst;
        const double d2 = d1 - vst;
        const double df_f = qp::vm::detail::exp1(-rfi * Ti);
        const double df_d = qp::vm::detail::exp1(-rdi * Ti);
        const double pdf = kInvSqrt2Pi * qp::vm::detail::exp1(-0.5 * d1 * d1);
        const double n1 = 0.5 * qp::vm::detail::erfc1(-p * d1 * kRsqrt2);
        const double n2 = 0.5 * qp::vm::detail::erfc1(-p * d2 * kRsqrt2);
        const double vega = Si * df_f * pdf * sqrt_t;
        o0[i] = p * (Si * df_f * n1 - Ki * df_d * n2);
        o1[i] = p * df_f * n1;
        o2[i] = p * n1;
        o3[i] = df_f * pdf / (Si * vst);
        o4[i] = vega;
        o5[i] = -Si * df_f * pdf * vi / (2.0 * sqrt_t) +
                p * (rfi * Si * df_f * n1 - rdi * Ki * df_d * n2);
        o6[i] = p * Ki * Ti * df_d * n2;
        o7[i] = -p * Si * Ti * df_f * n1;
        o8[i] = -df_f * pdf * d2 / vi;
        o9[i] = vega * d1 * d2 / vi;
        slow[i] = static_cast<std::uint8_t>(ok ^ 1);
        any_slow |= static_cast<unsigned>(ok ^ 1);
    }
    return any_slow;
}

/// Full validation pass; on failure, the first bad index and its scalar error.
std::expected<void, BatchError> check_price_inputs(const GkBatchInputs& in,
                                                   std::size_t n) noexcept {
    if (any_bad<false>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r =
                try_validate_inputs(in.S[i], in.K[i], in.T[i], in.r_d[i], in.r_f[i], in.sigma[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    return {};
}

/// First non-finite output, if any, as a NonFiniteResult error.
std::expected<void, BatchError> check_finite_outputs(std::span<const double> out) noexcept {
    unsigned non_finite = 0;
    for (double x : out) non_finite |= static_cast<unsigned>(!(std::fabs(x) <= kMax));
    if (non_finite != 0) [[unlikely]] {
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (!std::isfinite(out[i])) {
                return std::unexpected(BatchError{i, Error{Errc::NonFiniteResult, "gk_price"}});
            }
        }
    }
    return {};
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
    if (auto e = check_price_inputs(in, n); !e) [[unlikely]]
        return e;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = gk_price_unchecked(in.S[i], in.K[i], in.T[i], in.r_d[i], in.r_f[i], in.sigma[i],
                                    in.type[i]);
    }
    return check_finite_outputs(out);
}

std::expected<void, BatchError> gk_price_batch_fast(const GkBatchInputs& in,
                                                    std::span<double> out) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    // MSVC does not vectorise the kernel and the scalar vmath code is about
    // 3-4x slower than the library calls (measured), so use the exact path.
    return gk_price_batch(in, out);
#else
    const std::size_t n = out.size();
    if (!same_length(in, n)) [[unlikely]] {
        return std::unexpected(
            BatchError{0, Error{Errc::SizeMismatch, "gk_price_batch_fast",
                                static_cast<double>(in.S.size()), static_cast<double>(n)}});
    }
    if (auto e = check_price_inputs(in, n); !e) [[unlikely]]
        return e;
    // Blocks: a vectorised pass, then the scalar formula for the few elements
    // it does not cover (T = 0, sigma sqrt(T) <= kMinVol, S / K not normal).
    constexpr std::size_t kBlock = 256;
    std::uint8_t slow[kBlock];
    for (std::size_t b = 0; b < n; b += kBlock) {
        const std::size_t m = std::min(kBlock, n - b);
        const double* S = in.S.data() + b;
        const double* K = in.K.data() + b;
        const double* T = in.T.data() + b;
        const double* rd = in.r_d.data() + b;
        const double* rf = in.r_f.data() + b;
        const double* v = in.sigma.data() + b;
        const OptionType* ty = in.type.data() + b;
        double* o = out.data() + b;
        if (fast_block(S, K, T, rd, rf, v, ty, o, slow, m) != 0) [[unlikely]] {
            for (std::size_t i = 0; i < m; ++i) {
                if (slow[i] != 0)
                    o[i] = gk_price_unchecked(S[i], K[i], T[i], rd[i], rf[i], v[i], ty[i]);
            }
        }
    }
    return check_finite_outputs(out);
#endif
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

std::expected<void, BatchError> gk_greeks_batch_fast(const GkBatchInputs& in,
                                                     const GkGreeksBatchOutputs& out) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    // MSVC does not vectorise the kernel and the scalar vmath code is about
    // 3-4x slower than the library calls (measured), so use the exact path.
    return gk_greeks_batch(in, out);
#else
    const std::size_t n = out.price.size();
    const bool sizes = same_length(in, n) && out.delta_spot.size() == n &&
                       out.delta_forward.size() == n && out.gamma.size() == n &&
                       out.vega.size() == n && out.theta.size() == n &&
                       out.rho_domestic.size() == n && out.rho_foreign.size() == n &&
                       out.vanna.size() == n && out.volga.size() == n;
    if (!sizes) [[unlikely]] {
        return std::unexpected(
            BatchError{0, Error{Errc::SizeMismatch, "gk_greeks_batch_fast",
                                static_cast<double>(in.S.size()), static_cast<double>(n)}});
    }
    if (any_bad<true>(in, n) != 0) [[unlikely]] {
        for (std::size_t i = 0; i < n; ++i) {
            const auto r = try_analytic_greeks(in.S[i], in.K[i], in.T[i], in.r_d[i], in.r_f[i],
                                               in.sigma[i], in.type[i]);
            if (!r) return std::unexpected(BatchError{i, r.error()});
        }
    }
    constexpr std::size_t kBlock = 256;
    std::uint8_t slow[kBlock];
    for (std::size_t b = 0; b < n; b += kBlock) {
        const std::size_t m = std::min(kBlock, n - b);
        double* const o[10] = {out.price.data() + b,         out.delta_spot.data() + b,
                               out.delta_forward.data() + b, out.gamma.data() + b,
                               out.vega.data() + b,          out.theta.data() + b,
                               out.rho_domestic.data() + b,  out.rho_foreign.data() + b,
                               out.vanna.data() + b,         out.volga.data() + b};
        const double* S = in.S.data() + b;
        const double* K = in.K.data() + b;
        const double* T = in.T.data() + b;
        const double* rd = in.r_d.data() + b;
        const double* rf = in.r_f.data() + b;
        const double* v = in.sigma.data() + b;
        const OptionType* ty = in.type.data() + b;
        if (fast_greeks_block(S, K, T, rd, rf, v, ty, o[0], o[1], o[2], o[3], o[4], o[5], o[6],
                              o[7], o[8], o[9], slow, m) != 0) [[unlikely]] {
            for (std::size_t i = 0; i < m; ++i) {
                if (slow[i] == 0) continue;
                const GreeksResult g =
                    analytic_greeks_unchecked(S[i], K[i], T[i], rd[i], rf[i], v[i], ty[i]);
                const double vals[10] = {g.price, g.delta_spot, g.delta_forward, g.gamma,
                                         g.vega,  g.theta,      g.rho_domestic,  g.rho_foreign,
                                         g.vanna, g.volga};
                for (int f = 0; f < 10; ++f) o[f][i] = vals[f];
            }
        }
    }
    return {};
#endif
}

}  // namespace fxopt
