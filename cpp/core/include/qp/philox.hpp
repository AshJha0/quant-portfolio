// qp/philox.hpp: Philox4x32-10 counter-based random numbers.
//
// Reference: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw,
// "Parallel random numbers: as easy as 1, 2, 3", SC '11, 2011, section 3.2
// and Table 2. The constants and the round function are those of the
// Random123 library (philox.h); the known-answer vectors in the core tests
// are Random123's kat_vectors for philox4x32_10.
//
// A counter-based generator is a pure function of (key, counter). A Monte
// Carlo path p can take key = seed and counter = (p, block, 0, 0), so any
// thread can produce any path's numbers without sharing state, and the
// result does not depend on the number of threads or on scheduling. The
// engines' existing streams (mt19937_64, xoshiro) are not changed: switching
// a stream would change every Monte Carlo golden.

#pragma once

#include <array>
#include <cstdint>

#include "qp/rng.hpp"

namespace qp {

using PhiloxCounter = std::array<std::uint32_t, 4>;
using PhiloxKey = std::array<std::uint32_t, 2>;

namespace philox_detail {
inline constexpr std::uint32_t kM0 = 0xD2511F53U;
inline constexpr std::uint32_t kM1 = 0xCD9E8D57U;
inline constexpr std::uint32_t kW0 = 0x9E3779B9U;  // golden ratio
inline constexpr std::uint32_t kW1 = 0xBB67AE85U;  // sqrt(3) - 1

constexpr void mulhilo(std::uint32_t a, std::uint32_t b, std::uint32_t& hi,
                       std::uint32_t& lo) noexcept {
    const std::uint64_t p = static_cast<std::uint64_t>(a) * b;
    hi = static_cast<std::uint32_t>(p >> 32);
    lo = static_cast<std::uint32_t>(p);
}
}  // namespace philox_detail

/// One Philox4x32 block with `Rounds` rounds (10 is the standard choice).
template <int Rounds = 10>
[[nodiscard]] constexpr PhiloxCounter philox4x32(PhiloxCounter ctr, PhiloxKey key) noexcept {
    using namespace philox_detail;
    for (int r = 0; r < Rounds; ++r) {
        if (r > 0) {
            key[0] += kW0;
            key[1] += kW1;
        }
        std::uint32_t hi0 = 0, lo0 = 0, hi1 = 0, lo1 = 0;
        mulhilo(kM0, ctr[0], hi0, lo0);
        mulhilo(kM1, ctr[2], hi1, lo1);
        ctr = {hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
    }
    return ctr;
}

/// A stream over Philox4x32-10: key = 64-bit seed, counter = (lo, hi of a
/// 64-bit stream id, lo, hi of a 64-bit block index). Each block gives two
/// 64-bit words. `next_u64()` and `uniform()` (open (0, 1), see
/// `uniform_open53`) consume them in order. Stream `s` and stream `s + 1` are
/// independent by construction; there is nothing to seed or split.
class PhiloxStream {
public:
    constexpr PhiloxStream(std::uint64_t seed, std::uint64_t stream_id) noexcept
        : key_{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)},
          stream_(stream_id) {}

    /// Position the stream at word `i` (0-based) without generating the
    /// words before it.
    constexpr void seek(std::uint64_t word) noexcept {
        block_ = word / 2;
        have_ = 0;
        if (word % 2 != 0) {
            refill();
            have_ = 1;
        }
    }

    [[nodiscard]] constexpr std::uint64_t next_u64() noexcept {
        if (have_ == 0) {
            refill();
            have_ = 2;
        }
        const std::uint64_t w = buf_[2 - have_];
        --have_;
        return w;
    }

    [[nodiscard]] constexpr double uniform() noexcept { return uniform_open53(next_u64()); }

private:
    constexpr void refill() noexcept {
        const PhiloxCounter out = philox4x32<10>(
            {static_cast<std::uint32_t>(stream_), static_cast<std::uint32_t>(stream_ >> 32),
             static_cast<std::uint32_t>(block_), static_cast<std::uint32_t>(block_ >> 32)},
            key_);
        buf_[0] = (static_cast<std::uint64_t>(out[1]) << 32) | out[0];
        buf_[1] = (static_cast<std::uint64_t>(out[3]) << 32) | out[2];
        ++block_;
    }

    PhiloxKey key_;
    std::uint64_t stream_;
    std::uint64_t block_ = 0;
    std::uint64_t buf_[2] = {0, 0};
    int have_ = 0;
};

}  // namespace qp
