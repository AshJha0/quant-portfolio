// fuzz_input.hpp: turn the fuzzer's raw bytes into bounded doubles.
//
// libFuzzer hands a target an arbitrary byte string. Reading doubles
// straight from it gives NaN, infinities and subnormals most of the time,
// which only exercises the input guards. FuzzInput maps each 8-byte chunk
// to a value in a caller-chosen interval instead, so the search spends its
// time inside the domain where the numerics live. A missing chunk reads as
// zero, so every input, however short, is valid.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace qp_fuzz {

class FuzzInput {
public:
    FuzzInput(const std::uint8_t* data, std::size_t size) noexcept : data_(data), size_(size) {}

    /// Next 64 bits (zero-padded when the input runs out).
    std::uint64_t u64() noexcept {
        std::uint64_t v = 0;
        const std::size_t take = pos_ < size_ ? (size_ - pos_ < 8 ? size_ - pos_ : 8) : 0;
        if (take > 0) std::memcpy(&v, data_ + pos_, take);
        pos_ += 8;
        return v;
    }

    /// Uniform-ish value in [lo, hi]: the top 53 bits as a fraction.
    double in(double lo, double hi) noexcept {
        const double f = static_cast<double>(u64() >> 11) * 0x1.0p-53;
        return lo + (hi - lo) * f;
    }

    bool flag() noexcept { return (u64() & 1U) != 0; }

private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

}  // namespace qp_fuzz
