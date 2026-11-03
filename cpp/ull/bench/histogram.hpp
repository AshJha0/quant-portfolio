// Log-linear latency histogram in the style of HdrHistogram (Tene).
//
// Plain idea: values below 2^S are counted exactly; above that, each power
// of two [2^e, 2^(e+1)) is split into 2^(S-1) equal sub-buckets, so the
// bucket width is at most value / 2^(S-1). With S = 11 every recorded value
// is kept to within 1/1024 (about 0.1 %, three significant digits) from one
// tick up to 2^63, in a fixed array of about 56k counters allocated once.
// Recording is an index computation and an increment: no allocation, no
// branches on data beyond one compare.
//
// Formal statement: for v < 2^S, index(v) = v. For v >= 2^S, let
// e = floor(log2 v) - (S - 1); then index(v) = 2^S + (e - 1) * 2^(S-1) +
// ((v >> e) - 2^(S-1)). The bucket covers [ (v >> e) << e, ((v >> e) + 1) << e ).
// value_at_percentile reports the bucket's highest value, as HdrHistogram's
// "highest equivalent value" does, so a reported percentile is never below
// the true one.
//
// This is a re-implementation of the HdrHistogram layout idea, not the
// library; it is cross-checked against an exact sorted-array percentile in
// tests/test_bench.cpp.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>

namespace ull::bench {

class Histogram {
public:
    static constexpr int kS = 11;
    static constexpr std::uint64_t kExact = 1ull << kS;       // 2048
    static constexpr std::uint64_t kHalf = 1ull << (kS - 1);  // 1024
    static constexpr std::size_t kBuckets = kExact + (64 - kS) * kHalf;

    Histogram() : c_(std::make_unique<std::uint64_t[]>(kBuckets)) {}

    static std::size_t index_of(std::uint64_t v) noexcept {
        if (v < kExact) return static_cast<std::size_t>(v);
        const int e = (63 - std::countl_zero(v)) - (kS - 1);
        return static_cast<std::size_t>(kExact + static_cast<std::uint64_t>(e - 1) * kHalf +
                                        ((v >> e) - kHalf));
    }
    static std::uint64_t highest_equivalent(std::size_t i) noexcept {
        if (i < kExact) return i;
        const std::uint64_t k = i - kExact;
        const int e = static_cast<int>(k / kHalf) + 1;
        const std::uint64_t sub = (k % kHalf) + kHalf;
        return ((sub + 1) << e) - 1;
    }

    void record(std::uint64_t v) noexcept {
        ++c_[index_of(v)];
        ++n_;
        if (v > max_) max_ = v;
        if (v < min_) min_ = v;
    }
    void record_n(std::uint64_t v, std::uint64_t k) noexcept {
        if (k == 0) return;
        c_[index_of(v)] += k;
        n_ += k;
        if (v > max_) max_ = v;
        if (v < min_) min_ = v;
    }
    void reset() noexcept {
        for (std::size_t i = 0; i < kBuckets; ++i) c_[i] = 0;
        n_ = 0;
        max_ = 0;
        min_ = UINT64_MAX;
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return n_; }
    [[nodiscard]] std::uint64_t max() const noexcept { return max_; }
    [[nodiscard]] std::uint64_t min() const noexcept { return n_ ? min_ : 0; }

    // Bucket-top value of the sample with rank ceil(p/100 * n) (nearest-rank
    // definition, Hyndman and Fan 1996 type 1), capped at the recorded max.
    [[nodiscard]] std::uint64_t value_at_percentile(double p) const noexcept {
        if (n_ == 0) return 0;
        auto target = static_cast<std::uint64_t>(std::ceil(p / 100.0 * static_cast<double>(n_)));
        if (target < 1) target = 1;
        if (target > n_) target = n_;
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < kBuckets; ++i) {
            acc += c_[i];
            if (acc >= target) {
                const std::uint64_t h = highest_equivalent(i);
                return h < max_ ? h : max_;
            }
        }
        return max_;
    }
    // Number of samples with value >= v (bucket resolution).
    [[nodiscard]] std::uint64_t count_at_or_above(std::uint64_t v) const noexcept {
        std::uint64_t acc = 0;
        for (std::size_t i = index_of(v); i < kBuckets; ++i) acc += c_[i];
        return acc;
    }

private:
    std::unique_ptr<std::uint64_t[]> c_;
    std::uint64_t n_{0};
    std::uint64_t max_{0};
    std::uint64_t min_{UINT64_MAX};
};

}  // namespace ull::bench
