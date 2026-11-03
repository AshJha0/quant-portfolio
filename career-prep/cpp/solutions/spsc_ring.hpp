// C13. Single-producer single-consumer lock-free ring buffer.
//
// The producer owns head_, the consumer owns tail_. Each side reads the
// other side's index with acquire and publishes its own with release, which
// orders the slot write before the index the other side will read. The
// indices live on separate cache lines so the two cores do not fight over
// one line (false sharing). Capacity is a power of two so the slot is
// index & (N - 1), and the indices run freely (unsigned wrap-around is
// well defined), so "full" is head - tail == N with no wasted slot.
#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

namespace cp {

template <class T, std::size_t N>
class SpscRing {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "capacity must be a power of two");

public:
    bool try_push(const T& v) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_.load(std::memory_order_acquire) == N) return false;  // full
        buf_[h & (N - 1)] = v;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    std::optional<T> try_pop() {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return std::nullopt;  // empty
        T v = buf_[t & (N - 1)];
        tail_.store(t + 1, std::memory_order_release);
        return v;
    }

    static constexpr std::size_t capacity() { return N; }

private:
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
    alignas(64) std::array<T, N> buf_{};
};

}  // namespace cp
