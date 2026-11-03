// C19. Sequence lock: one writer, many readers of a small value.
//
// The writer makes the counter odd, writes, then makes it even again. A
// reader copies the value between two reads of the counter and retries if
// the counter was odd or changed. The payload is stored as relaxed atomic
// words, so the racing copy is not a data race in the C++ memory model.
// Ordering: the writer publishes each word with release, so a reader that
// acquires a new word must also see the odd counter written before it, and
// its re-read of the counter fails. This is the release/acquire variant in
// Boehm, "Can seqlocks get along with programming language memory
// models?", MSPC 2012. (The fence-based variant is a little cheaper on
// x86 reads but ThreadSanitizer does not model fences.)
#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace cp {

template <class T>
class SeqLock {
    static_assert(std::is_trivially_copyable_v<T>);
    static constexpr std::size_t kWords = (sizeof(T) + 7) / 8;

public:
    void store(const T& v) noexcept {  // single writer only
        std::array<std::uint64_t, kWords> w{};
        std::memcpy(w.data(), &v, sizeof(T));
        const std::uint64_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);
        for (std::size_t i = 0; i < kWords; ++i) data_[i].store(w[i], std::memory_order_release);
        seq_.store(s + 2, std::memory_order_release);
    }

    T load() const noexcept {
        std::array<std::uint64_t, kWords> w{};
        for (;;) {
            const std::uint64_t s0 = seq_.load(std::memory_order_acquire);
            if (s0 & 1) continue;  // a write is in progress
            for (std::size_t i = 0; i < kWords; ++i) w[i] = data_[i].load(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == s0) break;
        }
        T v;
        std::memcpy(&v, w.data(), sizeof(T));
        return v;
    }

private:
    std::atomic<std::uint64_t> seq_{0};
    std::array<std::atomic<std::uint64_t>, kWords> data_{};
};

}  // namespace cp
