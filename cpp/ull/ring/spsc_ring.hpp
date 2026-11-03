// Bounded single-producer single-consumer ring buffer.
//
// Plain idea: one thread writes, one thread reads, and each side owns one
// index. The producer publishes a slot by storing its tail with release
// order; the consumer sees the slot contents after loading that tail with
// acquire order. No locks and no read-modify-write instructions are needed
// because no index has two writers (Lamport 1983, "Specifying concurrent
// program modules", ACM TOPLAS 5(2); the cached-index refinement follows
// Vyukov's bounded queue notes and Rigtorp's SPSCQueue).
//
// Formal statement: with capacity C (a power of two) and monotone 64-bit
// counters head (consumer) and tail (producer), the invariant
// 0 <= tail - head <= C holds; slot i % C is owned by the producer while
// i >= tail and by the consumer while head <= i < tail. The release store of
// tail = i + 1 happens-before the consumer's acquire load that observes it,
// so the write of slot i is visible before it is read (C++20 [intro.races]).
//
// Each index sits on its own cache line, and each side keeps a private copy
// of the other side's index so the shared line is read only when the cached
// copy says the ring looks full (producer) or empty (consumer).
//
// Worked example: C = 4, producer pushes a,b,c (tail = 3), consumer pops a
// (head = 1); a fourth and fifth push succeed (tail = 5, tail - head = 4 = C)
// and a sixth fails until the consumer pops again.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace ull::ring {

inline constexpr std::size_t kCacheLine = 64;

template <class T>
class SpscRing {
    static_assert(std::is_nothrow_move_constructible_v<T> ||
                      std::is_trivially_copyable_v<T>,
                  "T must be nothrow-movable so pop cannot leave a hole");

public:
    explicit SpscRing(std::size_t capacity)
        : cap_(capacity), mask_(capacity - 1),
          slots_(static_cast<Slot*>(::operator new(
              sizeof(Slot) * capacity, std::align_val_t{alignof(Slot)}))) {
        if (capacity < 2 || (capacity & (capacity - 1)) != 0) {
            ::operator delete(slots_, std::align_val_t{alignof(Slot)});
            throw std::invalid_argument("SpscRing capacity must be a power of two >= 2");
        }
    }

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    ~SpscRing() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            const auto t = prod_.tail.load(std::memory_order_acquire);
            for (auto h = cons_.head.load(std::memory_order_relaxed); h != t; ++h) {
                std::launder(reinterpret_cast<T*>(&slots_[h & mask_]))->~T();
            }
        }
        ::operator delete(slots_, std::align_val_t{alignof(Slot)});
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return cap_; }

    // Producer side only.
    template <class... Args>
    bool try_emplace(Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args...>) {
        const std::uint64_t t = prod_.tail.load(std::memory_order_relaxed);
        if (t - prod_.head_cache >= cap_) {
            prod_.head_cache = cons_.head.load(std::memory_order_acquire);
            if (t - prod_.head_cache >= cap_) return false;
        }
        ::new (static_cast<void*>(&slots_[t & mask_])) T(std::forward<Args>(args)...);
        prod_.tail.store(t + 1, std::memory_order_release);
        return true;
    }

    bool try_push(const T& v) noexcept(std::is_nothrow_copy_constructible_v<T>) {
        return try_emplace(v);
    }

    // Consumer side only.
    bool try_pop(T& out) noexcept {
        const std::uint64_t h = cons_.head.load(std::memory_order_relaxed);
        if (h == cons_.tail_cache) {
            cons_.tail_cache = prod_.tail.load(std::memory_order_acquire);
            if (h == cons_.tail_cache) return false;
        }
        T* p = std::launder(reinterpret_cast<T*>(&slots_[h & mask_]));
        out = std::move(*p);
        p->~T();
        cons_.head.store(h + 1, std::memory_order_release);
        return true;
    }

    // Approximate; exact only when both sides are quiescent.
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const auto t = prod_.tail.load(std::memory_order_acquire);
        const auto h = cons_.head.load(std::memory_order_acquire);
        return static_cast<std::size_t>(t - h);
    }

private:
    struct alignas(T) Slot {
        std::byte b[sizeof(T)];
    };
    struct alignas(kCacheLine) ProducerSide {
        std::atomic<std::uint64_t> tail{0};
        std::uint64_t head_cache{0};
    };
    struct alignas(kCacheLine) ConsumerSide {
        std::atomic<std::uint64_t> head{0};
        std::uint64_t tail_cache{0};
    };

    const std::size_t cap_;
    const std::size_t mask_;
    Slot* slots_;
    ProducerSide prod_;
    ConsumerSide cons_;
};

}  // namespace ull::ring
