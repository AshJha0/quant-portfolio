// C15. Fixed-capacity object pool: no heap allocation after construction.
//
// Storage is an array of raw, suitably aligned slots; the free slots form an
// intrusive singly linked list threaded through the slot memory itself.
#pragma once
#include <cstddef>
#include <new>
#include <utility>

namespace cp {

template <class T, std::size_t N>
class ObjectPool {
    static_assert(N >= 1);
    union Slot {
        Slot* next;
        alignas(T) unsigned char bytes[sizeof(T)];
    };

public:
    ObjectPool() noexcept {
        for (std::size_t i = 0; i + 1 < N; ++i) slots_[i].next = &slots_[i + 1];
        slots_[N - 1].next = nullptr;
        free_ = &slots_[0];
    }
    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;

    // Returns nullptr when the pool is exhausted.
    template <class... A>
    T* create(A&&... a) {
        if (!free_) return nullptr;
        Slot* s = free_;
        free_ = s->next;
        try {
            T* p = ::new (static_cast<void*>(s->bytes)) T(std::forward<A>(a)...);
            ++live_;
            return p;
        } catch (...) {
            s->next = free_;  // the constructor threw: give the slot back
            free_ = s;
            throw;
        }
    }

    void destroy(T* p) noexcept {
        p->~T();
        Slot* s = reinterpret_cast<Slot*>(p);
        s->next = free_;
        free_ = s;
        --live_;
    }

    std::size_t live() const noexcept { return live_; }

private:
    Slot slots_[N];
    Slot* free_ = nullptr;
    std::size_t live_ = 0;
};

}  // namespace cp
