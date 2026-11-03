// C16. Test-and-test-and-set spinlock.
//
// Spinning on a plain load keeps the cache line shared between waiters;
// only when the lock looks free do we try the exchange, which needs the
// line exclusive. Acquire on success and release on unlock make the
// critical section's writes visible to the next owner.
#pragma once
#include <atomic>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CP_CPU_RELAX() _mm_pause()
#else
#define CP_CPU_RELAX() ((void)0)
#endif

namespace cp {

class SpinLock {
public:
    void lock() noexcept {
        for (;;) {
            if (!locked_.exchange(true, std::memory_order_acquire)) return;
            while (locked_.load(std::memory_order_relaxed)) CP_CPU_RELAX();
        }
    }
    bool try_lock() noexcept {
        return !locked_.load(std::memory_order_relaxed) && !locked_.exchange(true, std::memory_order_acquire);
    }
    void unlock() noexcept { locked_.store(false, std::memory_order_release); }

private:
    std::atomic<bool> locked_{false};
};

}  // namespace cp
