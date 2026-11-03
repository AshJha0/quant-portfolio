// Multi-threaded checks for C13, C14, C16, C17 and C19. Built with
// ThreadSanitizer, so a data race fails the run even if the results look
// right; the CHECKs catch lost, duplicated, reordered or torn values.
#include <atomic>
#include <cstdint>
#include <future>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "blocking_queue.hpp"
#include "check.hpp"
#include "seqlock.hpp"
#include "spinlock.hpp"
#include "spsc_ring.hpp"
#include "thread_pool.hpp"

namespace {

void check_spsc() {
    cp::SpscRing<std::uint64_t, 64> ring;
    constexpr std::uint64_t kN = 200000;
    std::thread producer([&] {
        for (std::uint64_t i = 1; i <= kN; ++i)
            while (!ring.try_push(i)) std::this_thread::yield();
    });
    std::uint64_t expect = 1;
    while (expect <= kN) {
        if (auto v = ring.try_pop()) {
            CHECK(*v == expect);  // FIFO, nothing lost or duplicated
            ++expect;
        }
    }
    producer.join();
    CHECK(!ring.try_pop());
    // full/empty edges, single-threaded
    cp::SpscRing<int, 2> small;
    CHECK(small.try_push(1) && small.try_push(2) && !small.try_push(3));
    CHECK(*small.try_pop() == 1 && small.try_push(3) && *small.try_pop() == 2 && *small.try_pop() == 3);
    CHECK(!small.try_pop());
    PASS("C13");
}

void check_blocking_queue() {
    cp::BlockingQueue<std::int64_t> q;
    constexpr int kProducers = 4, kConsumers = 4, kPer = 20000;
    std::atomic<std::int64_t> sum{0}, count{0};
    std::vector<std::thread> consumers;
    for (int c = 0; c < kConsumers; ++c)
        consumers.emplace_back([&] {
            while (auto v = q.pop()) {
                sum += *v;
                ++count;
            }
        });
    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p)
        producers.emplace_back([&, p] {
            for (int i = 0; i < kPer; ++i) CHECK(q.push(static_cast<std::int64_t>(p) * kPer + i + 1));
        });
    for (auto& t : producers) t.join();
    q.close();
    for (auto& t : consumers) t.join();
    const std::int64_t n = std::int64_t{kProducers} * kPer;
    CHECK(count == n && sum == n * (n + 1) / 2);
    CHECK(!q.push(1) && !q.pop());  // closed and drained
    PASS("C14");
}

void check_spinlock() {
    cp::SpinLock lock;
    std::int64_t counter = 0;  // plain int: the lock is the only protection
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
        ts.emplace_back([&] {
            for (int i = 0; i < 50000; ++i) {
                std::lock_guard g(lock);
                ++counter;
            }
        });
    for (auto& t : ts) t.join();
    CHECK(counter == 200000);
    CHECK(lock.try_lock() && !lock.try_lock());
    lock.unlock();
    PASS("C16");
}

void check_thread_pool() {
    std::vector<std::future<int>> fs;
    {
        cp::ThreadPool pool(4);
        for (int i = 1; i <= 1000; ++i) fs.push_back(pool.submit([i] { return i * i; }));
        auto bad = pool.submit([]() -> int { throw std::runtime_error("boom"); });
        bool threw = false;
        try {
            bad.get();
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }  // destructor drains and joins
    long long total = 0;
    for (auto& f : fs) total += f.get();
    CHECK(total == 1000LL * 1001 * 2001 / 6);
    PASS("C17");
}

struct Pair {
    std::uint64_t a, b, c;
};

void check_seqlock() {
    cp::SeqLock<Pair> sl;
    sl.store({0, 0, 0});
    std::atomic<bool> done{false};
    constexpr std::uint64_t kWrites = 100000;
    std::thread writer([&] {
        for (std::uint64_t i = 1; i <= kWrites; ++i) sl.store({i, i * 3, i * 7});
        done = true;
    });
    std::uint64_t last = 0, reads = 0;
    while (!done || reads == 0) {
        Pair p = sl.load();
        CHECK(p.b == p.a * 3 && p.c == p.a * 7);  // never torn
        CHECK(p.a >= last);  // never goes backwards
        last = p.a;
        ++reads;
    }
    writer.join();
    CHECK(sl.load().a == kWrites);
    PASS("C19");
}

}  // namespace

int main() {
    check_spsc();
    check_blocking_queue();
    check_spinlock();
    check_thread_pool();
    check_seqlock();
    return 0;
}
