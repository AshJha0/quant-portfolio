// SPSC ring tests. Built twice by CMake: plain, and with -fsanitize=thread
// (target ull_ring_tsan) so TSan checks the two-thread transfer for races.
#include "../spsc_ring.hpp"
#include "../../bench/ull_test.hpp"

#include <cstdint>
#include <string>
#include <thread>

using ull::ring::SpscRing;

static void single_thread_semantics() {
    SpscRing<int> r(4);
    int v = 0;
    ULL_CHECK(!r.try_pop(v));
    for (int i = 0; i < 4; ++i) ULL_CHECK(r.try_push(i));
    ULL_CHECK(!r.try_push(99));  // full at exactly capacity
    ULL_CHECK(r.size_approx() == 4);
    ULL_CHECK(r.try_pop(v) && v == 0);
    ULL_CHECK(r.try_push(4));
    for (int want = 1; want <= 4; ++want) ULL_CHECK(r.try_pop(v) && v == want);
    ULL_CHECK(!r.try_pop(v));
}

static void rejects_bad_capacity() {
    bool threw = false;
    try {
        SpscRing<int> r(6);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ULL_CHECK(threw);
}

static void non_trivial_type() {
    SpscRing<std::string> r(8);
    ULL_CHECK(r.try_emplace(100, 'x'));
    ULL_CHECK(r.try_push(std::string("abc")));
    std::string s;
    ULL_CHECK(r.try_pop(s) && s.size() == 100);
    // one element left in the ring: the destructor must free it (ASan/LSan).
}

// Two threads, every value delivered once and in order; wrap-around many
// times on a small ring so the cached-index paths are exercised.
static void two_thread_transfer(std::uint64_t n) {
    SpscRing<std::uint64_t> r(64);
    std::thread prod([&] {
        for (std::uint64_t i = 1; i <= n; ++i) {
            while (!r.try_push(i)) {
            }
        }
    });
    std::uint64_t expect = 1, v = 0, sum = 0;
    bool ordered = true;
    while (expect <= n) {
        if (r.try_pop(v)) {
            ordered &= (v == expect);
            sum += v;
            ++expect;
        }
    }
    prod.join();
    ULL_CHECK(ordered);
    ULL_CHECK(sum == n * (n + 1) / 2);
}

int main() {
    single_thread_semantics();
    rejects_bad_capacity();
    non_trivial_type();
    two_thread_transfer(2'000'000);
    ULL_TEST_MAIN_END();
}
