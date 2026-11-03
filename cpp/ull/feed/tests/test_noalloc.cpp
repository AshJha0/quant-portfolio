// W30 acceptance "no allocation on the hot path", for the crypto feed.
//
// Global operator new is replaced with a counting version. A Binance handler
// is synced from a recorded (synthetic) fixture; then every remaining frame
// is delivered on line A and again on line B (one copy applies, one is a
// duplicate), which must allocate exactly zero times. Half the frames carry
// the combined-stream wrapper. A second case drives FlatL3Book (the Coinbase
// `full` book) through open/match/resize/done with string ids, also with zero
// allocations.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <vector>

#include "qp/ull/handlers.hpp"
#include "qp/ull/recorder.hpp"
#include "qp/ull/testing.hpp"

static std::atomic<std::uint64_t> g_allocs{0};

void* operator new(std::size_t n) {
    ++g_allocs;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace qp::ull;
using namespace qp::ull::feed;

#ifndef QP_FEED_FIXTURES
#error "QP_FEED_FIXTURES must be defined by the build"
#endif

TEST(binance_hot_path_allocates_nothing) {
    for (const char* name : {"binance_depth_21", "binance_depth_22"}) {
        std::ifstream in(std::string(QP_FEED_FIXTURES) + "/" + name + ".qprc", std::ios::binary);
        Reader rd(in);
        std::vector<Record> recs;
        while (auto r = rd.next()) recs.push_back(*r);
        BinanceDepth h("BTCUSDT");
        std::size_t i = 0;
        for (; i < recs.size(); ++i) {  // up to and including the snapshot
            if (recs[i].kind == Kind::Snapshot) {
                h.on_snapshot(recs[i].bytes);
                ++i;
                break;
            }
            h.on_message(recs[i].bytes, static_cast<std::int64_t>(recs[i].recv_ns));
        }
        CHECK(!h.stale());
        const std::uint64_t probe = g_allocs.load();
        {  // the counter is live (self-check); an explicit call is not elided
            void* volatile p = ::operator new(16);
            ::operator delete(p);
        }
        CHECK_EQ(g_allocs.load(), probe + 1);
        const std::uint64_t fast0 = h.fast_decoded(), applied0 = h.sync().stats().applied;
        std::uint64_t frames = 0;
        const std::uint64_t before = g_allocs.load();
        for (; i < recs.size(); ++i) {
            if (recs[i].kind != Kind::Frame) continue;
            const auto t = static_cast<std::int64_t>(recs[i].recv_ns);
            h.on_message(recs[i].bytes, t);
            h.on_message(recs[i].bytes, t);
            ++frames;
        }
        const std::uint64_t after = g_allocs.load();
        CHECK_EQ(after - before, 0u);
        CHECK(frames > 300);
        CHECK_EQ(h.fast_decoded() - fast0, 2 * frames);
        CHECK_EQ(h.sync().stats().applied - applied0, frames);
        CHECK(!h.stale());

        // Contrast: the same frames through the generic json::parse path (no
        // room reserved for the fast decoder) allocate on every frame. This
        // is the path the handler used before the flat book and fast decoder.
        BinanceDepth g("BTCUSDT", {}, {}, 0);
        std::size_t j = 0;
        for (; j < recs.size(); ++j) {
            if (recs[j].kind == Kind::Snapshot) {
                g.on_snapshot(recs[j].bytes);
                ++j;
                break;
            }
            g.on_message(recs[j].bytes, static_cast<std::int64_t>(recs[j].recv_ns));
        }
        const std::uint64_t g0 = g_allocs.load();
        for (; j < recs.size(); ++j)
            if (recs[j].kind == Kind::Frame)
                g.on_message(recs[j].bytes, static_cast<std::int64_t>(recs[j].recv_ns));
        CHECK(g_allocs.load() - g0 > frames);
        CHECK_EQ(g.fast_decoded(), 0u);
        CHECK(g.book() == h.book());
    }
}

TEST(flat_l3_book_ops_allocate_nothing) {
    FlatL3Book b;
    std::vector<std::tuple<std::string, Side, std::int64_t, std::int64_t>> snap{
        {"seed-bid", Side::Bid, 100 * kScale, kScale}, {"seed-ask", Side::Ask, 101 * kScale, kScale}};
    b.load(snap);
    std::vector<std::string> ids;
    for (int k = 0; k < 4000; ++k) ids.push_back("0e4f1c2a-0000-4000-8000-" + std::to_string(100000000000 + k));
    const std::uint64_t before = g_allocs.load();
    std::uint64_t x = 88172645463325252ull;
    for (int round = 0; round < 25; ++round) {
        for (int k = 0; k < 4000; ++k) {
            x ^= x << 13, x ^= x >> 7, x ^= x << 17;
            const Side s = (k & 1) ? Side::Ask : Side::Bid;
            const std::int64_t px = (s == Side::Bid ? 99 * kScale : 102 * kScale) +
                                    static_cast<std::int64_t>(x % 200) * (kScale / 100) * (s == Side::Bid ? -1 : 1);
            b.open(ids[k], s, px, kScale / 10 * static_cast<std::int64_t>(1 + x % 9));
        }
        for (int k = 0; k < 4000; ++k) {
            switch (k % 4) {
                case 0:
                    b.reduce(ids[k], kScale / 100);
                    break;
                case 1:
                    b.resize(ids[k], kScale / 20);
                    break;
                default:
                    break;
            }
        }
        for (int k = 0; k < 4000; ++k) b.done(ids[k]);
    }
    const std::uint64_t after = g_allocs.load();
    CHECK_EQ(after - before, 0u);
    CHECK_EQ(b.order_count(), 2u);
}

QPT_MAIN()
