// No allocation on the hot path.
//
// Replaces global operator new/delete with counting versions, sets
// everything up, then runs the hot path (ring hop -> packet decode -> feed
// handler arbitration and gap buffering -> book update -> latency record)
// over a generated stream that includes gaps filled by the second line.
// The allocation count during the hot loop must be exactly zero.
#include "../histogram.hpp"
#include "../tsc.hpp"
#include "../ull_test.hpp"
#include "../../book/l3_book.hpp"
#include "../../mcast/feed_handler.hpp"
#include "../../qpbf/generator.hpp"
#include "../../ring/spsc_ring.hpp"

#include <atomic>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

static std::atomic<std::uint64_t> g_allocs{0};

void* operator new(std::size_t n) {
    ++g_allocs;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    ++g_allocs;
    if (void* p = std::aligned_alloc(static_cast<std::size_t>(a), (n + static_cast<std::size_t>(a) - 1) & ~(static_cast<std::size_t>(a) - 1))) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

using namespace ull;

struct Sink {
    book::BookSet* books;
    std::uint64_t errors{0};
    void on_msg(std::uint64_t, const qpbf::Msg& m) {
        const auto r = books->apply(m);
        errors += (r != book::Result::Ok && r != book::Result::Ignored);
    }
    void on_reset() { books->clear(); }
    void request_snapshot() {}
    void on_stale(bool) {}
};

struct Pkt {
    std::uint16_t len;
    std::uint8_t bytes[qpbf::kMaxPacketSize];
};

int main() {
    // ---- setup (allocations allowed) ----
    qpbf::GeneratorConfig cfg;
    cfg.instruments = 2;
    cfg.levels = 1024;
    cfg.max_live_per_instrument = 5000;
    qpbf::Generator g(cfg);
    std::vector<Pkt> packets;
    qpbf::PacketWriter w;
    std::uint64_t seq = 1;
    for (int i = 0; i < 20000; ++i) {
        w.begin(1, seq);
        for (int k = 0; k < 8; ++k) { qpbf::Msg m; g.next(m); w.add(m); }
        Pkt p{};
        p.len = static_cast<std::uint16_t>(w.finish());
        std::memcpy(p.bytes, w.data(), p.len);
        packets.push_back(p);
        seq += 8;
    }
    book::BookSet books(cfg.instruments, cfg.base_price, cfg.levels, cfg.max_live_per_instrument);
    Sink sink{&books};
    mcast::FeedHandler<Sink> fh(sink, mcast::FeedConfig{});
    ring::SpscRing<const Pkt*> rq(1024);
    bench::Histogram hist;

    // ---- hot loop (must not allocate) ----
    const std::uint64_t before = g_allocs.load();
    std::uint64_t now = 0;
    for (std::size_t i = 0; i < packets.size(); ++i) {
        // Line A drops every 50th packet; line B (one packet late) fills it.
        if (i % 50 != 7) rq.try_push(&packets[i]);
        if (i > 0) rq.try_push(&packets[i - 1]);
        const Pkt* p = nullptr;
        while (rq.try_pop(p)) {
            const auto t0 = bench::tsc_start();
            fh.on_packet(0, p->bytes, p->len, now += 1000);
            hist.record(bench::tsc_stop() - t0);
        }
    }
    rq.try_push(&packets.back());
    const Pkt* p = nullptr;
    while (rq.try_pop(p)) fh.on_packet(1, p->bytes, p->len, now += 1000);
    const std::uint64_t during = g_allocs.load() - before;

    std::printf("hot loop: %zu packets, %llu gaps, stale=%d, allocations=%llu\n", packets.size(),
                static_cast<unsigned long long>(fh.stats().gaps), fh.stale(),
                static_cast<unsigned long long>(during));
    ULL_CHECK(during == 0);
    ULL_CHECK(fh.stats().gaps > 0);
    ULL_CHECK(!fh.stale());
    ULL_CHECK(sink.errors == 0);
    ULL_CHECK(fh.next_seq() == seq);
    ULL_TEST_MAIN_END();
}
