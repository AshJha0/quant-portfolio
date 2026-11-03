// Gap-recovery tests for the feed handler: every injected pattern (line
// loss filled by the other line, loss on both lines, duplicates, reordering,
// a burst wider than the window, corrupt packets, a corrupt snapshot)
// must converge to the publisher's book, and the book must be flagged stale
// whenever it is not known to be correct.
#include "../../bench/ull_test.hpp"
#include "../../book/l3_book.hpp"
#include "../../qpbf/generator.hpp"
#include "../feed_handler.hpp"
#include "../snapshot.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace ull;
using qpbf::Msg;

namespace {

constexpr std::uint32_t kInst = 2;
constexpr std::uint32_t kLevels = 512;
constexpr std::uint32_t kMaxOrders = 4000;
constexpr std::uint32_t kSession = 0xC0FFEE;

using Packet = std::vector<std::uint8_t>;

struct Stream {
    std::vector<Packet> packets;
    std::vector<std::uint64_t> last_seq;  // last seq in each packet
    std::vector<Msg> hist;                // hist[s - 1] = message with seq s
};

Stream make_stream(std::uint64_t seed, std::size_t npackets) {
    qpbf::GeneratorConfig cfg;
    cfg.seed = seed;
    cfg.instruments = kInst;
    cfg.levels = kLevels;
    cfg.max_live_per_instrument = kMaxOrders;
    qpbf::Generator g(cfg);
    qpbf::Xoshiro256ss r(seed + 1);
    Stream s;
    std::uint64_t seq = 1;
    qpbf::PacketWriter w;
    for (std::size_t i = 0; i < npackets; ++i) {
        w.begin(kSession, seq);
        const auto k = 1 + r.below(16);
        for (std::uint64_t j = 0; j < k; ++j) {
            Msg m;
            g.next(m);
            w.add(m);
            s.hist.push_back(m);
        }
        const auto n = w.finish();
        s.packets.emplace_back(w.data(), w.data() + n);
        seq += k;
        s.last_seq.push_back(seq - 1);
    }
    return s;
}

book::BookSet replay(const Stream& s, std::uint64_t upto) {
    book::BookSet b(kInst, 100'000, kLevels, kMaxOrders);
    for (std::uint64_t q = 1; q <= upto; ++q) b.apply(s.hist[q - 1]);
    return b;
}

bool same(const book::BookSet& a, const book::BookSet& b) {
    for (std::uint32_t i = 0; i < kInst; ++i) {
        if (a[i].order_count() != b[i].order_count()) return false;
        if (a[i].best_bid() != b[i].best_bid() || a[i].best_ask() != b[i].best_ask()) return false;
        std::vector<std::uint64_t> x, y;
        a[i].for_each_live([&](std::uint64_t id, qpbf::Side, std::int64_t p, std::uint32_t q) {
            x.insert(x.end(), {id, static_cast<std::uint64_t>(p), q});
        });
        b[i].for_each_live([&](std::uint64_t id, qpbf::Side, std::int64_t p, std::uint32_t q) {
            y.insert(y.end(), {id, static_cast<std::uint64_t>(p), q});
        });
        if (x != y) return false;
    }
    return true;
}

bool cheap_same(const book::BookSet& a, const book::BookSet& b) {
    for (std::uint32_t i = 0; i < kInst; ++i)
        if (a[i].order_count() != b[i].order_count() || a[i].best_bid() != b[i].best_bid() ||
            a[i].best_ask() != b[i].best_ask())
            return false;
    return true;
}

struct Sink {
    book::BookSet book{kInst, 100'000, kLevels, kMaxOrders};
    bool want_snapshot{false};
    std::uint64_t apply_errors{0}, stale_edges{0};
    void on_msg(std::uint64_t, const Msg& m) {
        const auto r = book.apply(m);
        apply_errors += (r != book::Result::Ok && r != book::Result::Ignored);
    }
    void on_reset() { book.clear(); }
    void request_snapshot() { want_snapshot = true; }
    void on_stale(bool) { ++stale_edges; }
};

struct Event {
    int line;
    std::size_t pkt;
    bool corrupt{false};
};

struct Outcome {
    bool converged{false};
    std::uint64_t violations{0};  // !stale but book differs from reference
    std::uint64_t stale_events{0};
    mcast::FeedStats stats{};
    std::uint64_t apply_errors{0};
};

Outcome run(const Stream& s, const std::vector<Event>& sched, mcast::FeedConfig cfg,
            int corrupt_snapshots = 0) {
    Sink sink;
    mcast::FeedHandler<Sink> fh(sink, cfg);
    book::BookSet ref(kInst, 100'000, kLevels, kMaxOrders);
    std::uint64_t ref_upto = 0, now = 0;
    std::size_t pub_pos = 0;
    long snap_due = -1;
    Outcome o;
    auto serve = [&] {
        const std::uint64_t as_of = s.last_seq[pub_pos];
        const auto pb = replay(s, as_of);
        bool first = true;
        mcast::build_snapshot(pb, kSession, as_of, now, [&](const std::uint8_t* p, std::size_t n) {
            Packet q(p, p + n);
            if (first && corrupt_snapshots > 0) {
                q[2] ^= 0xFF;
                --corrupt_snapshots;
            }
            first = false;
            fh.on_snapshot_packet(q.data(), q.size());
        });
    };
    auto check = [&](std::size_t step) {
        if (fh.stale()) {
            ++o.stale_events;
            return;
        }
        const std::uint64_t upto = fh.next_seq() - 1;
        if (upto < ref_upto) {
            ref = replay(s, upto);
            ref_upto = upto;
        }
        while (ref_upto < upto) ref.apply(s.hist[ref_upto++]);
        const bool ok = (step % 64 == 0) ? same(sink.book, ref) : cheap_same(sink.book, ref);
        o.violations += !ok;
    };
    long step = 0;
    for (const auto& e : sched) {
        now += 1'000;
        pub_pos = std::max(pub_pos, e.pkt);
        Packet q = s.packets[e.pkt];
        // Structural corruption (the decoder must reject it). Payload bit
        // flips inside a field are the UDP checksum's job (QPBF.md s.8).
        if (e.corrupt) q[(e.pkt % 3 == 0) ? 0 : (e.pkt % 3 == 1 ? 2 : 18)] ^= 0x5A;
        fh.on_packet(e.line, q.data(), q.size(), now);
        if (sink.want_snapshot && snap_due < 0) {
            snap_due = step + 20;
            sink.want_snapshot = false;
        }
        if (snap_due >= 0 && step >= snap_due) {
            serve();
            snap_due = -1;
        }
        check(static_cast<std::size_t>(step));
        ++step;
    }
    // Quiet period: ticks fire any pending timeout; serve snapshots.
    for (int i = 0; i < 50 && (fh.stale() || sink.want_snapshot || snap_due >= 0); ++i) {
        now += cfg.gap_timeout_ns;
        fh.tick(now);
        if (sink.want_snapshot) {
            sink.want_snapshot = false;
            serve();
            snap_due = -1;
        }
        if (snap_due >= 0) {
            serve();
            snap_due = -1;
        }
    }
    const auto final_ref = replay(s, s.last_seq.back());
    o.converged =
        !fh.stale() && fh.next_seq() == s.last_seq.back() + 1 && same(sink.book, final_ref);
    o.stats = fh.stats();
    o.apply_errors = sink.apply_errors;
    return o;
}

void report(const char* name, const Outcome& o) {
    std::printf(
        "%-28s converged=%d violations=%llu stale_events=%llu gaps=%llu dups=%llu "
        "snap_req=%llu snap_applied=%llu bad=%llu evicted=%llu\n",
        name, o.converged, static_cast<unsigned long long>(o.violations),
        static_cast<unsigned long long>(o.stale_events),
        static_cast<unsigned long long>(o.stats.gaps),
        static_cast<unsigned long long>(o.stats.duplicates),
        static_cast<unsigned long long>(o.stats.snapshots_requested),
        static_cast<unsigned long long>(o.stats.snapshots_applied),
        static_cast<unsigned long long>(o.stats.bad_packets),
        static_cast<unsigned long long>(o.stats.overflow_evictions));
}

std::vector<Event> interleave(std::size_t n, std::function<bool(int, std::size_t)> keep,
                              std::size_t lag_b = 0) {
    std::vector<Event> v;
    for (std::size_t i = 0; i < n + lag_b; ++i) {
        if (i < n && keep(0, i)) v.push_back({0, i});
        if (i >= lag_b && keep(1, i - lag_b)) v.push_back({1, i - lag_b});
    }
    return v;
}

}  // namespace

int main() {
    const Stream s = make_stream(42, 6000);
    const std::size_t n = s.packets.size();
    const std::size_t tail = n - 20;  // never drop the final packets (no heartbeat after)
    mcast::FeedConfig cfg;
    cfg.window = 1u << 12;
    cfg.gap_timeout_ns = 50'000;
    qpbf::Xoshiro256ss r(9);

    {
        auto o = run(s, interleave(n, [](int, std::size_t) { return true; }), cfg);
        report("clean A+B", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.gaps == 0);
        ULL_CHECK(o.stats.snapshots_requested == 0 && o.apply_errors == 0);
        ULL_CHECK(o.stats.duplicates == s.hist.size());  // every B copy is a duplicate
    }
    {
        std::vector<bool> dropA(n);
        // Packet 0 is kept: a receiver whose first packet is not seq 1 must
        // treat it as a late join and fetch a snapshot (tested below).
        for (std::size_t i = 1; i < tail; ++i) dropA[i] = r.below(10) == 0;
        auto o = run(
            s, interleave(n, [&](int l, std::size_t i) { return l == 1 || !dropA[i]; }, 3), cfg);
        report("A drops 10%, B lags 3", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.gaps > 0 && o.stale_events > 0);
        ULL_CHECK(o.stats.snapshots_requested == 0);  // B filled every hole in time
    }
    {
        auto o =
            run(s, interleave(n, [&](int, std::size_t i) { return i < 1000 || i >= 1030; }), cfg);
        report("both drop 30 packets", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.snapshots_applied >= 1);
    }
    {
        std::vector<Event> v;
        for (std::size_t i = 0; i < n; ++i) {
            v.push_back({0, i});
            if (r.below(5) == 0) v.push_back({0, i});
            v.push_back({1, i});
        }
        auto o = run(s, v, cfg);
        report("duplicates on A", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.gaps == 0);
    }
    {
        auto v = interleave(n, [](int, std::size_t) { return true; });
        for (std::size_t i = 0; i + 3 < v.size() && v[i].pkt < tail; i += 2)
            if (r.below(8) == 0) std::swap(v[i], v[i + 2]);  // A(i) <-> A(i+1)
        auto o = run(s, v, cfg);
        report("reorder", o);
        ULL_CHECK(o.converged && o.violations == 0);
    }
    {
        auto o =
            run(s, interleave(n, [&](int, std::size_t i) { return i < 500 || i >= 3500; }), cfg);
        report("burst 3000 pkts > window", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.snapshots_applied >= 1);
        ULL_CHECK(o.stale_events > 0);
    }
    {
        std::vector<Event> v;
        for (std::size_t i = 0; i < n; ++i) {
            if (i < tail && r.below(20) == 0) v.push_back({0, i, true});
            v.push_back({0, i});
            v.push_back({1, i});
        }
        auto o = run(s, v, cfg);
        report("corrupt packets", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.bad_packets > 0 && o.stats.gaps == 0);
    }
    {
        auto o = run(s, interleave(n, [&](int, std::size_t i) { return i < 1000 || i >= 1030; }),
                     cfg, 1);
        report("gap + corrupt snapshot", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.snapshots_requested >= 2);
    }
    {
        // Join mid-session: first packet seen is not seq 1.
        std::vector<Event> v;
        for (std::size_t i = 200; i < n; ++i) v.push_back({0, i});
        auto o = run(s, v, cfg);
        report("late join", o);
        ULL_CHECK(o.converged && o.violations == 0 && o.stats.snapshots_applied >= 1);
    }
    {
        // Sink-detected inconsistency (QPBF.md s.5 rule 5) forces recovery.
        Sink sink;
        mcast::FeedHandler<Sink> fh(sink, cfg);
        for (std::size_t i = 0; i < 100; ++i)
            fh.on_packet(0, s.packets[i].data(), s.packets[i].size(), i * 1000);
        ULL_CHECK(!fh.stale() && !sink.want_snapshot);
        fh.invalidate();
        ULL_CHECK(fh.stale() && sink.want_snapshot && fh.awaiting_snapshot());
        const auto pb = replay(s, s.last_seq[99]);
        mcast::build_snapshot(
            pb, kSession, s.last_seq[99], 0,
            [&](const std::uint8_t* p, std::size_t n) { fh.on_snapshot_packet(p, n); });
        ULL_CHECK(!fh.stale() && same(sink.book, pb));
    }
    ULL_TEST_MAIN_END();
}
