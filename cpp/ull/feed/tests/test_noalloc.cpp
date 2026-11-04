// W30 acceptance "no allocation on the hot path", for the crypto feed.
//
// Global operator new is replaced with a counting version. A Binance handler
// is synced from a recorded (synthetic) fixture; then every remaining frame
// is delivered on line A and again on line B (one copy applies, one is a
// duplicate), which must allocate exactly zero times. Half the frames carry
// the combined-stream wrapper. A second case drives FlatL3Book (the Coinbase
// `full` book) through open/match/resize/done with string ids, also with zero
// allocations. Two more cases cover the Coinbase level2 decoder: zero
// allocations on the live recordings, and agreement with the generic path on
// those recordings, generated frames and mutated bytes.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "qp/ull/handlers.hpp"
#include "qp/ull/recorder.hpp"
#include "qp/ull/testing.hpp"
#include "inflate.hpp"

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
        {"seed-bid", Side::Bid, 100 * kScale, kScale},
        {"seed-ask", Side::Ask, 101 * kScale, kScale}};
    b.load(snap);
    std::vector<std::string> ids;
    for (int k = 0; k < 4000; ++k)
        ids.push_back("0e4f1c2a-0000-4000-8000-" + std::to_string(100000000000 + k));
    const std::uint64_t before = g_allocs.load();
    std::uint64_t x = 88172645463325252ull;
    for (int round = 0; round < 25; ++round) {
        for (int k = 0; k < 4000; ++k) {
            x ^= x << 13, x ^= x >> 7, x ^= x << 17;
            const Side s = (k & 1) ? Side::Ask : Side::Bid;
            const std::int64_t px =
                (s == Side::Bid ? 99 * kScale : 102 * kScale) +
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

// Every frame of every recorded Coinbase session.
static std::vector<std::string> coinbase_frames() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(QP_FEED_FIXTURES) / ".." / ".." / ".." / ".." / ".." / "tools" /
                         "recorder" / "recordings";
    std::vector<std::string> frames;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("coinbase_", 0) != 0 || !name.ends_with(".qprc.gz")) continue;
        std::ifstream in(e.path(), std::ios::binary);
        std::stringstream packed;
        packed << in.rdbuf();
        std::istringstream raw(gz::gunzip(packed.str()));
        Reader rd(raw);
        while (auto r = rd.next()) frames.push_back(r->bytes);
    }
    return frames;
}

// What the generic path would apply: nullopt unless the frame is an l2update
// for `product` whose every change is valid.
static std::optional<std::vector<LevelChange>> generic_level2(const std::string& raw,
                                                              const std::string& product) {
    try {
        const json::Value m = json::parse(raw);
        if (!m.has("type") || m.at("type").str() != "l2update") return std::nullopt;
        if (m.has("product_id") && m.at("product_id").str() != product) return std::nullopt;
        std::vector<LevelChange> out;
        for (const auto& c : m.at("changes").arr)
            out.push_back(
                {side_of(c.at(0).str()), parse_fixed(c.at(1).str()), parse_fixed(c.at(2).str())});
        return out;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

namespace {
struct Rng {
    std::uint64_t x;
    std::uint64_t next() {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        return x;
    }
    std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};
}  // namespace

static std::string generated_l2update(Rng& r) {
    std::string c;
    const std::size_t n = r.below(6);
    for (std::size_t k = 0; k < n; ++k) {
        char b[80];
        std::snprintf(b, sizeof b, "%s[\"%s\",\"%zu.%02zu\",\"0.%08zu\"]", k ? "," : "",
                      r.below(2) ? "buy" : "sell", 70000 + r.below(9000), r.below(100),
                      r.below(100000000));
        c += b;
    }
    const std::string t = "\"type\":\"l2update\"", pid = "\"product_id\":\"BTC-USD\"",
                      ch = "\"changes\":[" + c + "]",
                      time = "\"time\":\"2026-11-03T19:08:00.498600Z\"";
    switch (r.below(4)) {
        case 0:
            return "{" + ch + "," + t + "," + time + "," + pid + "}";
        case 1:
            return " { " + t + " , " + pid + " ,\n" + ch + " , " + time + " }\r\n";
        default:
            return "{" + t + "," + pid + "," + ch + "," + time + "}";
    }
}

static void mutate(Rng& r, std::string& m) {
    switch (r.below(3)) {
        case 0:
            m.resize(r.below(m.size()));
            break;
        case 1:
            for (std::size_t k = 0, n = 1 + r.below(3); k < n; ++k)
                m[r.below(m.size())] = static_cast<char>(r.next());
            break;
        default:
            m.insert(r.below(m.size()), 1, " ,:[]{}\"0.9e-\\"[r.below(14)]);
    }
}

// Decode `raw`; when the fast path accepts, the generic path must agree.
static bool agrees(const std::string& raw, std::vector<LevelChange>& out) {
    if (decode_level2(raw, "BTC-USD", out)) {
        const auto g = generic_level2(raw, "BTC-USD");
        CHECK(g.has_value());
        CHECK(g && *g == out);
        return true;
    }
    CHECK(out.empty());
    return false;
}

TEST(coinbase_level2_decoder_allocates_nothing) {
    const auto frames = coinbase_frames();
    CHECK(!frames.empty());
    std::uint64_t updates = 0;
    for (const auto& f : frames)
        if (generic_level2(f, "BTC-USD")) ++updates;
    CHECK(updates > 0);
    std::vector<LevelChange> out;
    out.reserve(4096);
    std::uint64_t fast = 0;
    const std::uint64_t before = g_allocs.load();
    for (int pass = 0; pass < 20; ++pass)
        for (const auto& f : frames)
            if (decode_level2(f, "BTC-USD", out)) ++fast;
    const std::uint64_t after = g_allocs.load();
    CHECK_EQ(after - before, 0u);
    CHECK_EQ(fast, 20 * updates);
}

TEST(coinbase_level2_decoder_agrees_with_generic_path) {
    Rng r{88172645463325252ull};
    std::vector<LevelChange> out;
    out.reserve(4096);
    std::uint64_t fuzzed = 0, accepted = 0;
    const auto frames = coinbase_frames();
    CHECK(!frames.empty());
    for (const auto& f : frames) {
        CHECK_EQ(agrees(f, out), generic_level2(f, "BTC-USD").has_value());
        for (int k = 0; k < 50; ++k) {
            std::string m = f;
            mutate(r, m);
            ++fuzzed;
            accepted += agrees(m, out) ? 1 : 0;
        }
    }
    for (int k = 0; k < 20000; ++k) {
        std::string m = generated_l2update(r);
        CHECK(agrees(m, out));
        mutate(r, m);
        ++fuzzed;
        accepted += agrees(m, out) ? 1 : 0;
    }
    std::printf("fuzzed %llu, accepted %llu\n", (unsigned long long)fuzzed,
                (unsigned long long)accepted);
    CHECK(accepted > 0 && accepted < fuzzed);

    for (const char* bad : {
             R"({"type":"snapshot","product_id":"BTC-USD","bids":[],"asks":[]})",
             R"({"type":"subscriptions","channels":[]})",
             R"({"type":"l2update","changes":[]})",
             R"({"type":"l2update","product_id":"ETH-USD","changes":[]})",
             R"({"type":"l2update","product_id":"BTC-USD","product_id":"BTC-USD","changes":[]})",
             R"({"type":"l2update","product_id":"BTC-USD","changes":[],"changes":[]})",
             R"({"type":"l2update","product_id":"BTC-USD","changes":[["bid","1","2"]]})",
             R"({"type":"l2update","product_id":"BTC-USD","changes":[["buy","1","2","3"]]})",
             R"({"type":"l2update","product_id":"BTC-USD","changes":[["buy","1","-2"]]})",
             R"({"type":"l2update","product_id":"BTC-USD","changes":"x"})",
             "{\"type\":\"l2update\",\"product_id\":\"BTC-US\\u0044\",\"changes\":[]}",
         })
        CHECK(!decode_level2(bad, "BTC-USD", out));

    // The handler gives the same book and stale flag with and without the
    // decoder, frame by frame, including the frames before the snapshot.
    CoinbaseLevel2 fast_h("BTC-USD"), generic_h("BTC-USD", 4096, 0);
    for (const auto& f : frames) {
        fast_h.on_message(f);
        generic_h.on_message(f);
        CHECK(fast_h.book() == generic_h.book());
        CHECK_EQ(fast_h.stale(), generic_h.stale());
    }
    CHECK(fast_h.fast_decoded() > 0);
    CHECK_EQ(fast_h.stats().malformed, generic_h.stats().malformed);
    CHECK_EQ(fast_h.stats().ignored, generic_h.stats().ignored);
}

QPT_MAIN()
