// QPBF encoder/decoder/generator tests.
//
//   test_qpbf            unit tests + 1e6-message round trip + mutation fuzz
//   test_qpbf N          round trip of N messages (acceptance: N = 1e8)
#include "../generator.hpp"
#include "../qpbf.hpp"
#include "../../bench/ull_test.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ull::qpbf;

// Re-encode a decoded packet from its header and messages.
static std::size_t reencode(const PacketHeader& h, const Msg* ms, std::uint8_t* out) {
    PacketWriter w;
    w.begin(h.session_id, h.seq);
    for (int i = 0; i < h.msg_count; ++i) w.add(ms[i]);
    const std::size_t n = w.finish();
    std::memcpy(out, w.data(), n);
    return n;
}

static void wire_layout_is_fixed() {
    // Golden bytes for one Add: pins the spec's offsets (docs/QPBF.md s.4).
    Msg m;
    m.type = MsgType::Add;
    m.instrument = 7;
    m.ts_ns = 0x0102030405060708ull;
    m.order_id = 42;
    m.price = -3;
    m.qty = 100;
    m.side = Side::Ask;
    std::uint8_t b[40];
    ULL_CHECK(encode_msg(m, b) == 40);
    const std::uint8_t want[40] = {
        40, 0, 1, 0,  7, 0, 0, 0,  8, 7, 6, 5, 4, 3, 2, 1,
        42, 0, 0, 0, 0, 0, 0, 0,  0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        100, 0, 0, 0,  2, 0, 0, 0};
    ULL_CHECK(std::memcmp(b, want, 40) == 0);

    PacketWriter w;
    w.begin(0xAABBCCDDu, 1000);
    ULL_CHECK(w.add(m));
    const std::size_t n = w.finish();
    ULL_CHECK(n == 56);
    const std::uint8_t hdr[16] = {0x51, 0x50, 1, 1, 0xDD, 0xCC, 0xBB, 0xAA,
                                  0xE8, 0x03, 0, 0, 0, 0, 0, 0};
    ULL_CHECK(std::memcmp(w.data(), hdr, 16) == 0);
}

static void sizes_match_spec() {
    ULL_CHECK(msg_size(MsgType::Add) == 40);
    ULL_CHECK(msg_size(MsgType::Modify) == 40);
    ULL_CHECK(msg_size(MsgType::Cancel) == 32);
    ULL_CHECK(msg_size(MsgType::Execute) == 40);
    ULL_CHECK(msg_size(MsgType::Trade) == 40);
    ULL_CHECK(msg_size(MsgType::Snapshot) == 48);
    ULL_CHECK(msg_size(MsgType::Heartbeat) == 16);
}

static void snapshot_messages() {
    Msg s;
    s.type = MsgType::Snapshot;
    s.order_id = 9;
    s.price = 100;
    s.qty = 5;
    s.side = Side::Bid;
    s.as_of_seq = 77;
    Msg end;
    end.type = MsgType::Snapshot;
    end.flags = kFlagLastInSnapshot;
    end.as_of_seq = 77;
    PacketWriter w;
    w.begin(1, 77);
    w.add(s);
    w.add(end);
    const std::size_t n = w.finish();
    PacketHeader h;
    std::vector<Msg> got;
    ULL_CHECK(decode_packet(w.data(), n, h, [&](std::uint64_t, const Msg& m) { got.push_back(m); }) ==
              Status::Ok);
    ULL_CHECK(got.size() == 2 && got[0] == s && got[1] == end);

    // A marker without the LAST flag is rejected.
    std::uint8_t b[48];
    end.flags = 0;
    encode_msg(end, b);
    Msg out;
    std::size_t used;
    ULL_CHECK(decode_msg(b, 48, out, &used) == Status::BadFlags);
}

static void rejects_malformed() {
    Msg m;
    m.type = MsgType::Cancel;
    m.order_id = 5;
    m.qty = 1;
    PacketWriter w;
    w.begin(1, 1);
    w.add(m);
    const std::size_t n = w.finish();
    std::vector<std::uint8_t> p(w.data(), w.data() + n);
    PacketHeader h;
    auto dec = [&](std::vector<std::uint8_t> q) {
        return decode_packet(q.data(), q.size(), h, [](std::uint64_t, const Msg&) {});
    };
    ULL_CHECK(dec(p) == Status::Ok);
    { auto q = p; q[0] ^= 1; ULL_CHECK(dec(q) == Status::BadMagic); }
    { auto q = p; q[2] = 2; ULL_CHECK(dec(q) == Status::BadVersion); }
    { auto q = p; q[3] = 2; ULL_CHECK(dec(q) == Status::Truncated); }
    { auto q = p; q[3] = 0; ULL_CHECK(dec(q) == Status::TrailingBytes); }
    { auto q = p; q.push_back(0); ULL_CHECK(dec(q) == Status::TrailingBytes); }
    { auto q = p; q.pop_back(); ULL_CHECK(dec(q) == Status::Truncated); }
    { auto q = p; q[16 + 2] = 9; ULL_CHECK(dec(q) == Status::BadType); }
    { auto q = p; q[16 + 2] = 0; ULL_CHECK(dec(q) == Status::BadType); }
    { auto q = p; q[16] = 40; ULL_CHECK(dec(q) == Status::BadLength); }
    { auto q = p; q[16 + 3] = 1; ULL_CHECK(dec(q) == Status::BadFlags); }
    { auto q = p; q[16 + 30] = 1; ULL_CHECK(dec(q) == Status::BadPadding); }
    { auto q = p; for (int i = 0; i < 8; ++i) q[16 + 16 + i] = 0; ULL_CHECK(dec(q) == Status::BadField); }
    { auto q = p; q.resize(10); ULL_CHECK(dec(q) == Status::Truncated); }
    { std::vector<std::uint8_t> q(kMaxPacketSize + 1, 0); ULL_CHECK(dec(q) == Status::TooLarge); }
}

// Generate n messages, pack them, decode, compare field by field and
// byte by byte, and check per-type counts against the generator's ledger.
static void round_trip(std::uint64_t n, std::uint64_t seed, std::uint32_t instruments) {
    GeneratorConfig cfg;
    cfg.seed = seed;
    cfg.instruments = instruments;
    Generator g(cfg);
    Xoshiro256ss pr(seed ^ 0xABCDEF);
    Ledger decoded{};
    Msg sent[kMaxMessagesPerPacket];
    Msg got[kMaxMessagesPerPacket];
    std::uint8_t again[kMaxPacketSize];
    PacketWriter w;
    std::uint64_t seq = 1, done = 0, packets = 0, mismatches = 0, seq_errors = 0;
    Msg pending;
    bool have_pending = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (done < n) {
        const auto want = static_cast<std::uint32_t>(1 + pr.below(40));
        w.begin(0x5E55, seq);
        std::uint32_t k = 0;
        while (k < want && done + k < n) {
            if (!have_pending) { g.next(pending); have_pending = true; }
            if (!w.add(pending)) break;
            sent[k++] = pending;
            have_pending = false;
        }
        const std::size_t len = w.finish();
        PacketHeader h;
        std::uint32_t i = 0;
        const Status s = decode_packet(w.data(), len, h, [&](std::uint64_t sq, const Msg& m) {
            if (sq != seq + i) ++seq_errors;
            got[i++] = m;
            ++decoded.by_type[static_cast<std::size_t>(m.type)];
        });
        if (s != Status::Ok || i != k) { ++mismatches; break; }
        for (std::uint32_t j = 0; j < k; ++j) mismatches += !(got[j] == sent[j]);
        const std::size_t len2 = reencode(h, got, again);
        mismatches += (len2 != len || std::memcmp(again, w.data(), len) != 0);
        seq += k;
        done += k;
        ++packets;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("round trip: %llu messages in %llu packets, %.1f s (%.1f M msg/s)\n",
                static_cast<unsigned long long>(done), static_cast<unsigned long long>(packets), sec,
                done / sec / 1e6);
    for (std::size_t t = 1; t < kNumMsgTypes; ++t)
        std::printf("  %-9s generated %llu decoded %llu\n", msg_name(static_cast<MsgType>(t)),
                    static_cast<unsigned long long>(g.ledger().by_type[t]),
                    static_cast<unsigned long long>(decoded.by_type[t]));
    ULL_CHECK(mismatches == 0);
    ULL_CHECK(seq_errors == 0);
    ULL_CHECK(decoded == g.ledger());
    ULL_CHECK(g.ledger().total() == n);
}

static void generator_is_deterministic() {
    GeneratorConfig cfg;
    cfg.seed = 99;
    Generator a(cfg), b(cfg);
    Msg x, y;
    bool same = true;
    for (int i = 0; i < 100000; ++i) { a.next(x); b.next(y); same &= (x == y); }
    ULL_CHECK(same);
    cfg.seed = 100;
    Generator c(cfg);
    Generator d(GeneratorConfig{.seed = 99});
    int diff = 0;
    for (int i = 0; i < 1000; ++i) { c.next(x); d.next(y); diff += !(x == y); }
    ULL_CHECK(diff > 900);
}

// Mutation fuzz: random byte flips, truncations and extensions of valid
// packets. Property: decode never crashes (run under ASan/UBSan in CI), and
// any packet it accepts re-encodes to exactly the same bytes (canonical).
static void mutation_fuzz(std::uint64_t iters) {
    GeneratorConfig cfg;
    cfg.seed = 7;
    Generator g(cfg);
    Xoshiro256ss r(12345);
    std::uint64_t accepted = 0, noncanonical = 0;
    std::vector<std::uint8_t> buf;
    Msg ms[kMaxMessagesPerPacket];
    std::uint8_t again[kMaxPacketSize];
    for (std::uint64_t it = 0; it < iters; ++it) {
        PacketWriter w;
        w.begin(static_cast<std::uint32_t>(r()), r());
        const auto k = 1 + r.below(8);
        Msg m;
        for (std::uint64_t j = 0; j < k; ++j) { g.next(m); w.add(m); }
        const std::size_t n = w.finish();
        buf.assign(w.data(), w.data() + n);
        const auto flips = r.below(4);
        for (std::uint64_t f = 0; f < flips; ++f) buf[r.below(buf.size())] ^= static_cast<std::uint8_t>(1 + r.below(255));
        switch (r.below(4)) {
            case 0: buf.resize(r.below(buf.size() + 1)); break;
            case 1: buf.push_back(static_cast<std::uint8_t>(r())); break;
            default: break;
        }
        PacketHeader h;
        int i = 0;
        const Status s = decode_packet(buf.data(), buf.size(), h,
                                       [&](std::uint64_t, const Msg& mm) { ms[i++] = mm; });
        if (s == Status::Ok) {
            ++accepted;
            const std::size_t n2 = reencode(h, ms, again);
            noncanonical += (n2 != buf.size() || std::memcmp(again, buf.data(), n2) != 0);
        }
    }
    std::printf("mutation fuzz: %llu cases, %llu accepted, %llu non-canonical\n",
                static_cast<unsigned long long>(iters), static_cast<unsigned long long>(accepted),
                static_cast<unsigned long long>(noncanonical));
    ULL_CHECK(noncanonical == 0);
    ULL_CHECK(accepted > 0);
}

int main(int argc, char** argv) {
    if (argc > 2 && std::string(argv[1]) == "--write-seeds") {
        // Seed corpus for the libFuzzer target: 64 valid packets.
        GeneratorConfig cfg;
        Generator g(cfg);
        for (int i = 0; i < 64; ++i) {
            PacketWriter w;
            w.begin(1, 1 + i * 4);
            for (int k = 0; k < 1 + i % 4; ++k) { Msg m; g.next(m); w.add(m); }
            const auto n = w.finish();
            const std::string path = std::string(argv[2]) + "/seed" + std::to_string(i);
            if (std::FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(w.data(), 1, n, f); std::fclose(f); }
        }
        return 0;
    }
    if (argc > 1) {
        const auto n = std::strtoull(argv[1], nullptr, 10);
        round_trip(n, 2026, 4);
        ULL_TEST_MAIN_END();
    }
    sizes_match_spec();
    wire_layout_is_fixed();
    snapshot_messages();
    rejects_malformed();
    generator_is_deterministic();
    round_trip(1'000'000, 1, 1);
    round_trip(1'000'000, 2, 8);
    mutation_fuzz(300'000);
    ULL_TEST_MAIN_END();
}
