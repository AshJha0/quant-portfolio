// Replay a QPRC recording through the cpp/ull/feed handler and print, after
// every record, "index stale seq digest" where digest is the additive level hash
// below (hex) and seq is the handler's applied update id (-1 for
// Coinbase level2, which has none). tools/recorder/tests compares this with
// the Python reference book.
//
//   replay_dump binance  <file.qprc> <SYMBOL>
//   replay_dump coinbase-level2 <file.qprc> <PRODUCT>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include "qp/ull/handlers.hpp"
#include "qp/ull/recorder.hpp"

using namespace qp::ull::feed;

// SplitMix64 finaliser; digest = sum over levels of
// mix(mix(px) ^ qty ^ tag(side)) mod 2^64 (tools/recorder/qp_recorder/refbook.py).
static std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::uint64_t digest(const L2Book& b) {
    std::uint64_t h = 0;
    for (Side s : {Side::Bid, Side::Ask})
        for (const auto& l : b.levels(s))
            if (l.qty > 0)
                h += mix(mix(std::uint64_t(l.px)) ^ std::uint64_t(l.qty) ^
                         (s == Side::Bid ? 0ULL : 0xa5a5a5a5a5a5a5a5ULL));
    return h;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: replay_dump binance|coinbase-level2 FILE SYMBOL\n");
        return 2;
    }
    std::string venue = argv[1];
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[2]);
        return 2;
    }
    Reader rd(in);
    std::uint64_t i = 0;
    if (venue == "binance") {
        BinanceDepth h(argv[3]);
        while (auto r = rd.next()) {
            if (r->kind == Kind::Snapshot)
                h.on_snapshot(r->bytes);
            else
                h.on_message(r->bytes, std::int64_t(r->recv_ns));
            std::printf("%llu %d %lld %016llx\n", (unsigned long long)i++, int(h.stale()),
                        (long long)h.sync().applied_seq(), (unsigned long long)digest(h.book()));
        }
    } else if (venue == "coinbase-level2") {
        CoinbaseLevel2 h(argv[3]);
        while (auto r = rd.next()) {
            h.on_message(r->bytes);
            std::printf("%llu %d -1 %016llx\n", (unsigned long long)i++, int(h.stale()),
                        (unsigned long long)digest(h.book()));
        }
    } else {
        std::fprintf(stderr, "unknown venue %s\n", venue.c_str());
        return 2;
    }
    if (rd.truncated()) std::fprintf(stderr, "truncated final record\n");
    return 0;
}
