// End-to-end over real sockets on loopback: the replay publisher sends a
// recorded stream on multicast lines A and B (A drops every 25th packet,
// and every 400th packet is dropped on both lines), the receiver
// busy-polls both sockets, arbitrates, recovers holes through the TCP
// snapshot server, and must end with the publisher's book.
//
// Skips (exit 0 with a message) when the host cannot route loopback
// multicast, which is common in containers; CI then relies on
// test_recovery, which covers the same logic without sockets.
#include "../feed_handler.hpp"
#include "../publisher.hpp"
#include "../recording.hpp"
#include "../../bench/ull_test.hpp"
#include "../../qpbf/generator.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

using namespace ull;

namespace {

struct Sink {
    book::BookSet books{4, 100'000, 4096, 20'000};
    bool want{false};
    std::uint64_t errors{0}, stale_on{0};
    void on_msg(std::uint64_t, const qpbf::Msg& m) {
        const auto r = books.apply(m);
        errors += (r != book::Result::Ok && r != book::Result::Ignored);
    }
    void on_reset() { books.clear(); }
    void request_snapshot() { want = true; }
    void on_stale(bool s) { stale_on += s; }
};

bool same(const book::BookSet& a, const book::BookSet& b) {
    for (std::uint32_t i = 0; i < a.size(); ++i) {
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

}  // namespace

int main() {
    qpbf::GeneratorConfig gc;
    gc.instruments = 4;
    gc.seed = 77;
    auto pk = mcast::generate_packets(gc, 20'010, 0x51504246u);  // last packet not a drop-both index

    // Recording round trip through a file.
    const char* path = "ull_loopback_test.qpbfrec";
    ULL_REQUIRE(mcast::write_recording(path, pk));
    std::vector<std::vector<std::uint8_t>> back;
    ULL_REQUIRE(mcast::read_recording(path, back));
    ULL_CHECK(back == pk);
    std::remove(path);

    mcast::PublisherConfig pc;
    pc.port = 31007;
    pc.packets_per_s = 40'000;
    pc.drop_a_every = 25;
    pc.drop_both_every = 400;
    mcast::ReplayPublisher pub(pc);
    mcast::net::McastReceiver ra, rb;
    std::string err = pub.open();
    if (err.empty()) err = ra.open(pc.group_a, pc.port);
    if (err.empty()) err = rb.open(pc.group_b, pc.port);
    if (!err.empty()) {
        std::printf("SKIP: loopback multicast unavailable (%s)\n", err.c_str());
        return 0;
    }

    Sink sink;
    mcast::FeedConfig fc;
    fc.gap_timeout_ns = 2'000'000;  // 2 ms: covers A/B skew on a busy laptop
    mcast::FeedHandler<Sink> fh(sink, fc);
    std::thread th([&] { pub.publish(pk); });

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto now_ns = [&] {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count());
    };
    std::uint8_t buf[2048];
    std::uint64_t received = 0;
    const std::uint64_t last = 20'010ull * 8;
    auto last_rx = clk::now();
    while (true) {
        bool any = false;
        for (int line = 0; line < 2; ++line) {
            const ssize_t n = (line == 0 ? ra : rb).poll(buf, sizeof buf);
            if (n > 0) {
                fh.on_packet(line, buf, static_cast<std::size_t>(n), now_ns());
                ++received;
                any = true;
            }
        }
        if (any) last_rx = clk::now();
        fh.tick(now_ns());
        if (sink.want) {
            sink.want = false;
            mcast::net::Fd c = mcast::net::tcp_connect(pub.snapshot_port());
            long n = 0;
            while (c.ok() && (n = mcast::net::read_frame(c.get(), buf, sizeof buf)) > 0)
                if (!fh.on_snapshot_packet(buf, static_cast<std::size_t>(n))) break;
        }
        if (fh.next_seq() == last + 1 && !fh.stale()) break;
        if (clk::now() - last_rx > std::chrono::seconds(3)) break;
    }
    th.join();
    pub.stop_snapshot_server();

    // Publisher-side truth: replay all packets into a fresh book.
    book::BookSet truth(4, 100'000, 4096, 20'000);
    for (auto& p : pk) {
        qpbf::PacketHeader h;
        qpbf::decode_packet(p.data(), p.size(), h, [&](std::uint64_t, const qpbf::Msg& m) { truth.apply(m); });
    }
    const auto& st = fh.stats();
    std::printf("loopback: %llu datagrams, gaps %llu, dups %llu, snapshots %llu/%llu, next %llu of %llu, stale %d\n",
                static_cast<unsigned long long>(received), static_cast<unsigned long long>(st.gaps),
                static_cast<unsigned long long>(st.duplicates),
                static_cast<unsigned long long>(st.snapshots_applied),
                static_cast<unsigned long long>(st.snapshots_requested),
                static_cast<unsigned long long>(fh.next_seq() - 1), static_cast<unsigned long long>(last),
                fh.stale());
    ULL_CHECK(!fh.stale());
    ULL_CHECK(fh.next_seq() == last + 1);
    ULL_CHECK(same(sink.books, truth));
    ULL_CHECK(st.gaps > 0);
    ULL_CHECK(sink.errors == 0);
    ULL_TEST_MAIN_END();
}
