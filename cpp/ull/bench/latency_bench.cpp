// ull_latency_bench: software in-process latency, not wire-to-wire.
//
// Measures, per call, with rdtscp + lfence and a log-linear histogram:
//   decode_book   decode one QPBF packet and apply its messages to the L3 book
//   book_apply    apply one already-decoded message to the L3 book
//   ring_hop      one SPSC ring hop between two threads (pinned if possible)
// Each in closed loop (service time) and, for decode_book, open loop at a
// fixed rate (response time, coordinated-omission aware).
//
// Output is one JSON object on stdout: the system read-back (CPU model,
// hypervisor flag, affinity actually in effect, THP and huge pages, busy
// poll), the TSC calibration, and percentiles in nanoseconds. See
// docs/LATENCY_METHOD.md for what these numbers mean and do not mean.
//
// Usage: ull_latency_bench [--cpu N] [--cpu2 M] [--calls N] [--rate-per-s R]
#include "../book/l3_book.hpp"
#include "../qpbf/generator.hpp"
#include "../ring/spsc_ring.hpp"
#include "harness.hpp"
#include "histogram.hpp"
#include "sysinfo.hpp"
#include "tsc.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace ull;

namespace {

std::string pct_json(const bench::Histogram& h, double tpn) {
    char b[512];
    auto ns = [&](double p) { return h.value_at_percentile(p) / tpn; };
    std::snprintf(
        b, sizeof b,
        "{\"n\":%llu,\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"p99_9\":%.1f,\"p99_99\":%.1f,"
        "\"max\":%.1f}",
        static_cast<unsigned long long>(h.count()), ns(50), ns(90), ns(99), ns(99.9), ns(99.99),
        h.max() / tpn);
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    int cpu = -1, cpu2 = -1;
    std::uint64_t calls = 2'000'000;
    double rate = 2e5;  // below decode_book capacity on the dev laptop
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--cpu"))
            cpu = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--cpu2"))
            cpu2 = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--calls"))
            calls = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--rate-per-s"))
            rate = std::atof(argv[i + 1]);
    }
    const bool pinned = cpu >= 0 && bench::pin_to_cpu(cpu);
    const double tpn = bench::calibrate_ticks_per_ns(200);

    // Pre-generate packets and decoded messages (setup, not timed).
    qpbf::GeneratorConfig cfg;
    cfg.instruments = 4;
    cfg.levels = 4096;
    qpbf::Generator g(cfg);
    const std::uint64_t npk = calls / 8 + 1;
    std::vector<std::uint8_t> blob;
    std::vector<std::uint32_t> off, len;
    std::vector<qpbf::Msg> msgs;
    blob.reserve(npk * 300);
    qpbf::PacketWriter w;
    std::uint64_t seq = 1;
    for (std::uint64_t i = 0; i < npk; ++i) {
        w.begin(1, seq);
        for (int k = 0; k < 8; ++k) {
            qpbf::Msg m;
            g.next(m);
            w.add(m);
            msgs.push_back(m);
        }
        const auto n = w.finish();
        off.push_back(static_cast<std::uint32_t>(blob.size()));
        len.push_back(static_cast<std::uint32_t>(n));
        blob.insert(blob.end(), w.data(), w.data() + n);
        seq += 8;
    }

    auto fresh_books = [&] {
        return book::BookSet(cfg.instruments, cfg.base_price, cfg.levels,
                             cfg.max_live_per_instrument);
    };
    std::string out = "{\"label\":\"software in-process latency, not wire-to-wire\"";
    const auto si = bench::read_sysinfo();
    out += ",\"host_class\":\"";
    out += si.hypervisor == "yes" ? "shared-VM indicative" : "bare metal (verify isolation)";
    out += "\",\"pinned\":" + std::string(pinned ? "true" : "false");
    out += ",\"sysinfo\":" + si.to_json();
    char tb[64];
    std::snprintf(tb, sizeof tb, ",\"tsc_ticks_per_ns\":%.4f", tpn);
    out += tb;

    // book_apply, closed loop.
    {
        auto books = fresh_books();
        bench::Histogram svc, rsp;
        bench::RunConfig rc;
        rc.calls = std::min<std::uint64_t>(calls, msgs.size() - 20'000);
        rc.warmup = 20'000;
        bench::run_open_loop([&](std::uint64_t i) { books.apply(msgs[i]); }, rc, svc, rsp);
        out += ",\"book_apply\":" + pct_json(svc, tpn);
    }
    // decode_book, closed loop and open loop.
    for (int open = 0; open < 2; ++open) {
        auto books = fresh_books();
        bench::Histogram svc, rsp;
        bench::RunConfig rc;
        rc.warmup = 2'000;
        rc.calls = npk - rc.warmup;
        if (open) rc.interval_ticks = static_cast<std::uint64_t>(tpn * 1e9 / rate);
        bench::run_open_loop(
            [&](std::uint64_t i) {
                qpbf::PacketHeader h;
                qpbf::decode_packet(blob.data() + off[i], len[i], h,
                                    [&](std::uint64_t, const qpbf::Msg& m) { books.apply(m); });
            },
            rc, svc, rsp);
        if (open) {
            out += ",\"decode_book_open_loop\":{\"rate_per_s\":" + std::to_string(rate) +
                   ",\"service\":" + pct_json(svc, tpn) + ",\"response\":" + pct_json(rsp, tpn) +
                   "}";
        } else {
            out += ",\"decode_book\":" + pct_json(svc, tpn);
        }
    }
    // ring_hop: producer stamps the TSC, consumer records now - stamp.
    {
        ring::SpscRing<std::uint64_t> r(1024);
        std::atomic<bool> go{false};
        const std::uint64_t hops = std::min<std::uint64_t>(calls, 1'000'000);
        std::thread prod([&] {
            if (cpu2 >= 0) bench::pin_to_cpu(cpu2);
            while (!go.load(std::memory_order_acquire)) {
            }
            for (std::uint64_t i = 0; i < hops; ++i) {
                const auto t = bench::tsc_start();
                while (!r.try_push(t)) {
                }
                // pace so the ring stays near empty (measure hop, not queueing)
                const auto until = t + static_cast<std::uint64_t>(tpn * 500);
                while (bench::tsc_start() < until) {
                }
            }
        });
        bench::Histogram h;
        go.store(true, std::memory_order_release);
        std::uint64_t got = 0, v = 0;
        while (got < hops) {
            if (r.try_pop(v)) {
                const auto now = bench::tsc_stop();
                h.record(now > v ? now - v : 0);
                ++got;
            }
        }
        prod.join();
        out += ",\"ring_hop\":" + pct_json(h, tpn);
    }
    out += "}";
    std::printf("%s\n", out.c_str());
    return 0;
}
