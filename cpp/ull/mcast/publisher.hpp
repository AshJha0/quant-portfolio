// Loopback replay publisher: sends recorded QPBF packets on two multicast
// lines (A and B) at a fixed packet rate, keeps the publisher-side book,
// and serves TCP snapshots of that book on request. Used by the
// ull_replay_publisher tool and by tests/test_loopback.cpp.
//
// Fault injection for tests: drop every k-th packet on line A, or on both
// lines (a hole only the snapshot can repair).
#pragma once

#include "../book/l3_book.hpp"
#include "../qpbf/generator.hpp"
#include "net.hpp"
#include "snapshot.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace ull::mcast {

struct PublisherConfig {
    const char* group_a{"239.255.10.1"};
    const char* group_b{"239.255.10.2"};
    std::uint16_t port{31001};
    double packets_per_s{50'000};
    std::uint32_t drop_a_every{0};
    std::uint32_t drop_both_every{0};
    std::uint32_t instruments{4};
    std::int64_t base_price{100'000};
    std::uint32_t levels{4096};
    std::uint32_t max_orders{20'000};
};

class ReplayPublisher {
public:
    explicit ReplayPublisher(PublisherConfig c)
        : cfg_(c), books_(c.instruments, c.base_price, c.levels, c.max_orders) {}

    ~ReplayPublisher() { stop_snapshot_server(); }

    std::string open() {
        if (auto e = a_.open(cfg_.group_a, cfg_.port); !e.empty()) return "line A " + e;
        if (auto e = b_.open(cfg_.group_b, cfg_.port); !e.empty()) return "line B " + e;
        listen_ = net::tcp_listen(0);
        if (!listen_.ok()) return "snapshot listen failed";
        snap_port_ = net::local_port(listen_.get());
        stop_ = false;
        server_ = std::thread([this] { serve(); });
        return {};
    }
    [[nodiscard]] std::uint16_t snapshot_port() const { return snap_port_; }

    // Publish all packets, paced. Returns packets sent per line.
    std::size_t publish(const std::vector<std::vector<std::uint8_t>>& pkts) {
        using clk = std::chrono::steady_clock;
        const auto gap = std::chrono::duration<double>(1.0 / cfg_.packets_per_s);
        const auto t0 = clk::now();
        std::size_t i = 0;
        for (const auto& p : pkts) {
            while (clk::now() - t0 < gap * static_cast<double>(i)) {
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                qpbf::PacketHeader h;
                qpbf::decode_packet(p.data(), p.size(), h,
                                    [&](std::uint64_t seq, const qpbf::Msg& m) {
                                        books_.apply(m);
                                        last_seq_ = seq;
                                    });
                session_ = h.session_id;
            }
            const bool both =
                cfg_.drop_both_every && (i % cfg_.drop_both_every) == cfg_.drop_both_every - 1;
            const bool drop_a =
                both || (cfg_.drop_a_every && (i % cfg_.drop_a_every) == cfg_.drop_a_every - 1);
            if (!drop_a) a_.send(p.data(), p.size());
            if (!both) b_.send(p.data(), p.size());
            ++i;
        }
        return i;
    }

    void stop_snapshot_server() {
        if (server_.joinable()) {
            stop_ = true;
            ::shutdown(listen_.get(), SHUT_RDWR);
            server_.join();
        }
    }

private:
    void serve() {
        while (!stop_) {
            const int c = ::accept(listen_.get(), nullptr, nullptr);
            if (c < 0) continue;
            net::Fd conn(c);
            std::vector<std::vector<std::uint8_t>> frames;
            {
                std::lock_guard<std::mutex> lk(mu_);
                build_snapshot(
                    books_, session_, last_seq_, 0,
                    [&](const std::uint8_t* p, std::size_t n) { frames.emplace_back(p, p + n); });
            }
            for (auto& f : frames)
                if (!net::write_frame(conn.get(), f.data(), f.size())) break;
        }
    }

    PublisherConfig cfg_;
    book::BookSet books_;
    net::McastSender a_, b_;
    net::Fd listen_;
    std::uint16_t snap_port_{0};
    std::thread server_;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::uint64_t last_seq_{0};
    std::uint32_t session_{0};
};

// Generate `n` packets of up to `per_packet` messages from a seeded generator.
inline std::vector<std::vector<std::uint8_t>> generate_packets(const qpbf::GeneratorConfig& gc,
                                                               std::size_t n, std::uint32_t session,
                                                               std::uint32_t per_packet = 8) {
    qpbf::Generator g(gc);
    std::vector<std::vector<std::uint8_t>> out;
    qpbf::PacketWriter w;
    std::uint64_t seq = 1;
    for (std::size_t i = 0; i < n; ++i) {
        w.begin(session, seq);
        for (std::uint32_t k = 0; k < per_packet; ++k) {
            qpbf::Msg m;
            g.next(m);
            w.add(m);
        }
        const auto len = w.finish();
        out.emplace_back(w.data(), w.data() + len);
        seq += per_packet;
    }
    return out;
}

}  // namespace ull::mcast
