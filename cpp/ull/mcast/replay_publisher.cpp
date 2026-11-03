// ull_replay_publisher: generate a QPBF recording, or replay one onto two
// loopback multicast lines with a TCP snapshot server.
//
//   ull_replay_publisher generate <out.qpbfrec> <packets> [seed]
//   ull_replay_publisher replay <in.qpbfrec> [packets_per_s] [drop_a_every]
//
// Replay prints the snapshot TCP port, then publishes and exits.
#include "publisher.hpp"
#include "recording.hpp"
#include "../qpbf/generator.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ull;

int main(int argc, char** argv) {
    if (argc >= 4 && !std::strcmp(argv[1], "generate")) {
        qpbf::GeneratorConfig gc;
        gc.instruments = 4;
        gc.seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
        const auto pk = mcast::generate_packets(gc, std::strtoull(argv[3], nullptr, 10), 0x51504246u);
        if (!mcast::write_recording(argv[2], pk)) { std::perror("write"); return 1; }
        std::printf("wrote %zu packets to %s\n", pk.size(), argv[2]);
        return 0;
    }
    if (argc >= 3 && !std::strcmp(argv[1], "replay")) {
        std::vector<std::vector<std::uint8_t>> pk;
        if (!mcast::read_recording(argv[2], pk)) { std::fprintf(stderr, "bad recording\n"); return 1; }
        mcast::PublisherConfig pc;
        if (argc > 3) pc.packets_per_s = std::atof(argv[3]);
        if (argc > 4) pc.drop_a_every = static_cast<std::uint32_t>(std::atoi(argv[4]));
        mcast::ReplayPublisher pub(pc);
        if (auto e = pub.open(); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        std::printf("snapshot port %u; publishing %zu packets on %s and %s port %u\n", pub.snapshot_port(),
                    pk.size(), pc.group_a, pc.group_b, pc.port);
        std::fflush(stdout);
        pub.publish(pk);
        return 0;
    }
    std::fprintf(stderr,
                 "usage: %s generate <out> <packets> [seed]\n"
                 "       %s replay <in> [packets_per_s] [drop_a_every]\n",
                 argv[0], argv[0]);
    return 2;
}
