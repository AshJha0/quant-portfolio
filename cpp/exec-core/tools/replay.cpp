// exec_core_replay EVENTS.bin DIGESTS.bin
//
// Reads the event file written by qp_exec.lob.events.write_events (records
// of eleven little-endian int64), applies each event to qp::exec::Book and
// writes one little-endian uint64 digest per event. Prints the event count,
// the wall time and the final order count to stdout.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "qp/exec/book.hpp"

using namespace qp::exec;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: exec_core_replay EVENTS.bin DIGESTS.bin\n");
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  std::ofstream out(argv[2], std::ios::binary);
  if (!in || !out) { std::fprintf(stderr, "cannot open files\n"); return 2; }
  Book book;
  std::vector<Fill> fills;
  std::int64_t ev[11];
  std::uint64_t n = 0;
  double t_apply = 0;
  while (in.read(reinterpret_cast<char*>(ev), sizeof ev)) {
    fills.clear();
    auto t0 = std::chrono::steady_clock::now();
    switch (ev[0]) {
      case 1: {
        NewOrder o{ev[2], static_cast<Side>(ev[3]), ev[4], ev[5], static_cast<OrdType>(ev[6]),
                   static_cast<Tif>(ev[7]), (ev[8] & 1) != 0, (ev[8] & 2) != 0, ev[9],
                   static_cast<Peg>(ev[10])};
        book.add(o, ev[1], fills);
        break;
      }
      case 2: book.cancel(ev[2], ev[1], fills); break;
      case 3: book.modify(ev[2], ev[4], ev[1], fills); break;
      case 4: book.start_auction(); break;
      case 5: book.uncross(ev[1], ev[5], fills); break;
      default: std::fprintf(stderr, "bad op at %llu\n", static_cast<unsigned long long>(n)); return 3;
    }
    t_apply += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::uint64_t d = book.digest(fills);
    out.write(reinterpret_cast<const char*>(&d), sizeof d);
    ++n;
  }
  std::printf("events %llu apply_seconds %.3f orders %zu\n", static_cast<unsigned long long>(n), t_apply,
              book.order_count());
  return 0;
}
