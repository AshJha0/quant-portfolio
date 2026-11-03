// libFuzzer target for the QPBF packet decoder.
//
// Build (clang): clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address,undefined \
//                cpp/ull/qpbf/fuzz/fuzz_decode.cpp -o fuzz_decode
// Run:           ./fuzz_decode -max_total_time=60 -max_len=1600
//
// Properties: no crash or sanitizer report on any input, and every packet the
// decoder accepts re-encodes to the identical bytes (the format is canonical).
#include "../qpbf.hpp"

#include <cstdlib>
#include <cstring>

using namespace ull::qpbf;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    PacketHeader h;
    Msg ms[kMaxMessagesPerPacket];
    int n = 0;
    const Status s = decode_packet(data, size, h, [&](std::uint64_t, const Msg& m) { ms[n++] = m; });
    if (s != Status::Ok) return 0;
    PacketWriter w;
    w.begin(h.session_id, h.seq);
    for (int i = 0; i < n; ++i)
        if (!w.add(ms[i])) std::abort();
    const std::size_t len = w.finish();
    if (len != size || std::memcmp(w.data(), data, size) != 0) std::abort();
    return 0;
}
