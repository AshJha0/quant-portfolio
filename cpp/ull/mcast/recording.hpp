// QPBF recording file: the raw packet bytes as they would appear on the
// wire, so a replay reproduces the feed exactly (QPBF.md section 9).
//
//   file   = magic "QPBFREC1" (8 bytes) then records
//   record = u16 little-endian length, then that many packet bytes
#pragma once

#include "../qpbf/qpbf.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ull::mcast {

inline constexpr char kRecMagic[8] = {'Q', 'P', 'B', 'F', 'R', 'E', 'C', '1'};

inline bool write_recording(const char* path, const std::vector<std::vector<std::uint8_t>>& pkts) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = std::fwrite(kRecMagic, 1, 8, f) == 8;
    for (const auto& p : pkts) {
        const std::uint8_t h[2] = {static_cast<std::uint8_t>(p.size()), static_cast<std::uint8_t>(p.size() >> 8)};
        ok = ok && std::fwrite(h, 1, 2, f) == 2 && std::fwrite(p.data(), 1, p.size(), f) == p.size();
    }
    return std::fclose(f) == 0 && ok;
}

// Returns false on a missing/short/corrupt file (a record longer than the
// QPBF maximum is corrupt).
inline bool read_recording(const char* path, std::vector<std::vector<std::uint8_t>>& pkts) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char m[8];
    bool ok = std::fread(m, 1, 8, f) == 8 && std::memcmp(m, kRecMagic, 8) == 0;
    std::uint8_t h[2];
    while (ok && std::fread(h, 1, 2, f) == 2) {
        const std::size_t n = h[0] | (h[1] << 8);
        if (n > qpbf::kMaxPacketSize) { ok = false; break; }
        std::vector<std::uint8_t> p(n);
        if (std::fread(p.data(), 1, n, f) != n) { ok = false; break; }
        pkts.push_back(std::move(p));
    }
    std::fclose(f);
    return ok;
}

}  // namespace ull::mcast
