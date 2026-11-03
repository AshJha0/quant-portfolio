// Raw-byte recorder and replay for the feed handler.
//
// Plain idea: exchanges change their message formats. If we keep only the
// parsed book, a parser bug or a format change destroys the evidence. So the
// recorder sits in front of the parser and writes every frame exactly as it
// came off the wire, with the receive time and the line it came from. Replay
// feeds those bytes back through the same handler, so a recorded session is
// a deterministic test.
//
// File format (QPRC v1, all integers little-endian):
//   header : "QPRC" magic, u16 version = 1, u16 reserved = 0
//   record : u64 recv_ns, u8 line (0 = A, 1 = B, 255 = REST), u8 kind
//            (0 = websocket frame, 1 = snapshot body), u32 len, len raw bytes
// A truncated final record (a crash mid-write) is reported, not thrown, so a
// file cut short by kill -9 still replays up to the last whole record.
#pragma once

#include <cstdint>
#include <cstring>
#include <istream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qp::ull::feed {

enum class Line : std::uint8_t { A = 0, B = 1, Rest = 255 };
enum class Kind : std::uint8_t { Frame = 0, Snapshot = 1 };

struct Record {
    std::uint64_t recv_ns = 0;
    Line line = Line::A;
    Kind kind = Kind::Frame;
    std::string bytes;
    bool operator==(const Record&) const = default;
};

struct RecorderError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace detail {
template <class T>
inline void put_le(std::ostream& o, T v) {
    char b[sizeof(T)];
    for (std::size_t i = 0; i < sizeof(T); ++i) b[i] = char((std::uint64_t(v) >> (8 * i)) & 0xFF);
    o.write(b, sizeof(T));
}
template <class T>
inline bool get_le(std::istream& in, T& v) {
    unsigned char b[sizeof(T)];
    if (!in.read(reinterpret_cast<char*>(b), sizeof(T))) return false;
    std::uint64_t x = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) x |= std::uint64_t(b[i]) << (8 * i);
    v = T(x);
    return true;
}
}  // namespace detail

class Recorder {
public:
    explicit Recorder(std::ostream& out) : out_(out) {
        out_.write("QPRC", 4);
        detail::put_le<std::uint16_t>(out_, 1);
        detail::put_le<std::uint16_t>(out_, 0);
    }
    void write(std::uint64_t recv_ns, Line line, Kind kind, std::string_view bytes) {
        if (bytes.size() > 0xFFFFFFFFu) throw RecorderError("frame too large");
        detail::put_le<std::uint64_t>(out_, recv_ns);
        detail::put_le<std::uint8_t>(out_, std::uint8_t(line));
        detail::put_le<std::uint8_t>(out_, std::uint8_t(kind));
        detail::put_le<std::uint32_t>(out_, std::uint32_t(bytes.size()));
        out_.write(bytes.data(), std::streamsize(bytes.size()));
        ++count_;
    }
    void flush() { out_.flush(); }
    std::uint64_t count() const { return count_; }

private:
    std::ostream& out_;
    std::uint64_t count_ = 0;
};

class Reader {
public:
    explicit Reader(std::istream& in) : in_(in) {
        char magic[4];
        std::uint16_t ver = 0, res = 0;
        if (!in_.read(magic, 4) || std::memcmp(magic, "QPRC", 4) != 0)
            throw RecorderError("not a QPRC file");
        if (!detail::get_le(in_, ver) || !detail::get_le(in_, res) || ver != 1)
            throw RecorderError("unsupported QPRC version");
    }
    // Next whole record, or nullopt at end of file (truncated() tells which).
    std::optional<Record> next() {
        Record r;
        std::uint8_t line = 0, kind = 0;
        std::uint32_t len = 0;
        if (!detail::get_le(in_, r.recv_ns)) {
            truncated_ = in_.gcount() != 0;
            return std::nullopt;
        }
        if (!detail::get_le(in_, line) || !detail::get_le(in_, kind) || !detail::get_le(in_, len)) {
            truncated_ = true;
            return std::nullopt;
        }
        if (kind > 1 || (line > 1 && line != 255))
            throw RecorderError("corrupt QPRC record header");
        r.line = Line(line);
        r.kind = Kind(kind);
        r.bytes.resize(len);
        if (len && !in_.read(r.bytes.data(), std::streamsize(len))) {
            truncated_ = true;
            return std::nullopt;
        }
        return r;
    }
    bool truncated() const { return truncated_; }

private:
    std::istream& in_;
    bool truncated_ = false;
};

// Replay a recording into a handler with on_message(bytes, now_ns) and
// on_snapshot(bytes). Returns the number of records replayed.
template <class Handler>
std::uint64_t replay(std::istream& in, Handler& h) {
    Reader rd(in);
    std::uint64_t n = 0;
    while (auto r = rd.next()) {
        if (r->kind == Kind::Snapshot)
            h.on_snapshot(r->bytes);
        else
            h.on_message(r->bytes, std::int64_t(r->recv_ns));
        ++n;
    }
    return n;
}

}  // namespace qp::ull::feed
