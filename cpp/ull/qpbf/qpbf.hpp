// QPBF: the quant-portfolio binary feed. Encoder and strict decoder.
//
// The wire format is specified in docs/QPBF.md; this header is the C++
// reference implementation (the Rust lane implements the same spec). All
// integers are little-endian and are read and written byte by byte, so the
// code is correct on any host. Every message has a fixed size per type.
//
// The decoder is strict: it rejects anything the spec does not allow
// (wrong magic, version, length, type, side, flags, non-zero padding,
// truncation, trailing bytes). Strictness is what makes "decode then
// re-encode" bit-exact, and it is what the fuzzer checks.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ull::qpbf {

inline constexpr std::uint16_t kMagic = 0x5051;  // bytes 'Q','P' on the wire
inline constexpr std::uint8_t kVersion = 1;
inline constexpr std::size_t kPacketHeaderSize = 16;
inline constexpr std::size_t kMaxPacketSize = 1472;  // 1500 MTU - IPv4 - UDP
inline constexpr std::size_t kMaxMessagesPerPacket = 255;

enum class MsgType : std::uint8_t {
    Add = 1,
    Modify = 2,
    Cancel = 3,
    Execute = 4,
    Trade = 5,
    Snapshot = 6,
    Heartbeat = 7,
};
inline constexpr std::size_t kNumMsgTypes = 8;  // index 0 unused

enum class Side : std::uint8_t { None = 0, Bid = 1, Ask = 2 };

inline constexpr std::uint8_t kFlagLastInSnapshot = 0x01;

// Size in bytes of each message type, header included.
constexpr std::size_t msg_size(MsgType t) noexcept {
    switch (t) {
        case MsgType::Add: return 40;
        case MsgType::Modify: return 40;
        case MsgType::Cancel: return 32;
        case MsgType::Execute: return 40;
        case MsgType::Trade: return 40;
        case MsgType::Snapshot: return 48;
        case MsgType::Heartbeat: return 16;
    }
    return 0;
}

constexpr const char* msg_name(MsgType t) noexcept {
    switch (t) {
        case MsgType::Add: return "add";
        case MsgType::Modify: return "modify";
        case MsgType::Cancel: return "cancel";
        case MsgType::Execute: return "execute";
        case MsgType::Trade: return "trade";
        case MsgType::Snapshot: return "snapshot";
        case MsgType::Heartbeat: return "heartbeat";
    }
    return "?";
}

// One decoded message. Fields a type does not carry are zero.
struct Msg {
    MsgType type{MsgType::Heartbeat};
    std::uint8_t flags{0};
    std::uint32_t instrument{0};
    std::uint64_t ts_ns{0};
    std::uint64_t order_id{0};
    std::int64_t price{0};  // integer ticks
    std::uint32_t qty{0};
    Side side{Side::None};
    std::uint64_t match_id{0};
    std::uint64_t as_of_seq{0};  // Snapshot only

    friend bool operator==(const Msg&, const Msg&) = default;
};

struct PacketHeader {
    std::uint8_t msg_count{0};
    std::uint32_t session_id{0};
    std::uint64_t seq{0};  // sequence number of the first message
    friend bool operator==(const PacketHeader&, const PacketHeader&) = default;
};

enum class Status : std::uint8_t {
    Ok = 0,
    Truncated,
    BadMagic,
    BadVersion,
    BadType,
    BadLength,
    BadSide,
    BadFlags,
    BadPadding,
    BadField,
    TrailingBytes,
    TooLarge,
};

constexpr const char* status_name(Status s) noexcept {
    constexpr std::array<const char*, 12> n{
        "ok",        "truncated", "bad_magic",   "bad_version",
        "bad_type",  "bad_length", "bad_side",   "bad_flags",
        "bad_padding", "bad_field", "trailing_bytes", "too_large"};
    return n[static_cast<std::size_t>(s)];
}

namespace detail {
inline void put8(std::uint8_t* p, std::uint8_t v) noexcept { p[0] = v; }
inline void put16(std::uint8_t* p, std::uint16_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
}
inline void put32(std::uint8_t* p, std::uint32_t v) noexcept {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
inline void put64(std::uint8_t* p, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
inline std::uint16_t get16(const std::uint8_t* p) noexcept {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
inline std::uint32_t get32(const std::uint8_t* p) noexcept {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
    return v;
}
inline std::uint64_t get64(const std::uint8_t* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}
inline bool zero(const std::uint8_t* p, std::size_t n) noexcept {
    std::uint8_t acc = 0;
    for (std::size_t i = 0; i < n; ++i) acc |= p[i];
    return acc == 0;
}
}  // namespace detail

// Encode one message at `out` (which must have msg_size(m.type) bytes).
// Returns bytes written. Fields the type does not carry are ignored.
inline std::size_t encode_msg(const Msg& m, std::uint8_t* out) noexcept {
    using namespace detail;
    const std::size_t n = msg_size(m.type);
    std::memset(out, 0, n);
    put16(out + 0, static_cast<std::uint16_t>(n));
    put8(out + 2, static_cast<std::uint8_t>(m.type));
    put8(out + 3, m.flags);
    put32(out + 4, m.instrument);
    put64(out + 8, m.ts_ns);
    switch (m.type) {
        case MsgType::Add:
        case MsgType::Snapshot:
            put64(out + 16, m.order_id);
            put64(out + 24, static_cast<std::uint64_t>(m.price));
            put32(out + 32, m.qty);
            put8(out + 36, static_cast<std::uint8_t>(m.side));
            if (m.type == MsgType::Snapshot) put64(out + 40, m.as_of_seq);
            break;
        case MsgType::Modify:
            put64(out + 16, m.order_id);
            put64(out + 24, static_cast<std::uint64_t>(m.price));
            put32(out + 32, m.qty);
            break;
        case MsgType::Cancel:
            put64(out + 16, m.order_id);
            put32(out + 24, m.qty);
            break;
        case MsgType::Execute:
            put64(out + 16, m.order_id);
            put32(out + 24, m.qty);
            put64(out + 32, m.match_id);
            break;
        case MsgType::Trade:
            put64(out + 16, m.match_id);
            put64(out + 24, static_cast<std::uint64_t>(m.price));
            put32(out + 32, m.qty);
            put8(out + 36, static_cast<std::uint8_t>(m.side));
            break;
        case MsgType::Heartbeat:
            break;
    }
    return n;
}

// Decode one message from [p, p+avail). On Ok, *used is the message size.
inline Status decode_msg(const std::uint8_t* p, std::size_t avail, Msg& m,
                         std::size_t* used) noexcept {
    using namespace detail;
    if (avail < 4) return Status::Truncated;
    const std::uint16_t len = get16(p);
    const std::uint8_t ty = p[2];
    if (ty == 0 || ty >= kNumMsgTypes) return Status::BadType;
    const auto t = static_cast<MsgType>(ty);
    if (len != msg_size(t)) return Status::BadLength;
    if (avail < len) return Status::Truncated;
    m = Msg{};
    m.type = t;
    m.flags = p[3];
    if (m.flags != 0 && !(t == MsgType::Snapshot && m.flags == kFlagLastInSnapshot))
        return Status::BadFlags;
    m.instrument = get32(p + 4);
    m.ts_ns = get64(p + 8);
    auto side_ok = [](std::uint8_t s) { return s == 1 || s == 2; };
    switch (t) {
        case MsgType::Add:
        case MsgType::Snapshot: {
            m.order_id = get64(p + 16);
            m.price = static_cast<std::int64_t>(get64(p + 24));
            m.qty = get32(p + 32);
            const std::uint8_t s = p[36];
            if (!zero(p + 37, 3)) return Status::BadPadding;
            if (t == MsgType::Snapshot) m.as_of_seq = get64(p + 40);
            // A snapshot end marker (order_id 0) carries no order.
            if (t == MsgType::Snapshot && m.order_id == 0) {
                if (s != 0 || m.qty != 0 || m.price != 0) return Status::BadField;
                if (m.flags != kFlagLastInSnapshot) return Status::BadFlags;
                break;
            }
            if (!side_ok(s)) return Status::BadSide;
            m.side = static_cast<Side>(s);
            if (m.order_id == 0 || m.qty == 0) return Status::BadField;
            break;
        }
        case MsgType::Modify:
            m.order_id = get64(p + 16);
            m.price = static_cast<std::int64_t>(get64(p + 24));
            m.qty = get32(p + 32);
            if (!zero(p + 36, 4)) return Status::BadPadding;
            if (m.order_id == 0 || m.qty == 0) return Status::BadField;
            break;
        case MsgType::Cancel:
            m.order_id = get64(p + 16);
            m.qty = get32(p + 24);
            if (!zero(p + 28, 4)) return Status::BadPadding;
            if (m.order_id == 0 || m.qty == 0) return Status::BadField;
            break;
        case MsgType::Execute:
            m.order_id = get64(p + 16);
            m.qty = get32(p + 24);
            if (!zero(p + 28, 4)) return Status::BadPadding;
            m.match_id = get64(p + 32);
            if (m.order_id == 0 || m.qty == 0) return Status::BadField;
            break;
        case MsgType::Trade: {
            m.match_id = get64(p + 16);
            m.price = static_cast<std::int64_t>(get64(p + 24));
            m.qty = get32(p + 32);
            const std::uint8_t s = p[36];
            if (!zero(p + 37, 3)) return Status::BadPadding;
            if (!side_ok(s)) return Status::BadSide;
            m.side = static_cast<Side>(s);
            if (m.qty == 0) return Status::BadField;
            break;
        }
        case MsgType::Heartbeat:
            break;
    }
    *used = len;
    return Status::Ok;
}

inline void encode_header(const PacketHeader& h, std::uint8_t* out) noexcept {
    using namespace detail;
    put16(out, kMagic);
    put8(out + 2, kVersion);
    put8(out + 3, h.msg_count);
    put32(out + 4, h.session_id);
    put64(out + 8, h.seq);
}

inline Status decode_header(const std::uint8_t* p, std::size_t n,
                            PacketHeader& h) noexcept {
    using namespace detail;
    if (n < kPacketHeaderSize) return Status::Truncated;
    if (n > kMaxPacketSize) return Status::TooLarge;
    if (get16(p) != kMagic) return Status::BadMagic;
    if (p[2] != kVersion) return Status::BadVersion;
    h.msg_count = p[3];
    h.session_id = get32(p + 4);
    h.seq = get64(p + 8);
    return Status::Ok;
}

// Decode a whole packet, calling on_msg(seq, const Msg&) for each message.
// Validation is all-or-nothing: messages are delivered only after the whole
// packet has been checked, so a malformed tail never half-applies a packet.
template <class F>
Status decode_packet(const std::uint8_t* p, std::size_t n, PacketHeader& h,
                     F&& on_msg) noexcept {
    Status s = decode_header(p, n, h);
    if (s != Status::Ok) return s;
    std::size_t off = kPacketHeaderSize;
    Msg m;
    for (std::uint32_t i = 0; i < h.msg_count; ++i) {  // pass 1: validate
        std::size_t used = 0;
        s = decode_msg(p + off, n - off, m, &used);
        if (s != Status::Ok) return s;
        off += used;
    }
    if (off != n) return Status::TrailingBytes;
    off = kPacketHeaderSize;
    for (std::uint32_t i = 0; i < h.msg_count; ++i) {  // pass 2: deliver
        std::size_t used = 0;
        (void)decode_msg(p + off, n - off, m, &used);
        on_msg(h.seq + i, static_cast<const Msg&>(m));
        off += used;
    }
    return Status::Ok;
}

// Fixed-buffer packet builder; never allocates.
class PacketWriter {
public:
    void begin(std::uint32_t session_id, std::uint64_t first_seq) noexcept {
        hdr_ = PacketHeader{0, session_id, first_seq};
        len_ = kPacketHeaderSize;
    }
    [[nodiscard]] bool fits(MsgType t) const noexcept {
        return hdr_.msg_count < kMaxMessagesPerPacket &&
               len_ + msg_size(t) <= kMaxPacketSize;
    }
    bool add(const Msg& m) noexcept {
        if (!fits(m.type)) return false;
        len_ += encode_msg(m, buf_.data() + len_);
        ++hdr_.msg_count;
        return true;
    }
    // Writes the header and returns the finished packet length.
    std::size_t finish() noexcept {
        encode_header(hdr_, buf_.data());
        return len_;
    }
    [[nodiscard]] const std::uint8_t* data() const noexcept { return buf_.data(); }
    [[nodiscard]] std::uint8_t count() const noexcept { return hdr_.msg_count; }
    [[nodiscard]] std::uint64_t first_seq() const noexcept { return hdr_.seq; }

private:
    std::array<std::uint8_t, kMaxPacketSize> buf_{};
    PacketHeader hdr_{};
    std::size_t len_{kPacketHeaderSize};
};

}  // namespace ull::qpbf
