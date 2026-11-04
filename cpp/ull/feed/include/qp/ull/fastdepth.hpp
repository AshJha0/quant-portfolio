// Non-allocating Binance diff-depth decoder.
//
// Plain idea: json::parse builds a tree of std::string and std::map nodes
// for every message, which allocates many times. A diff-depth event has a
// fixed, flat shape, so the hot path reads it straight from the bytes into a
// DepthDelta the caller owns and reuses (its vector is cleared, never grown
// past the capacity reserved up front).
//
// The fast decoder never decides that a message is malformed. Anything it
// does not fully understand - an escape or non-ASCII byte in a string, a
// duplicate key, `a` before `b`, a different symbol, a snapshot body, a
// non-depth event, more changes than the reserved capacity - returns false,
// and the handler runs the generic json::parse path, which owns every error
// count. So the fast path can only be faster, never different; the
// differential test in tests/test_feed.cpp checks every accepted frame
// against the generic path, on the fixtures and on fuzzed bytes.
//
// Formal statement: decode_depth(raw, sym, d) returns true only if raw is a
// JSON object that json::parse accepts, optionally wrapped exactly as
// {"stream": <string>, "data": <event>}, whose event has "e":"depthUpdate",
// "s" equal to sym, plain non-negative integers U <= u that fit int64, and
// arrays "b" then "a" of [price, size] string pairs that parse_fixed accepts.
// Then d.first, d.last and d.changes equal what the generic path decodes.
//
// Worked example: {"e":"depthUpdate","s":"X","U":5,"u":6,"b":[["1.5","2"]],"a":[]}
// with sym "X" gives first 5, last 6 and one change (Bid, 150000000, 200000000).
//
// Coinbase level2: decode_level2(raw, product, out) applies the same rule to
// the `l2update` message of the `level2` and `level2_batch` channels. It
// returns true only if raw is a JSON object that json::parse accepts with
// exactly one "type":"l2update", exactly one "product_id" equal to product,
// and exactly one "changes" array of [side, price, size] string triples, side
// being buy or sell and both decimals non-negative. Then out equals the
// changes the generic path applies. Snapshots, subscription acks, another
// product, the bid/ask spellings side_of also takes and four-element changes
// all return false. tests/test_noalloc.cpp checks every accepted frame against
// the generic path, on generated frames, mutated bytes and the live recordings.
//
// Worked example: {"type":"l2update","product_id":"X","changes":[["buy","1.5","2"]]}
// with product "X" gives one change (Bid, 150000000, 200000000).
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "qp/ull/decimal.hpp"
#include "qp/ull/l2book.hpp"

namespace qp::ull::feed {

struct LevelChange {
    Side side;
    std::int64_t px;
    std::int64_t qty;
    bool operator==(const LevelChange&) const = default;
};

// One diff-depth event: update ids [first, last] and absolute level sizes.
struct DepthDelta {
    std::int64_t first = 0, last = 0;
    std::vector<LevelChange> changes;
    bool operator==(const DepthDelta&) const = default;
};

namespace detail {

class DepthCursor {
public:
    explicit DepthCursor(std::string_view b) : b_(b) {}
    std::size_t i = 0;

    void ws() {
        while (i < b_.size() && (b_[i] == ' ' || b_[i] == '\t' || b_[i] == '\n' || b_[i] == '\r'))
            ++i;
    }
    bool at_end() {
        ws();
        return i == b_.size();
    }
    int peek() const { return i < b_.size() ? static_cast<unsigned char>(b_[i]) : -1; }
    bool eat(char c) {
        ws();
        if (peek() != static_cast<unsigned char>(c)) return false;
        ++i;
        return true;
    }
    // Printable ASCII, no backslash.
    bool str(std::string_view& out) {
        ws();
        if (peek() != '"') return false;
        const std::size_t s = ++i;
        for (;;) {
            const int c = peek();
            if (c == '"') {
                out = b_.substr(s, i - s);
                ++i;
                return true;
            }
            if (c < 0x20 || c >= 0x7f || c == '\\') return false;
            ++i;
        }
    }
    bool num(std::string_view& out) {
        const std::size_t s = i;
        if (peek() == '-') ++i;
        if (peek() == '0') {
            ++i;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++i;
        } else {
            return false;
        }
        if (peek() == '.') {
            ++i;
            if (!(peek() >= '0' && peek() <= '9')) return false;
            while (peek() >= '0' && peek() <= '9') ++i;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++i;
            if (peek() == '+' || peek() == '-') ++i;
            if (!(peek() >= '0' && peek() <= '9')) return false;
            while (peek() >= '0' && peek() <= '9') ++i;
        }
        out = b_.substr(s, i - s);
        return true;
    }
    bool lit(std::string_view w) {
        if (b_.substr(i, w.size()) != w) return false;
        i += w.size();
        return true;
    }
    // Validate and skip one value; stricter depth than json.hpp (64).
    bool skip(int depth) {
        if (depth > 32) return false;
        ws();
        std::string_view t;
        switch (peek()) {
            case '"':
                return str(t);
            case '{':
                ++i;
                if (eat('}')) return true;
                for (;;) {
                    if (!str(t) || !eat(':') || !skip(depth + 1)) return false;
                    if (eat(',')) continue;
                    return eat('}');
                }
            case '[':
                ++i;
                if (eat(']')) return true;
                for (;;) {
                    if (!skip(depth + 1)) return false;
                    if (eat(',')) continue;
                    return eat(']');
                }
            case 't':
                return lit("true");
            case 'f':
                return lit("false");
            case 'n':
                return lit("null");
            default:
                return num(t);
        }
    }
    bool i64(std::int64_t& v) {
        ws();
        std::string_view t;
        if (!num(t) || t.empty()) return false;
        std::uint64_t x = 0;
        for (char c : t) {
            if (c < '0' || c > '9') return false;
            if (x > (static_cast<std::uint64_t>(INT64_MAX) - (c - '0')) / 10) return false;
            x = x * 10 + static_cast<std::uint64_t>(c - '0');
        }
        v = static_cast<std::int64_t>(x);
        return true;
    }
    bool levels(Side s, std::vector<LevelChange>& out) {
        if (!eat('[')) return false;
        if (eat(']')) return true;
        for (;;) {
            std::string_view p, q;
            if (!eat('[') || !str(p) || !eat(',') || !str(q) || !eat(']')) return false;
            if (out.size() == out.capacity()) return false;  // would allocate
            std::int64_t px = 0, qty = 0;
            try {
                px = parse_fixed(p);
                qty = parse_fixed(q);
            } catch (...) {
                return false;
            }
            out.push_back({s, px, qty});
            if (eat(',')) continue;
            return eat(']');
        }
    }
    // [["buy","px","qty"], ...]. A negative value returns false: L2Book::set
    // throws on it, and the generic path owns that error.
    bool changes(std::vector<LevelChange>& out) {
        if (!eat('[')) return false;
        if (eat(']')) return true;
        for (;;) {
            std::string_view sd, p, q;
            if (!eat('[') || !str(sd) || !eat(',') || !str(p) || !eat(',') || !str(q) || !eat(']'))
                return false;
            if (sd != "buy" && sd != "sell") return false;
            if (out.size() == out.capacity()) return false;  // would allocate
            std::int64_t px = 0, qty = 0;
            try {
                px = parse_fixed(p);
                qty = parse_fixed(q);
            } catch (...) {
                return false;
            }
            if (px < 0 || qty < 0) return false;
            out.push_back({sd == "buy" ? Side::Bid : Side::Ask, px, qty});
            if (eat(',')) continue;
            return eat(']');
        }
    }

private:
    std::string_view b_;
};

inline bool depth_event(DepthCursor& c, std::string_view sym, DepthDelta& d) {
    if (!c.eat('{')) return false;
    bool e = false, s = false, U = false, u = false, b = false, a = false;
    if (c.eat('}')) return false;
    for (;;) {
        std::string_view k;
        if (!c.str(k) || !c.eat(':')) return false;
        if (k == "e" && !e) {
            std::string_view v;
            if (!c.str(v) || v != "depthUpdate") return false;
            e = true;
        } else if (k == "s" && !s) {
            std::string_view v;
            if (!c.str(v) || v != sym) return false;
            s = true;
        } else if (k == "U" && !U) {
            if (!c.i64(d.first)) return false;
            U = true;
        } else if (k == "u" && !u) {
            if (!c.i64(d.last)) return false;
            u = true;
        } else if (k == "b" && !b && !a) {
            if (!c.levels(Side::Bid, d.changes)) return false;
            b = true;
        } else if (k == "a" && !a && b) {
            if (!c.levels(Side::Ask, d.changes)) return false;
            a = true;
        } else if (k == "e" || k == "s" || k == "U" || k == "u" || k == "b" || k == "a" ||
                   k == "data" || k == "stream" || k == "lastUpdateId") {
            return false;
        } else if (!c.skip(1)) {
            return false;
        }
        if (c.eat(',')) continue;
        if (!c.eat('}')) return false;
        break;
    }
    return e && s && U && u && b && a && d.first <= d.last;
}

inline bool level2_event(DepthCursor& c, std::string_view product, std::vector<LevelChange>& out) {
    if (!c.eat('{')) return false;
    bool type = false, pid = false, ch = false;
    if (c.eat('}')) return false;
    for (;;) {
        std::string_view k;
        if (!c.str(k) || !c.eat(':')) return false;
        if (k == "type" && !type) {
            std::string_view v;
            if (!c.str(v) || v != "l2update") return false;
            type = true;
        } else if (k == "product_id" && !pid) {
            std::string_view v;
            if (!c.str(v) || v != product) return false;
            pid = true;
        } else if (k == "changes" && !ch) {
            if (!c.changes(out)) return false;
            ch = true;
        } else if (k == "type" || k == "product_id" || k == "changes") {
            return false;
        } else if (!c.skip(1)) {
            return false;
        }
        if (c.eat(',')) continue;
        if (!c.eat('}')) return false;
        break;
    }
    return type && pid && ch;
}

}  // namespace detail

// Decode one Binance diff-depth frame into `d` without allocating. Returns
// false (and leaves d.changes empty) when the generic path must decide.
inline bool decode_depth(std::string_view raw, std::string_view sym, DepthDelta& d) {
    d.changes.clear();
    detail::DepthCursor c(raw);
    bool ok = false;
    const std::size_t start = (c.ws(), c.i);
    // Combined-stream wrapper, exactly {"stream": "...", "data": {...}}.
    std::string_view k, v;
    if (c.eat('{') && c.str(k) && k == "stream" && c.eat(':') && c.str(v) && c.eat(',') &&
        c.str(k) && k == "data" && c.eat(':')) {
        ok = detail::depth_event(c, sym, d) && c.eat('}') && c.at_end();
    } else {
        d.changes.clear();
        c.i = start;
        ok = detail::depth_event(c, sym, d) && c.at_end();
    }
    if (!ok) d.changes.clear();
    return ok;
}

// Decode one Coinbase l2update frame into `out` without allocating. Returns
// false (and leaves `out` empty) when the generic path must decide.
inline bool decode_level2(std::string_view raw, std::string_view product,
                          std::vector<LevelChange>& out) {
    out.clear();
    detail::DepthCursor c(raw);
    const bool ok = detail::level2_event(c, product, out) && c.at_end();
    if (!ok) out.clear();
    return ok;
}

}  // namespace qp::ull::feed
