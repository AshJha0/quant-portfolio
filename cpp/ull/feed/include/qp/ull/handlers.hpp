// Venue handlers: Coinbase Exchange `full` and `level2`, Binance spot
// diff-depth. Each turns raw websocket text (from line A or line B) and REST
// snapshot bodies into a book plus a stale flag.
//
// Message formats follow the public, no-auth documentation:
//  - Coinbase Exchange websocket feed, channels `full` (types received, open,
//    done, match, change, activate; every message has `sequence`) and
//    `level2` (`snapshot`, then `l2update` with absolute sizes); REST
//    GET /products/<id>/book?level=3 -> {"sequence", "bids":[[px,size,id]], ...}.
//  - Binance spot websocket <symbol>@depth / @depth@100ms ("e":"depthUpdate",
//    first id "U", final id "u", "b"/"a" absolute sizes) and REST
//    GET /api/v3/depth -> {"lastUpdateId", "bids":[[px,qty]], "asks":...}.
// Field names are taken from those documents; the tests use fixtures in the
// same shape (tests/fixtures/README.md says which are synthetic).
//
// The handlers never open sockets. The caller owns transport, which keeps
// the tests offline and lets the recorder sit in front of the parser.
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "qp/ull/json.hpp"
#include "qp/ull/l2book.hpp"
#include "qp/ull/seqsync.hpp"

namespace qp::ull::feed {

inline Side side_of(const std::string& s) {
    if (s == "buy" || s == "bid") return Side::Bid;
    if (s == "sell" || s == "ask") return Side::Ask;
    throw std::invalid_argument("bad side " + s);
}

struct HandlerStats {
    std::uint64_t messages = 0, ignored = 0, malformed = 0, inconsistent = 0;
};

// ---- Coinbase full (level 3, sequenced) --------------------------------------

class CoinbaseFull {
public:
    CoinbaseFull(std::string product, SyncConfig cfg = {})
        : product_(std::move(product)), sync_(cfg, [this](const json::Value& m) { apply(m); }) {
        sync_.on_snapshot_needed = [this] {
            if (on_snapshot_needed) on_snapshot_needed();
        };
        sync_.on_stale = [this](bool s) {
            if (on_stale) on_stale(s);
        };
    }

    std::function<void()> on_snapshot_needed;
    std::function<void(bool)> on_stale;

    // Raw websocket text from either line. Malformed input is counted, never
    // thrown: a feed handler must survive a bad frame.
    void on_message(std::string_view raw, std::int64_t now_ns) {
        ++hs_.messages;
        json::Value m;
        try {
            m = json::parse(raw);
            if (!m.is_object() || !m.has("type")) {
                ++hs_.ignored;
                return;
            }
            const std::string& t = m.at("type").str();
            if (t == "heartbeat" || t == "subscriptions" || !m.has("sequence") ||
                (m.has("product_id") && m.at("product_id").str() != product_)) {
                ++hs_.ignored;
                sync_.tick(now_ns);
                return;
            }
            std::int64_t seq = m.at("sequence").i64();
            try {
                sync_.on_update(seq, seq, std::move(m), now_ns);
            } catch (const std::exception&) {
                // The frame parsed but could not be applied (e.g. a match larger than
                // the resting order): the book no longer matches the venue. Resync.
                ++hs_.inconsistent;
                sync_.reset();
            }
        } catch (const std::exception&) {
            ++hs_.malformed;
        }
    }

    // REST level-3 snapshot body.
    void on_snapshot(std::string_view raw) {
        json::Value s = json::parse(raw);
        std::int64_t seq = s.at("sequence").i64();
        sync_.on_snapshot(seq, [&] {
            book_.clear();
            for (const auto& r : s.at("bids").arr)
                book_.open(r.at(2).str(), Side::Bid, parse_fixed(r.at(0).str()),
                           parse_fixed(r.at(1).str()));
            for (const auto& r : s.at("asks").arr)
                book_.open(r.at(2).str(), Side::Ask, parse_fixed(r.at(0).str()),
                           parse_fixed(r.at(1).str()));
        });
    }

    void tick(std::int64_t now_ns) { sync_.tick(now_ns); }
    void reset() { sync_.reset(); }

    bool stale() const { return sync_.stale(); }
    const L3Book& book() const { return book_; }
    const SeqSync<json::Value>& sync() const { return sync_; }
    const HandlerStats& stats() const { return hs_; }

private:
    std::string product_;
    L3Book book_;
    SeqSync<json::Value> sync_;
    HandlerStats hs_;

    void apply(const json::Value& m) {
        const std::string& t = m.at("type").str();
        if (t == "open") {
            book_.open(m.at("order_id").str(), side_of(m.at("side").str()),
                       parse_fixed(m.at("price").str()), parse_fixed(m.at("remaining_size").str()));
        } else if (t == "done") {
            book_.done(m.at("order_id").str());
        } else if (t == "match") {
            book_.reduce(m.at("maker_order_id").str(), parse_fixed(m.at("size").str()));
        } else if (t == "change") {
            if (m.has("new_size"))
                book_.resize(m.at("order_id").str(), parse_fixed(m.at("new_size").str()));
        }
        // received, activate: consume a sequence number, no resting-book change.
    }
};

// ---- Binance diff-depth (level 2, update-id ranges) --------------------------

class BinanceDepth {
public:
    BinanceDepth(std::string symbol, SyncConfig cfg = {})
        : symbol_(std::move(symbol)), sync_(cfg, [this](const json::Value& m) { apply(m); }) {
        sync_.on_snapshot_needed = [this] {
            if (on_snapshot_needed) on_snapshot_needed();
        };
        sync_.on_stale = [this](bool s) {
            if (on_stale) on_stale(s);
        };
    }

    std::function<void()> on_snapshot_needed;
    std::function<void(bool)> on_stale;

    void on_message(std::string_view raw, std::int64_t now_ns) {
        ++hs_.messages;
        try {
            json::Value m = json::parse(raw);
            if (m.has("data") && m.has("stream"))
                m = json::Value(m.at("data"));  // combined-stream wrapper
            if (!m.has("e") || m.at("e").str() != "depthUpdate" || m.at("s").str() != symbol_) {
                ++hs_.ignored;
                sync_.tick(now_ns);
                return;
            }
            std::int64_t first = m.at("U").i64(), last = m.at("u").i64();
            if (first > last) {
                ++hs_.malformed;
                return;
            }
            try {
                sync_.on_update(first, last, std::move(m), now_ns);
            } catch (const std::exception&) {
                ++hs_.inconsistent;
                sync_.reset();
            }
        } catch (const std::exception&) {
            ++hs_.malformed;
        }
    }

    void on_snapshot(std::string_view raw) {
        json::Value s = json::parse(raw);
        sync_.on_snapshot(s.at("lastUpdateId").i64(), [&] {
            book_.clear();
            for (const auto& r : s.at("bids").arr)
                book_.set(Side::Bid, parse_fixed(r.at(0).str()), parse_fixed(r.at(1).str()));
            for (const auto& r : s.at("asks").arr)
                book_.set(Side::Ask, parse_fixed(r.at(0).str()), parse_fixed(r.at(1).str()));
        });
    }

    void tick(std::int64_t now_ns) { sync_.tick(now_ns); }
    void reset() { sync_.reset(); }
    bool stale() const { return sync_.stale(); }
    const L2Book& book() const { return book_; }
    const SeqSync<json::Value>& sync() const { return sync_; }
    const HandlerStats& stats() const { return hs_; }

private:
    std::string symbol_;
    L2Book book_;
    SeqSync<json::Value> sync_;
    HandlerStats hs_;

    void apply(const json::Value& m) {
        for (const auto& r : m.at("b").arr)
            book_.set(Side::Bid, parse_fixed(r.at(0).str()), parse_fixed(r.at(1).str()));
        for (const auto& r : m.at("a").arr)
            book_.set(Side::Ask, parse_fixed(r.at(0).str()), parse_fixed(r.at(1).str()));
    }
};

// ---- Coinbase level2 (unsequenced, absolute sizes) ----------------------------
//
// The level2 channel carries no sequence number, so a dropped update cannot
// be detected from the stream itself. What we can do: (1) treat the book as
// stale until the in-band snapshot arrives and after every reset; (2) A/B
// arbitrate by exact payload, dropping a frame already seen in the last
// `dedupe_window` frames. Because l2update sizes are absolute, a duplicate
// applied twice is harmless; a reordered pair of updates to the same level is
// not, and is undetectable here. Use `full` where gap-safety matters.

class CoinbaseLevel2 {
public:
    explicit CoinbaseLevel2(std::string product, std::size_t dedupe_window = 4096)
        : product_(std::move(product)), window_(dedupe_window) {}

    std::function<void(bool)> on_stale;

    void on_message(std::string_view raw) {
        ++hs_.messages;
        std::size_t h = std::hash<std::string_view>{}(raw);
        if (seen_.count(h)) {
            ++dups_;
            return;
        }
        seen_.insert(h);
        order_.push_back(h);
        if (order_.size() > window_) {
            seen_.erase(order_.front());
            order_.pop_front();
        }
        try {
            json::Value m = json::parse(raw);
            if (!m.has("type") || (m.has("product_id") && m.at("product_id").str() != product_)) {
                ++hs_.ignored;
                return;
            }
            const std::string& t = m.at("type").str();
            if (t == "snapshot") {
                book_.clear();
                for (const auto& r : m.at("bids").arr)
                    book_.set(Side::Bid, parse_fixed(r.at(0).str()), parse_fixed(r.at(1).str()));
                for (const auto& r : m.at("asks").arr)
                    book_.set(Side::Ask, parse_fixed(r.at(0).str()), parse_fixed(r.at(1).str()));
                set_stale(false);
            } else if (t == "l2update") {
                if (stale_) {
                    ++hs_.ignored;
                    return;
                }
                for (const auto& c : m.at("changes").arr)
                    book_.set(side_of(c.at(0).str()), parse_fixed(c.at(1).str()),
                              parse_fixed(c.at(2).str()));
            } else {
                ++hs_.ignored;
            }
        } catch (const std::exception&) {
            ++hs_.malformed;
        }
    }
    void reset() { set_stale(true); }
    bool stale() const { return stale_; }
    const L2Book& book() const { return book_; }
    std::uint64_t duplicates() const { return dups_; }
    const HandlerStats& stats() const { return hs_; }

private:
    std::string product_;
    std::size_t window_;
    L2Book book_;
    bool stale_ = true;
    std::unordered_set<std::size_t> seen_;
    std::deque<std::size_t> order_;
    std::uint64_t dups_ = 0;
    HandlerStats hs_;
    void set_stale(bool s) {
        if (s != stale_) {
            stale_ = s;
            if (on_stale) on_stale(s);
        }
    }
};

}  // namespace qp::ull::feed
