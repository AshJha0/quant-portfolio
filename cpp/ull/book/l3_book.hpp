// Order-by-order (L3) book on flat price-indexed arrays.
//
// Plain idea: prices are integer ticks inside a fixed band, so a price level
// is just an array slot, `price - base`. Each slot holds the level's total
// quantity, its order count and the head and tail of an intrusive FIFO list
// of orders. Orders live in a preallocated pool, and an open-addressing hash
// table maps order id to pool index. After construction nothing allocates:
// add, modify, cancel and execute touch O(1) slots, except when the best
// level empties and the book scans outward for the next non-empty level
// (bounded by the band; in practice a few slots).
//
// Formal statement: after any accepted message sequence, for each side s and
// price p, level(s, p).qty = sum of qty over live orders at (s, p), level
// count = number of such orders, and the FIFO order equals arrival order,
// where a modify that keeps the price and does not increase qty keeps its
// place and any other modify moves the order to the tail (QPBF.md section 5).
//
// Worked example: band base = 100, 8 levels. add(1, bid, 103, 10),
// add(2, bid, 103, 5), add(3, ask, 105, 7): best bid 103 qty 15 count 2,
// best ask 105. cancel(1, 10) leaves 103 qty 5. execute(2, 5) empties 103,
// best bid becomes none.
#pragma once

#include "../qpbf/qpbf.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ull::book {

using qpbf::Msg;
using qpbf::MsgType;
using qpbf::Side;

enum class Result : std::uint8_t {
    Ok = 0,
    Ignored,        // trade/heartbeat: no book change
    DuplicateId,
    UnknownId,
    PriceOutOfBand,
    QtyTooLarge,    // cancel/execute more than the order has
    PoolFull,
    BadSide,
};

struct Level {
    std::uint64_t qty{0};
    std::uint32_t count{0};
    std::uint32_t head{kNil};
    std::uint32_t tail{kNil};
    static constexpr std::uint32_t kNil = 0xFFFFFFFFu;
};

class L3Book {
public:
    static constexpr std::uint32_t kNil = Level::kNil;
    static constexpr std::int64_t kNoPrice = INT64_MIN;

    L3Book(std::int64_t base_price, std::uint32_t levels, std::uint32_t max_orders)
        : base_(base_price), nlev_(levels), max_orders_(max_orders),
          bids_(std::make_unique<Level[]>(levels)),
          asks_(std::make_unique<Level[]>(levels)),
          pool_(std::make_unique<Order[]>(max_orders)) {
        if (levels == 0 || max_orders == 0)
            throw std::invalid_argument("L3Book: levels and max_orders must be > 0");
        hcap_ = 1;
        while (hcap_ < 2ull * max_orders) hcap_ <<= 1;
        hkeys_ = std::make_unique<std::uint64_t[]>(hcap_);
        hvals_ = std::make_unique<std::uint32_t[]>(hcap_);
        clear();
    }

    void clear() noexcept {
        for (std::uint32_t i = 0; i < nlev_; ++i) bids_[i] = Level{};
        for (std::uint32_t i = 0; i < nlev_; ++i) asks_[i] = Level{};
        for (std::uint64_t i = 0; i < hcap_; ++i) hkeys_[i] = 0;
        for (std::uint32_t i = 0; i < max_orders_; ++i) pool_[i].next = i + 1;
        if (max_orders_ > 0) pool_[max_orders_ - 1].next = kNil;
        free_ = 0;
        live_ = 0;
        best_bid_ = -1;
        best_ask_ = static_cast<std::int64_t>(nlev_);
    }

    // Apply one decoded message. Snapshot orders are applied as adds.
    Result apply(const Msg& m) noexcept {
        switch (m.type) {
            case MsgType::Add:
            case MsgType::Snapshot:
                if (m.type == MsgType::Snapshot && m.order_id == 0) return Result::Ignored;
                return add(m.order_id, m.side, m.price, m.qty);
            case MsgType::Modify: return modify(m.order_id, m.price, m.qty);
            case MsgType::Cancel:
            case MsgType::Execute: return reduce(m.order_id, m.qty);
            case MsgType::Trade:
            case MsgType::Heartbeat: return Result::Ignored;
        }
        return Result::Ignored;
    }

    Result add(std::uint64_t id, Side side, std::int64_t price, std::uint32_t qty) noexcept {
        if (side != Side::Bid && side != Side::Ask) return Result::BadSide;
        const std::int64_t li = price - base_;
        if (li < 0 || li >= static_cast<std::int64_t>(nlev_)) return Result::PriceOutOfBand;
        if (find(id) != kNil) return Result::DuplicateId;
        if (free_ == kNil) return Result::PoolFull;
        const std::uint32_t oi = free_;
        free_ = pool_[oi].next;
        Order& o = pool_[oi];
        o.id = id;
        o.lvl = static_cast<std::uint32_t>(li);
        o.qty = qty;
        o.side = side;
        insert_id(id, oi);
        link_tail(oi);
        ++live_;
        return Result::Ok;
    }

    Result modify(std::uint64_t id, std::int64_t price, std::uint32_t qty) noexcept {
        const std::uint32_t oi = find(id);
        if (oi == kNil) return Result::UnknownId;
        Order& o = pool_[oi];
        const std::int64_t li = price - base_;
        if (li < 0 || li >= static_cast<std::int64_t>(nlev_)) return Result::PriceOutOfBand;
        Level& L = level(o.side, o.lvl);
        if (static_cast<std::uint32_t>(li) == o.lvl && qty <= o.qty) {
            L.qty -= (o.qty - qty);  // keeps queue position
            o.qty = qty;
            return Result::Ok;
        }
        unlink(oi);
        o.lvl = static_cast<std::uint32_t>(li);
        o.qty = qty;
        link_tail(oi);
        return Result::Ok;
    }

    // Cancel or execute `qty` of an order; removes it when it reaches zero.
    Result reduce(std::uint64_t id, std::uint32_t qty) noexcept {
        const std::uint32_t oi = find(id);
        if (oi == kNil) return Result::UnknownId;
        Order& o = pool_[oi];
        if (qty > o.qty) return Result::QtyTooLarge;
        if (qty == o.qty) {
            unlink(oi);
            erase_id(id);
            o.next = free_;
            free_ = oi;
            --live_;
            return Result::Ok;
        }
        o.qty -= qty;
        level(o.side, o.lvl).qty -= qty;
        return Result::Ok;
    }

    // ---- queries -------------------------------------------------------
    [[nodiscard]] std::uint32_t order_count() const noexcept { return live_; }
    [[nodiscard]] std::int64_t base_price() const noexcept { return base_; }
    [[nodiscard]] std::uint32_t levels() const noexcept { return nlev_; }
    [[nodiscard]] std::int64_t best_bid() const noexcept {
        return best_bid_ >= 0 ? base_ + best_bid_ : kNoPrice;
    }
    [[nodiscard]] std::int64_t best_ask() const noexcept {
        return best_ask_ < static_cast<std::int64_t>(nlev_) ? base_ + best_ask_ : kNoPrice;
    }
    [[nodiscard]] const Level& level_at(Side s, std::int64_t price) const noexcept {
        static const Level empty{};
        const std::int64_t li = price - base_;
        if (li < 0 || li >= static_cast<std::int64_t>(nlev_)) return empty;
        return s == Side::Bid ? bids_[li] : asks_[li];
    }
    // Iterate orders at a level in FIFO order: f(id, qty).
    template <class F>
    void for_each_order(Side s, std::int64_t price, F&& f) const {
        const Level& L = level_at(s, price);
        for (std::uint32_t i = L.head; i != kNil; i = pool_[i].next) f(pool_[i].id, pool_[i].qty);
    }
    // Look up an order: returns false when unknown.
    bool get_order(std::uint64_t id, Side& s, std::int64_t& price, std::uint32_t& qty) const noexcept {
        const std::uint32_t oi = find(id);
        if (oi == kNil) return false;
        s = pool_[oi].side;
        price = base_ + pool_[oi].lvl;
        qty = pool_[oi].qty;
        return true;
    }
    // Visit every live order as a snapshot message body (any order).
    template <class F>
    void for_each_live(F&& f) const {
        for (std::uint32_t li = 0; li < nlev_; ++li) {
            for (Side s : {Side::Bid, Side::Ask}) {
                const Level& L = s == Side::Bid ? bids_[li] : asks_[li];
                for (std::uint32_t i = L.head; i != kNil; i = pool_[i].next)
                    f(pool_[i].id, s, base_ + li, pool_[i].qty);
            }
        }
    }

private:
    struct Order {
        std::uint64_t id{0};
        std::uint32_t lvl{0};
        std::uint32_t qty{0};
        std::uint32_t prev{kNil};
        std::uint32_t next{kNil};
        Side side{Side::None};
    };

    Level& level(Side s, std::uint32_t li) noexcept {
        return s == Side::Bid ? bids_[li] : asks_[li];
    }

    void link_tail(std::uint32_t oi) noexcept {
        Order& o = pool_[oi];
        Level& L = level(o.side, o.lvl);
        o.prev = L.tail;
        o.next = kNil;
        if (L.tail != kNil) pool_[L.tail].next = oi; else L.head = oi;
        L.tail = oi;
        L.qty += o.qty;
        ++L.count;
        const auto li = static_cast<std::int64_t>(o.lvl);
        if (o.side == Side::Bid) { if (li > best_bid_) best_bid_ = li; }
        else if (li < best_ask_) best_ask_ = li;
    }

    void unlink(std::uint32_t oi) noexcept {
        Order& o = pool_[oi];
        Level& L = level(o.side, o.lvl);
        if (o.prev != kNil) pool_[o.prev].next = o.next; else L.head = o.next;
        if (o.next != kNil) pool_[o.next].prev = o.prev; else L.tail = o.prev;
        L.qty -= o.qty;
        --L.count;
        if (L.count == 0) {
            const auto li = static_cast<std::int64_t>(o.lvl);
            if (o.side == Side::Bid && li == best_bid_) {
                while (best_bid_ >= 0 && bids_[best_bid_].count == 0) --best_bid_;
            } else if (o.side == Side::Ask && li == best_ask_) {
                const auto n = static_cast<std::int64_t>(nlev_);
                while (best_ask_ < n && asks_[best_ask_].count == 0) ++best_ask_;
            }
        }
    }

    // Open addressing, linear probing, backward-shift deletion (no
    // tombstones). Key 0 marks an empty slot; QPBF forbids order id 0.
    static std::uint64_t mix(std::uint64_t k) noexcept {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdull;
        k ^= k >> 33;
        return k;
    }
    std::uint32_t find(std::uint64_t id) const noexcept {
        if (id == 0) return kNil;
        const std::uint64_t mask = hcap_ - 1;
        for (std::uint64_t i = mix(id) & mask;; i = (i + 1) & mask) {
            if (hkeys_[i] == id) return hvals_[i];
            if (hkeys_[i] == 0) return kNil;
        }
    }
    void insert_id(std::uint64_t id, std::uint32_t v) noexcept {
        const std::uint64_t mask = hcap_ - 1;
        std::uint64_t i = mix(id) & mask;
        while (hkeys_[i] != 0) i = (i + 1) & mask;
        hkeys_[i] = id;
        hvals_[i] = v;
    }
    void erase_id(std::uint64_t id) noexcept {
        const std::uint64_t mask = hcap_ - 1;
        std::uint64_t i = mix(id) & mask;
        while (hkeys_[i] != id) i = (i + 1) & mask;
        std::uint64_t j = i;
        for (;;) {
            j = (j + 1) & mask;
            if (hkeys_[j] == 0) break;
            const std::uint64_t h = mix(hkeys_[j]) & mask;
            // Move j back to the hole at i unless h lies cyclically in (i, j].
            const bool keep = (i <= j) ? (i < h && h <= j) : (i < h || h <= j);
            if (!keep) {
                hkeys_[i] = hkeys_[j];
                hvals_[i] = hvals_[j];
                i = j;
            }
        }
        hkeys_[i] = 0;
    }

    std::int64_t base_;
    std::uint32_t nlev_;
    std::uint32_t max_orders_;
    std::unique_ptr<Level[]> bids_;
    std::unique_ptr<Level[]> asks_;
    std::unique_ptr<Order[]> pool_;
    std::unique_ptr<std::uint64_t[]> hkeys_;
    std::unique_ptr<std::uint32_t[]> hvals_;
    std::uint64_t hcap_{0};
    std::uint32_t free_{kNil};
    std::uint32_t live_{0};
    std::int64_t best_bid_{-1};
    std::int64_t best_ask_{0};
};

// One book per instrument id in [0, n).
class BookSet {
public:
    BookSet(std::uint32_t n, std::int64_t base, std::uint32_t levels, std::uint32_t max_orders) {
        books_.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) books_.emplace_back(std::make_unique<L3Book>(base, levels, max_orders));
    }
    Result apply(const Msg& m) noexcept {
        if (m.type == MsgType::Heartbeat || m.type == MsgType::Trade) return Result::Ignored;
        if (m.instrument >= books_.size()) return Result::UnknownId;
        return books_[m.instrument]->apply(m);
    }
    void clear() noexcept { for (auto& b : books_) b->clear(); }
    L3Book& operator[](std::uint32_t i) noexcept { return *books_[i]; }
    const L3Book& operator[](std::uint32_t i) const noexcept { return *books_[i]; }
    [[nodiscard]] std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(books_.size()); }

private:
    std::vector<std::unique_ptr<L3Book>> books_;
};

}  // namespace ull::book
