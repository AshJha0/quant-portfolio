// Crypto books on the part-1 flat-array L3 book (cpp/ull/book/l3_book.hpp).
//
// Plain idea: the part-1 book keeps one array slot per integer tick inside a
// fixed band, a preallocated order pool and an open-addressing id table, so
// nothing allocates after construction. Crypto feeds send decimal prices,
// decimal sizes and (Coinbase `full`) string order ids, so three small
// adapters sit in front of it:
//   - a per-product ProductSpec maps a fixed-point price to a tick
//     (px / tick) and a fixed-point size to whole lots (qty / lot);
//   - the band (the "window") is centred on the touch whenever a snapshot is
//     loaded; an update outside it throws OutOfWindow, which the handlers
//     treat like any inconsistency: the book goes stale and resynchronises
//     from a snapshot, and that snapshot re-centres the window;
//   - Coinbase order ids are interned into a fixed-capacity table of
//     fixed-length keys (at most kMaxId bytes; UUIDs are 36), giving each id a
//     u64 handle for the flat book.
// A level feed (Binance diff-depth, Coinbase level2) stores each level as one
// synthetic order with id 2 * tick + side + 1.
//
// Formal statement: inside the window, FlatL2Book holds exactly the levels
// L2Book would hold after the same calls, and FlatL3Book's levels are the
// per-price sums of its live orders. Snapshot entries outside the window are
// dropped and counted in clipped() (SPEC CHOICE: size the window to cover
// the snapshot depth you request). set/open/done/reduce/resize never allocate
// on success (tests/test_noalloc.cpp counts operator new).
//
// Worked example: tick 0.01, window 64 ticks. A snapshot with best bid
// 100.00 and best ask 100.02 centres the window on tick 10001, covering
// [99.69, 100.33). set(Ask, 100.05, 2.0) stores 200000000 lots at tick
// 10005; set(Ask, 101.00, 1.0) throws OutOfWindow.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../../../../book/l3_book.hpp"
#include "qp/ull/decimal.hpp"
#include "qp/ull/l2book.hpp"

namespace qp::ull::feed {

struct ProductSpec {
    std::int64_t tick = kScale / 100;  // 0.01
    std::int64_t lot = 1;              // 1e-8: up to ~42.9 units per order or level
    std::uint32_t n_levels = 1u << 14;  // +-81.92 around the touch at a 0.01 tick
    std::uint32_t max_orders = 1u << 14;
};

// Thrown for a price off the tick grid or outside the window. Its what() is a
// string literal, so constructing it does not format anything.
struct OutOfWindow : std::runtime_error {
    OutOfWindow() : std::runtime_error("price outside the product window") {}
};
struct Unrepresentable : std::runtime_error {
    Unrepresentable() : std::runtime_error("size not a whole number of lots that fits u32") {}
};
struct BookFull : std::runtime_error {
    BookFull() : std::runtime_error("book order pool or id table full") {}
};

namespace detail {

inline ::ull::qpbf::Side qside(Side s) {
    return s == Side::Bid ? ::ull::qpbf::Side::Bid : ::ull::qpbf::Side::Ask;
}

// Window, tick/lot conversion and the flat book, shared by both adapters.
class FlatBase {
public:
    explicit FlatBase(ProductSpec spec)
        : spec_(spec), book_(std::make_unique<::ull::book::L3Book>(0, spec.n_levels, spec.max_orders)) {
        if (spec.tick <= 0 || spec.lot <= 0 || spec.n_levels < 4)
            throw std::invalid_argument("ProductSpec: tick, lot > 0 and n_levels >= 4");
    }
    const ProductSpec& spec() const { return spec_; }
    // [lo, hi) in fixed point.
    std::pair<std::int64_t, std::int64_t> window() const {
        const std::int64_t b = book_->base_price();
        return {b * spec_.tick, (b + spec_.n_levels) * spec_.tick};
    }
    std::uint64_t clipped() const { return clipped_; }

protected:
    ProductSpec spec_;
    std::unique_ptr<::ull::book::L3Book> book_;
    std::uint64_t clipped_ = 0;

    std::int64_t tick_of(std::int64_t px) const {
        if (px < 0 || px % spec_.tick != 0) throw OutOfWindow();
        return px / spec_.tick;
    }
    std::uint32_t lots_of(std::int64_t qty) const {
        if (qty < 0 || qty % spec_.lot != 0 || qty / spec_.lot > 0xFFFFFFFFll) throw Unrepresentable();
        return static_cast<std::uint32_t>(qty / spec_.lot);
    }
    // Clear, moving the window when the touch `centre` (ticks) has left its
    // middle half. Rebuilding allocates; it happens on the snapshot path only.
    void recentre(std::int64_t centre, bool have) {
        const std::int64_t n = spec_.n_levels, b = book_->base_price();
        if (have && (centre < b + n / 4 || centre >= b + 3 * n / 4))
            book_ = std::make_unique<::ull::book::L3Book>(std::max<std::int64_t>(0, centre - n / 2),
                                                          spec_.n_levels, spec_.max_orders);
        else
            book_->clear();
    }
    static void check(::ull::book::Result r) {
        using ::ull::book::Result;
        switch (r) {
            case Result::Ok:
                return;
            case Result::PriceOutOfBand:
                throw OutOfWindow();
            case Result::PoolFull:
                throw BookFull();
            default:
                throw std::logic_error("flat book rejected an update");
        }
    }
    // Levels of one side, best first, in fixed point (reporting; allocates).
    std::vector<Level> scan(Side s) const {
        std::vector<Level> o;
        const auto qs = qside(s);
        const std::int64_t lo = book_->base_price(), hi = lo + spec_.n_levels;
        if (s == Side::Bid) {
            const std::int64_t bb = book_->best_bid();
            if (bb == ::ull::book::L3Book::kNoPrice) return o;
            for (std::int64_t t = bb; t >= lo; --t) {
                const auto& L = book_->level_at(qs, t);
                if (L.count) o.push_back({t * spec_.tick, static_cast<std::int64_t>(L.qty) * spec_.lot});
            }
        } else {
            const std::int64_t ba = book_->best_ask();
            if (ba == ::ull::book::L3Book::kNoPrice) return o;
            for (std::int64_t t = ba; t < hi; ++t) {
                const auto& L = book_->level_at(qs, t);
                if (L.count) o.push_back({t * spec_.tick, static_cast<std::int64_t>(L.qty) * spec_.lot});
            }
        }
        return o;
    }
    L2Book to_l2_impl() const {
        L2Book b;
        for (const auto& l : scan(Side::Bid)) b.set(Side::Bid, l.px, l.qty);
        for (const auto& l : scan(Side::Ask)) b.set(Side::Ask, l.px, l.qty);
        return b;
    }
};

}  // namespace detail

// Level book (absolute sizes per price) on the flat book.
class FlatL2Book : public detail::FlatBase {
public:
    explicit FlatL2Book(ProductSpec spec = {}) : FlatBase(spec) {}

    // Absolute size at a level; 0 removes it. Removing an absent level is a
    // no-op, including one outside the window (this book never holds it).
    void set(Side s, std::int64_t px, std::int64_t qty) {
        if (qty < 0) throw std::invalid_argument("negative level size");
        const std::int64_t t = tick_of(px);
        const std::uint64_t id = level_id(t, s);
        ::ull::qpbf::Side os{};
        std::int64_t op = 0;
        std::uint32_t oq = 0;
        const bool have = book_->get_order(id, os, op, oq);
        if (qty == 0) {
            if (have) check(book_->reduce(id, oq));
            return;
        }
        const std::uint32_t lots = lots_of(qty);
        if (lots == 0) throw Unrepresentable();
        check(have ? book_->modify(id, t, lots) : book_->add(id, detail::qside(s), t, lots));
    }
    std::int64_t get(Side s, std::int64_t px) const {
        if (px < 0 || px % spec_.tick != 0) return 0;
        return static_cast<std::int64_t>(book_->level_at(detail::qside(s), px / spec_.tick).qty) * spec_.lot;
    }
    // Load a snapshot: `bids` and `asks` are (px, qty) pairs in fixed point.
    template <class Rows>
    void load(const Rows& bids, const Rows& asks) {
        std::int64_t bb = INT64_MIN, ba = INT64_MAX;
        for (const auto& [p, q] : bids)
            if (q > 0) bb = std::max(bb, tick_of(p));
        for (const auto& [p, q] : asks)
            if (q > 0) ba = std::min(ba, tick_of(p));
        const bool hb = bb != INT64_MIN, ha = ba != INT64_MAX;
        recentre(hb && ha ? (bb + ba) / 2 : (hb ? bb : ba), hb || ha);
        auto put = [&](Side s, std::int64_t p, std::int64_t q) {
            if (q <= 0) return;
            try {
                set(s, p, q);
            } catch (const OutOfWindow&) {
                ++clipped_;
            }
        };
        for (const auto& [p, q] : bids) put(Side::Bid, p, q);
        for (const auto& [p, q] : asks) put(Side::Ask, p, q);
    }
    void clear() { book_->clear(); }
    std::vector<Level> levels(Side s) const { return scan(s); }
    bool empty() const { return book_->order_count() == 0; }
    bool crossed() const {
        const auto bb = book_->best_bid(), ba = book_->best_ask();
        return bb != ::ull::book::L3Book::kNoPrice && ba != ::ull::book::L3Book::kNoPrice && bb >= ba;
    }
    L2Book to_l2() const { return to_l2_impl(); }
    std::string dump() const { return to_l2().dump(); }
    bool operator==(const FlatL2Book& o) const {
        return levels(Side::Bid) == o.levels(Side::Bid) && levels(Side::Ask) == o.levels(Side::Ask);
    }

private:
    static std::uint64_t level_id(std::int64_t t, Side s) {
        return static_cast<std::uint64_t>(t) * 2 + (s == Side::Ask ? 1 : 0) + 1;
    }
};

// Fixed-capacity string-id -> u64 handle table. Keys are copied into fixed
// kMaxId-byte slots; linear probing with backward-shift delete (Knuth, TAOCP
// vol. 3, 6.4, Algorithm R); handles come from a preallocated free stack.
class IdInterner {
public:
    static constexpr std::size_t kMaxId = 48;
    static constexpr std::uint64_t kHandleBit = 1ull << 63;

    explicit IdInterner(std::uint32_t cap) : cap_(cap) {
        n_ = 1;
        while (n_ < 2ull * cap) n_ <<= 1;
        keys_ = std::make_unique<char[]>(n_ * kMaxId);
        lens_ = std::make_unique<std::uint8_t[]>(n_);
        vals_ = std::make_unique<std::uint64_t[]>(n_);
        free_ = std::make_unique<std::uint64_t[]>(cap);
        clear();
    }
    void clear() {
        std::fill_n(vals_.get(), n_, 0);
        for (std::uint32_t i = 0; i < cap_; ++i) free_[i] = kHandleBit | (cap_ - i);
        top_ = cap_;
    }
    std::uint64_t get(std::string_view k) const {
        if (k.size() > kMaxId) return 0;
        for (std::uint64_t i = home(k);; i = (i + 1) & (n_ - 1)) {
            if (vals_[i] == 0) return 0;
            if (eq(i, k)) return vals_[i];
        }
    }
    // Interns k (must be absent). Throws when k is too long or the table full.
    std::uint64_t insert(std::string_view k) {
        if (k.size() > kMaxId) throw std::invalid_argument("order id longer than kMaxId");
        if (top_ == 0) throw BookFull();
        const std::uint64_t h = free_[--top_];
        std::uint64_t i = home(k);
        while (vals_[i] != 0) i = (i + 1) & (n_ - 1);
        std::memcpy(&keys_[i * kMaxId], k.data(), k.size());
        lens_[i] = static_cast<std::uint8_t>(k.size());
        vals_[i] = h;
        return h;
    }
    void erase(std::string_view k) {
        if (k.size() > kMaxId) return;
        std::uint64_t i = home(k);
        for (;; i = (i + 1) & (n_ - 1)) {
            if (vals_[i] == 0) return;
            if (eq(i, k)) break;
        }
        free_[top_++] = vals_[i];
        std::uint64_t hole = i, j = i;
        for (;;) {
            j = (j + 1) & (n_ - 1);
            if (vals_[j] == 0) break;
            const std::uint64_t h = home(key(j));
            const bool keep = (hole <= j) ? (hole < h && h <= j) : (hole < h || h <= j);
            if (!keep) {
                std::memcpy(&keys_[hole * kMaxId], &keys_[j * kMaxId], kMaxId);
                lens_[hole] = lens_[j];
                vals_[hole] = vals_[j];
                hole = j;
            }
        }
        vals_[hole] = 0;
    }
    std::size_t size() const { return cap_ - top_; }
    template <class F>
    void for_each(F&& f) const {
        for (std::uint64_t i = 0; i < n_; ++i)
            if (vals_[i]) f(key(i), vals_[i]);
    }

private:
    std::uint32_t cap_;
    std::uint64_t n_ = 0;
    std::unique_ptr<char[]> keys_;
    std::unique_ptr<std::uint8_t[]> lens_;
    std::unique_ptr<std::uint64_t[]> vals_;
    std::unique_ptr<std::uint64_t[]> free_;
    std::uint32_t top_ = 0;

    std::string_view key(std::uint64_t i) const { return {&keys_[i * kMaxId], lens_[i]}; }
    bool eq(std::uint64_t i, std::string_view k) const { return key(i) == k; }
    std::uint64_t home(std::string_view k) const {
        std::uint64_t h = 1469598103934665603ull;  // FNV-1a
        for (unsigned char c : k) h = (h ^ c) * 1099511628211ull;
        return (h * 0x9E3779B97F4A7C15ull >> 32) & (n_ - 1);
    }
};

// Order book (Coinbase `full`) on the flat book, same interface and
// semantics as L3Book. As there, an order matched or resized down to zero
// stays live (it counts in order_count) until its `done`; the flat book cannot
// hold a zero-size order, so such an order keeps its id, side and tick in the
// interner's side tables and simply has no slot in the flat book.
class FlatL3Book : public detail::FlatBase {
public:
    using Order = L3Book::Order;

    explicit FlatL3Book(ProductSpec spec = {})
        : FlatBase(spec),
          ids_(spec.max_orders),
          side_(std::make_unique<Side[]>(spec.max_orders + 1)),
          tick_(std::make_unique<std::int64_t[]>(spec.max_orders + 1)) {}

    void clear() {
        book_->clear();
        ids_.clear();
    }
    // Load a snapshot of (id, side, px, qty) rows; re-centres the window.
    template <class Rows>
    void load(const Rows& rows) {
        std::int64_t bb = INT64_MIN, ba = INT64_MAX;
        for (const auto& r : rows) {
            const std::int64_t t = tick_of(std::get<2>(r));
            if (std::get<1>(r) == Side::Bid)
                bb = std::max(bb, t);
            else
                ba = std::min(ba, t);
        }
        const bool hb = bb != INT64_MIN, ha = ba != INT64_MAX;
        recentre(hb && ha ? (bb + ba) / 2 : (hb ? bb : ba), hb || ha);
        ids_.clear();
        for (const auto& r : rows) {
            try {
                open(std::get<0>(r), std::get<1>(r), std::get<2>(r), std::get<3>(r));
            } catch (const OutOfWindow&) {
                ++clipped_;
            }
        }
    }
    void open(std::string_view id, Side s, std::int64_t px, std::int64_t qty) {
        if (ids_.get(id)) throw std::logic_error("duplicate order id");
        const std::int64_t t = tick_of(px);
        const std::uint32_t lots = lots_of(qty);
        if (lots > 0) {
            // Check the band before interning so a failure leaves no trace.
            const std::int64_t li = t - book_->base_price();
            if (li < 0 || li >= static_cast<std::int64_t>(spec_.n_levels)) throw OutOfWindow();
        }
        const std::uint64_t h = ids_.insert(id);
        side_[slot(h)] = s;
        tick_[slot(h)] = t;
        if (lots == 0) return;
        const auto r = book_->add(h, detail::qside(s), t, lots);
        if (r != ::ull::book::Result::Ok) {
            ids_.erase(id);
            check(r);
        }
    }
    // Remove if live; orders that never rested (market, IOC) are not here.
    void done(std::string_view id) {
        const std::uint64_t h = ids_.get(id);
        if (!h) return;
        const std::uint32_t q = live_qty(h);
        if (q) check(book_->reduce(h, q));
        ids_.erase(id);
    }
    void reduce(std::string_view id, std::int64_t by) {
        const std::uint64_t h = ids_.get(id);
        if (!h) return;
        const std::uint32_t q = live_qty(h);
        const std::uint32_t lots = lots_of(by);
        if (lots > q) throw std::logic_error("match larger than resting size");
        if (lots) check(book_->reduce(h, lots));  // at zero the id stays live
    }
    void resize(std::string_view id, std::int64_t new_qty) {
        const std::uint64_t h = ids_.get(id);
        if (!h) return;
        const std::uint32_t q = live_qty(h), lots = lots_of(new_qty);
        if (lots == q) return;
        if (lots == 0)
            check(book_->reduce(h, q));
        else if (q == 0)
            check(book_->add(h, detail::qside(side_[slot(h)]), tick_[slot(h)], lots));
        else
            check(book_->modify(h, tick_[slot(h)], lots));
    }
    L2Book l2() const { return to_l2_impl(); }
    std::size_t order_count() const { return ids_.size(); }
    // Live orders by id (reporting; allocates).
    std::unordered_map<std::string, Order> orders() const {
        std::unordered_map<std::string, Order> m;
        ids_.for_each([&](std::string_view k, std::uint64_t h) {
            m.emplace(std::string(k), Order{side_[slot(h)], tick_[slot(h)] * spec_.tick,
                                            static_cast<std::int64_t>(live_qty(h)) * spec_.lot});
        });
        return m;
    }

private:
    IdInterner ids_;
    std::unique_ptr<Side[]> side_;
    std::unique_ptr<std::int64_t[]> tick_;

    static std::size_t slot(std::uint64_t h) { return static_cast<std::size_t>(h & ~IdInterner::kHandleBit); }
    std::uint32_t live_qty(std::uint64_t h) const {
        ::ull::qpbf::Side s{};
        std::int64_t p = 0;
        std::uint32_t q = 0;
        return book_->get_order(h, s, p, q) ? q : 0;
    }
};

}  // namespace qp::ull::feed
