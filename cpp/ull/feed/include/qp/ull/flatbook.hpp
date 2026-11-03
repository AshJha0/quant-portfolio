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
// Entries the window cannot hold - deep levels, prices off the tick grid,
// sizes above u32 lots, ids longer than kMaxId - go to an exact side store
// (a map-based L2Book plus a map of far orders), so the book is always
// complete. Only a bid above the window or an ask below it - the touch has
// moved out - throws OutOfWindow.
//
// Formal statement: FlatL2Book holds exactly the levels L2Book would hold
// after the same calls, and FlatL3Book matches L3Book call for call (levels,
// orders, order count); tests/test_feed.cpp checks both against the map
// books. Near the touch, set/open/done/reduce/resize never allocate
// (tests/test_noalloc.cpp counts operator new); the side store does.
//
// Worked example: tick 0.01, window 64 ticks. A snapshot with best bid
// 100.00 and best ask 100.02 centres the window on tick 10001, covering
// [99.69, 100.33). set(Ask, 100.05, 2.0) stores 200000000 lots at tick
// 10005; set(Ask, 101.00, 1.0) is a deep ask and goes to the side store;
// set(Bid, 101.00, 1.0) would put the touch above the window and throws
// OutOfWindow.
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

// Window, tick/lot conversion, the flat book and the exact side store
// (`far_`, a map-based L2Book) shared by both adapters.
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
    // Levels held in the side store (diagnostics).
    std::size_t far_levels() const {
        return far_.levels(Side::Bid).size() + far_.levels(Side::Ask).size();
    }

protected:
    ProductSpec spec_;
    std::unique_ptr<::ull::book::L3Book> book_;
    L2Book far_;  // deep levels, off-grid prices, sizes above u32 lots: exact

    bool on_grid(std::int64_t px) const { return px >= 0 && px % spec_.tick == 0; }
    bool in_window(std::int64_t t) const {
        const std::int64_t b = book_->base_price();
        return t >= b && t < b + static_cast<std::int64_t>(spec_.n_levels);
    }
    // True and (t, lots) set when the entry belongs in the flat book; false
    // for the side store. Throws OutOfWindow when the touch has left the
    // window (a bid above it or an ask below it), except while loading a
    // snapshot. A zero size in the window is flat with lots = 0.
    bool place(Side s, std::int64_t px, std::int64_t qty, bool snapshot, std::int64_t& t,
               std::uint32_t& lots) const {
        if (!on_grid(px)) return false;
        t = px / spec_.tick;
        if (!in_window(t)) {
            const std::int64_t b = book_->base_price();
            const bool beyond = s == Side::Bid ? t >= b + static_cast<std::int64_t>(spec_.n_levels) : t < b;
            if (beyond && !snapshot) throw OutOfWindow();
            return false;
        }
        if (qty < 0 || qty % spec_.lot != 0 || qty / spec_.lot > 0xFFFFFFFFll) return false;
        lots = static_cast<std::uint32_t>(qty / spec_.lot);
        return true;
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
        far_.clear();
    }
    template <class It>
    void recentre_on(It first, It last) {
        std::int64_t bb = INT64_MIN, ba = INT64_MAX;
        for (; first != last; ++first) {
            const auto [s, p, q] = *first;
            if (q <= 0 || !on_grid(p)) continue;
            if (s == Side::Bid)
                bb = std::max(bb, p / spec_.tick);
            else
                ba = std::min(ba, p / spec_.tick);
        }
        const bool hb = bb != INT64_MIN, ha = ba != INT64_MAX;
        recentre(hb && ha ? (bb + ba) / 2 : (hb ? bb : ba), hb || ha);
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
    // Flat levels plus the side store (reporting; allocates).
    L2Book to_l2_impl() const {
        L2Book b = far_;
        const std::int64_t lo = book_->base_price(), hi = lo + spec_.n_levels;
        for (Side s : {Side::Bid, Side::Ask}) {
            const auto qs = qside(s);
            for (std::int64_t t = lo; t < hi; ++t) {
                const auto& L = book_->level_at(qs, t);
                if (L.count) b.add(s, t * spec_.tick, static_cast<std::int64_t>(L.qty) * spec_.lot);
            }
        }
        return b;
    }
    // Best price on one side across both tiers, or nullopt-like INT64_MIN.
    std::int64_t best_px(Side s) const {
        const std::int64_t t = s == Side::Bid ? book_->best_bid() : book_->best_ask();
        std::int64_t flat = t == ::ull::book::L3Book::kNoPrice ? INT64_MIN : t * spec_.tick;
        const auto far = far_.best(s);
        if (flat == INT64_MIN) return far;
        if (far == INT64_MIN) return flat;
        return s == Side::Bid ? std::max(flat, far) : std::min(flat, far);
    }
};

}  // namespace detail

// Level book (absolute sizes per price) on the flat book.
class FlatL2Book : public detail::FlatBase {
public:
    explicit FlatL2Book(ProductSpec spec = {}) : FlatBase(spec) {}

    // Absolute size at a level; 0 removes it. Removing an absent level is a
    // no-op. Near the touch this does not allocate.
    void set(Side s, std::int64_t px, std::int64_t qty) { set_impl(s, px, qty, false); }
    std::int64_t get(Side s, std::int64_t px) const {
        std::int64_t q = far_.get(s, px);
        if (on_grid(px) && in_window(px / spec_.tick))
            q += static_cast<std::int64_t>(book_->level_at(detail::qside(s), px / spec_.tick).qty) * spec_.lot;
        return q;
    }
    // Load a snapshot: `bids` and `asks` are (px, qty) pairs in fixed point.
    template <class Rows>
    void load(const Rows& bids, const Rows& asks) {
        std::vector<std::tuple<Side, std::int64_t, std::int64_t>> all;
        for (const auto& [p, q] : bids) all.emplace_back(Side::Bid, p, q);
        for (const auto& [p, q] : asks) all.emplace_back(Side::Ask, p, q);
        recentre_on(all.begin(), all.end());
        for (const auto& [s, p, q] : all)
            if (q > 0) set_impl(s, p, q, true);
    }
    void clear() {
        book_->clear();
        far_.clear();
    }
    std::vector<Level> levels(Side s) const { return to_l2().levels(s); }
    bool empty() const { return book_->order_count() == 0 && far_.empty(); }
    bool crossed() const {
        const auto bb = best_px(Side::Bid), ba = best_px(Side::Ask);
        return bb != INT64_MIN && ba != INT64_MIN && bb >= ba;
    }
    L2Book to_l2() const { return to_l2_impl(); }
    std::string dump() const { return to_l2().dump(); }
    bool operator==(const FlatL2Book& o) const { return to_l2() == o.to_l2(); }

private:
    static std::uint64_t level_id(std::int64_t t, Side s) {
        return static_cast<std::uint64_t>(t) * 2 + (s == Side::Ask ? 1 : 0) + 1;
    }
    void set_impl(Side s, std::int64_t px, std::int64_t qty, bool snapshot) {
        if (qty < 0) throw std::invalid_argument("negative level size");
        if (!far_.empty()) far_.set(s, px, 0);
        const bool grid_in = on_grid(px) && in_window(px / spec_.tick);
        const std::uint64_t id = grid_in ? level_id(px / spec_.tick, s) : 0;
        ::ull::qpbf::Side os{};
        std::int64_t op = 0;
        std::uint32_t oq = 0;
        const bool have = grid_in && book_->get_order(id, os, op, oq);
        if (qty == 0) {
            if (have) check(book_->reduce(id, oq));
            return;
        }
        std::int64_t t = 0;
        std::uint32_t lots = 0;
        if (place(s, px, qty, snapshot, t, lots) && lots > 0) {
            check(have ? book_->modify(id, t, lots) : book_->add(id, detail::qside(s), t, lots));
            return;
        }
        if (have) check(book_->reduce(id, oq));
        far_.set(s, px, qty);
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
// interner's side tables and simply has no slot in the flat book. Orders the
// window cannot hold (deep, off-grid, above u32 lots) live in `far_orders_`
// and add to the side-store levels.
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
        far_.clear();
        far_orders_.clear();
    }
    // Load a snapshot of (id, side, px, qty) rows; re-centres the window.
    template <class Rows>
    void load(const Rows& rows) {
        std::vector<std::tuple<Side, std::int64_t, std::int64_t>> all;
        for (const auto& r : rows) all.emplace_back(std::get<1>(r), std::get<2>(r), std::get<3>(r));
        recentre_on(all.begin(), all.end());
        ids_.clear();
        far_orders_.clear();
        for (const auto& r : rows) open_impl(std::get<0>(r), std::get<1>(r), std::get<2>(r), std::get<3>(r), true);
    }
    void open(std::string_view id, Side s, std::int64_t px, std::int64_t qty) {
        open_impl(id, s, px, qty, false);
    }
    // Remove if live; orders that never rested (market, IOC) are not here.
    void done(std::string_view id) {
        if (const std::uint64_t h = ids_.get(id)) {
            const std::uint32_t q = live_qty(h);
            if (q) check(book_->reduce(h, q));
            ids_.erase(id);
            return;
        }
        if (far_orders_.empty()) return;
        auto it = far_orders_.find(std::string(id));
        if (it == far_orders_.end()) return;
        far_.add(it->second.side, it->second.px, -it->second.qty);
        far_orders_.erase(it);
    }
    void reduce(std::string_view id, std::int64_t by) {
        if (const std::uint64_t h = ids_.get(id)) {
            const std::uint32_t q = live_qty(h);
            if (by < 0 || by % spec_.lot != 0 || by / spec_.lot > q)
                throw std::logic_error("match larger than resting size");
            const auto lots = static_cast<std::uint32_t>(by / spec_.lot);
            if (lots) check(book_->reduce(h, lots));  // at zero the id stays live
            return;
        }
        if (far_orders_.empty()) return;
        auto it = far_orders_.find(std::string(id));
        if (it == far_orders_.end()) return;
        if (by > it->second.qty) throw std::logic_error("match larger than resting size");
        it->second.qty -= by;
        far_.add(it->second.side, it->second.px, -by);
    }
    void resize(std::string_view id, std::int64_t new_qty) {
        if (const std::uint64_t h = ids_.get(id)) {
            const std::uint32_t q = live_qty(h);
            const Side s = side_[slot(h)];
            const std::int64_t t = tick_[slot(h)];
            std::int64_t tt = 0;
            std::uint32_t lots = 0;
            if (!place(s, t * spec_.tick, new_qty, true, tt, lots)) {  // too big: move out
                if (q) check(book_->reduce(h, q));
                ids_.erase(id);
                far_orders_[std::string(id)] = Order{s, t * spec_.tick, new_qty};
                far_.add(s, t * spec_.tick, new_qty);
                return;
            }
            if (lots == q) return;
            if (lots == 0)
                check(book_->reduce(h, q));
            else if (q == 0)
                check(book_->add(h, detail::qside(s), t, lots));
            else
                check(book_->modify(h, t, lots));
            return;
        }
        if (far_orders_.empty()) return;
        auto it = far_orders_.find(std::string(id));
        if (it == far_orders_.end()) return;
        far_.add(it->second.side, it->second.px, new_qty - it->second.qty);
        it->second.qty = new_qty;
    }
    L2Book l2() const { return to_l2_impl(); }
    std::size_t order_count() const { return ids_.size() + far_orders_.size(); }
    // Live orders by id (reporting; allocates).
    std::unordered_map<std::string, Order> orders() const {
        std::unordered_map<std::string, Order> m(far_orders_);
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
    std::unordered_map<std::string, Order> far_orders_;

    static std::size_t slot(std::uint64_t h) { return static_cast<std::size_t>(h & ~IdInterner::kHandleBit); }
    std::uint32_t live_qty(std::uint64_t h) const {
        ::ull::qpbf::Side s{};
        std::int64_t p = 0;
        std::uint32_t q = 0;
        return book_->get_order(h, s, p, q) ? q : 0;
    }
    bool live(std::string_view id) const {
        return ids_.get(id) || (!far_orders_.empty() && far_orders_.count(std::string(id)));
    }
    void open_impl(std::string_view id, Side s, std::int64_t px, std::int64_t qty, bool snapshot) {
        if (live(id)) throw std::logic_error("duplicate order id");
        std::int64_t t = 0;
        std::uint32_t lots = 0;
        if (!place(s, px, qty, snapshot, t, lots) || id.size() > IdInterner::kMaxId) {
            if (qty < 0) throw std::invalid_argument("negative order size");
            far_orders_[std::string(id)] = Order{s, px, qty};
            far_.add(s, px, qty);
            return;
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
};

}  // namespace qp::ull::feed
