// qp::exec::Book: the fast limit order book (W25).
//
// Same semantics as the Python reference book,
// python/execution/src/qp_exec/lob/book.py, whose rules are written down in
// python/execution/src/qp_exec/lob/SPEC.md. The two must produce the same
// digest after every event; tests/test_differential_cpp.py in the Python
// package drives exec_core_replay over a seeded event file and compares.
//
// Layout: orders live in a slot pool with intrusive doubly linked queues
// (O(1) cancel), price levels in two ordered maps. The flat price-indexed L3
// book of the low-latency track (W30) tests against this book and the
// reference book.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace qp::exec {

enum class Side : std::uint8_t { Buy = 1, Sell = 2 };
enum class OrdType : std::uint8_t { Limit = 1, Market = 2 };
enum class Tif : std::uint8_t { Day = 1, Ioc = 2, Fok = 3 };
enum class Peg : std::uint8_t { None = 0, Mid = 1, Primary = 2 };
enum class Phase : std::uint8_t { Continuous = 0, Auction = 1 };
enum class Reject : std::uint8_t {
    None = 0,
    Invalid = 1,
    DuplicateId = 2,
    PostOnlyWouldCross = 3,
    FokUnfillable = 4,
    NotInAuction = 5
};

struct NewOrder {
    std::int64_t id = 0;
    Side side = Side::Buy;
    std::int64_t qty = 0;
    std::int64_t price = 0;
    OrdType type = OrdType::Limit;
    Tif tif = Tif::Day;
    bool post_only = false;
    bool hidden = false;
    std::int64_t display_qty = 0;
    Peg peg = Peg::None;
};

struct Fill {
    std::int64_t taker_id, maker_id, price, qty, ts;
    std::uint8_t aggressor;  // 0 in an auction, else the taker Side
};

struct AddResult {
    bool accepted;
    Reject reject;
    std::int64_t resting_qty;
};

class Book {
public:
    // Every call appends its fills to `fills` (not cleared).
    AddResult add(const NewOrder& o, std::int64_t ts, std::vector<Fill>& fills);
    bool cancel(std::int64_t id, std::int64_t ts, std::vector<Fill>& fills);
    bool modify(std::int64_t id, std::int64_t new_qty, std::int64_t ts, std::vector<Fill>& fills);
    void start_auction();
    void uncross(std::int64_t ts, std::int64_t ref_price, std::vector<Fill>& fills);

    [[nodiscard]] std::optional<std::int64_t> best_bid() const { return best_displayed(Side::Buy); }
    [[nodiscard]] std::optional<std::int64_t> best_ask() const {
        return best_displayed(Side::Sell);
    }
    [[nodiscard]] Phase phase() const { return phase_; }
    [[nodiscard]] std::size_t order_count() const { return ids_.size(); }

    void state_words(std::vector<std::int64_t>& out) const;
    [[nodiscard]] std::uint64_t digest(const std::vector<Fill>& fills) const;

private:
    struct Ord {
        std::int64_t id, price, visible, reserve, display, seq;
        Side side;
        Peg peg;
        bool hidden, active;
        std::int32_t prev = -1, next = -1;
        [[nodiscard]] bool displayed_class() const { return !hidden && peg != Peg::Mid; }
        [[nodiscard]] std::int64_t total() const { return visible + reserve; }
    };
    struct Level {
        std::int32_t dh = -1, dt = -1, hh = -1, ht = -1;
        [[nodiscard]] bool empty() const { return dh < 0 && hh < 0; }
        [[nodiscard]] std::int32_t head() const { return dh >= 0 ? dh : hh; }
    };
    using Bids = std::map<std::int64_t, Level, std::greater<>>;
    using Asks = std::map<std::int64_t, Level, std::less<>>;

    template <class F>
    void for_levels(Side s, F&& f) const {
        if (s == Side::Buy) {
            for (auto& [p, l] : bids_)
                if (!f(p, l)) return;
        } else {
            for (auto& [p, l] : asks_)
                if (!f(p, l)) return;
        }
    }
    template <class F>
    void for_queue(std::int32_t h, F&& f) const {
        for (std::int32_t i = h; i >= 0; i = pool_[static_cast<std::size_t>(i)].next) f(i);
    }
    Ord& at(std::int32_t i) { return pool_[static_cast<std::size_t>(i)]; }
    [[nodiscard]] const Ord& at(std::int32_t i) const { return pool_[static_cast<std::size_t>(i)]; }

    [[nodiscard]] std::optional<std::int64_t> best(Side s) const;
    [[nodiscard]] std::optional<std::int64_t> best_displayed(Side s) const;
    [[nodiscard]] std::optional<std::int64_t> ref_best(Side s) const;
    Level* level(Side s, std::int64_t p);
    std::int32_t alloc(const Ord& o);
    void insert(std::int32_t i);
    void unlink(std::int32_t i);
    void drop(std::int32_t i);
    void consume(std::int32_t i, std::int64_t q);
    static bool crosses(Side s, std::optional<std::int64_t> lim, std::int64_t px);
    std::int64_t available(Side s, std::optional<std::int64_t> lim, std::int64_t need) const;
    std::int64_t match(std::int64_t id, Side s, std::optional<std::int64_t> lim, std::int64_t qty,
                       std::int64_t ts, std::vector<Fill>& fills);
    [[nodiscard]] std::optional<std::int64_t> peg_target(const Ord& o) const;
    void after_event(std::int64_t ts, std::vector<Fill>& fills);
    std::int32_t rest_new(const NewOrder& o, std::int64_t qty);
    static bool valid(const NewOrder& o);
    std::pair<std::optional<std::int64_t>, std::int64_t> auction_price(std::int64_t ref) const;

    Bids bids_;
    Asks asks_;
    std::vector<Ord> pool_;
    std::vector<std::int32_t> free_;
    std::unordered_map<std::int64_t, std::int32_t> ids_;
    std::vector<std::int64_t> pegs_;
    std::int64_t seq_ = 0;
    Phase phase_ = Phase::Continuous;
};

}  // namespace qp::exec
