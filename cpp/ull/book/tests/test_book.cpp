// L3 book tests: hand-worked cases, then the flat-array book against a
// deliberately simple std::map/std::list reference after EVERY message of
// generated days (levels, quantities, counts and FIFO order, both sides).
//
// Seam: the plan names the W25 reference book as the oracle. W25 is not
// merged yet; RefBook below implements the same rules (QPBF.md section 5)
// and is replaced by the W25 book at integration.
#include "../../bench/ull_test.hpp"
#include "../../qpbf/generator.hpp"
#include "../l3_book.hpp"

#include <cstdio>
#include <list>
#include <map>
#include <unordered_map>

using namespace ull::book;
using namespace ull::qpbf;

namespace {

struct RefBook {
    struct O {
        Side s;
        std::int64_t p;
        std::uint32_t q;
    };
    std::unordered_map<std::uint64_t, O> orders;
    std::map<std::int64_t, std::list<std::uint64_t>> bids, asks;

    std::map<std::int64_t, std::list<std::uint64_t>>& side(Side s) {
        return s == Side::Bid ? bids : asks;
    }

    void remove_from_level(std::uint64_t id, const O& o) {
        auto& m = side(o.s);
        auto it = m.find(o.p);
        it->second.remove(id);
        if (it->second.empty()) m.erase(it);
    }
    void apply(const Msg& m) {
        switch (m.type) {
            case MsgType::Add:
            case MsgType::Snapshot:
                orders[m.order_id] = {m.side, m.price, m.qty};
                side(m.side)[m.price].push_back(m.order_id);
                break;
            case MsgType::Modify: {
                O& o = orders.at(m.order_id);
                if (m.price == o.p && m.qty <= o.q) {
                    o.q = m.qty;
                    break;
                }
                remove_from_level(m.order_id, o);
                o.p = m.price;
                o.q = m.qty;
                side(o.s)[o.p].push_back(m.order_id);
                break;
            }
            case MsgType::Cancel:
            case MsgType::Execute: {
                O& o = orders.at(m.order_id);
                if (m.qty == o.q) {
                    remove_from_level(m.order_id, o);
                    orders.erase(m.order_id);
                } else
                    o.q -= m.qty;
                break;
            }
            default:
                break;
        }
    }
};

// Full equality: every reference level matches the book level by qty, count
// and FIFO ids+qty; the book has no extra levels (order counts agree); best
// prices agree.
bool equal(const L3Book& b, RefBook& r) {
    if (b.order_count() != r.orders.size()) return false;
    const std::int64_t rb = r.bids.empty() ? L3Book::kNoPrice : r.bids.rbegin()->first;
    const std::int64_t ra = r.asks.empty() ? L3Book::kNoPrice : r.asks.begin()->first;
    if (b.best_bid() != rb || b.best_ask() != ra) return false;
    for (Side s : {Side::Bid, Side::Ask}) {
        for (auto& [p, ids] : r.side(s)) {
            const Level& L = b.level_at(s, p);
            std::uint64_t q = 0;
            for (auto id : ids) q += r.orders.at(id).q;
            if (L.qty != q || L.count != ids.size()) return false;
            auto it = ids.begin();
            bool ok = true;
            b.for_each_order(s, p, [&](std::uint64_t id, std::uint32_t qty) {
                ok &= (it != ids.end() && *it == id && r.orders.at(id).q == qty);
                if (it != ids.end()) ++it;
            });
            if (!ok) return false;
        }
    }
    return true;
}

void worked_example() {
    L3Book b(100, 8, 16);
    ULL_CHECK(b.add(1, Side::Bid, 103, 10) == Result::Ok);
    ULL_CHECK(b.add(2, Side::Bid, 103, 5) == Result::Ok);
    ULL_CHECK(b.add(3, Side::Ask, 105, 7) == Result::Ok);
    ULL_CHECK(b.best_bid() == 103 && b.level_at(Side::Bid, 103).qty == 15);
    ULL_CHECK(b.level_at(Side::Bid, 103).count == 2);
    ULL_CHECK(b.best_ask() == 105);
    ULL_CHECK(b.reduce(1, 10) == Result::Ok);
    ULL_CHECK(b.level_at(Side::Bid, 103).qty == 5);
    ULL_CHECK(b.reduce(2, 5) == Result::Ok);
    ULL_CHECK(b.best_bid() == L3Book::kNoPrice);
    ULL_CHECK(b.order_count() == 1);
}

void errors_and_priority() {
    L3Book b(100, 8, 2);
    ULL_CHECK(b.add(1, Side::Bid, 99, 1) == Result::PriceOutOfBand);
    ULL_CHECK(b.add(1, Side::Bid, 108, 1) == Result::PriceOutOfBand);
    ULL_CHECK(b.add(1, Side::None, 101, 1) == Result::BadSide);
    ULL_CHECK(b.add(1, Side::Bid, 101, 5) == Result::Ok);
    ULL_CHECK(b.add(1, Side::Bid, 101, 5) == Result::DuplicateId);
    ULL_CHECK(b.add(2, Side::Bid, 101, 5) == Result::Ok);
    ULL_CHECK(b.add(3, Side::Bid, 101, 5) == Result::PoolFull);
    ULL_CHECK(b.reduce(9, 1) == Result::UnknownId);
    ULL_CHECK(b.reduce(1, 6) == Result::QtyTooLarge);
    // qty down at same price keeps place: 1 stays ahead of 2.
    ULL_CHECK(b.modify(1, 101, 3) == Result::Ok);
    std::uint64_t first = 0;
    b.for_each_order(Side::Bid, 101, [&](std::uint64_t id, std::uint32_t) {
        if (!first) first = id;
    });
    ULL_CHECK(first == 1);
    // qty up loses place.
    ULL_CHECK(b.modify(1, 101, 4) == Result::Ok);
    first = 0;
    b.for_each_order(Side::Bid, 101, [&](std::uint64_t id, std::uint32_t) {
        if (!first) first = id;
    });
    ULL_CHECK(first == 2);
    ULL_CHECK(b.level_at(Side::Bid, 101).qty == 9);
    // Pool slots are reused after removal.
    ULL_CHECK(b.reduce(2, 5) == Result::Ok);
    ULL_CHECK(b.add(3, Side::Ask, 107, 1) == Result::Ok);
    ULL_CHECK(b.best_ask() == 107);
}

// Hash table stress: many ids colliding into the same probe chains, deleted
// in a different order, must all stay findable (backward-shift deletion).
void hash_deletion_stress() {
    L3Book b(0, 4, 4096);
    Xoshiro256ss r(5);
    std::vector<std::uint64_t> live;
    std::uint32_t bad = 0;
    for (int round = 0; round < 200000; ++round) {
        if (live.size() < 4000 && (live.empty() || r.below(2))) {
            const std::uint64_t id = 1 + r.below(1ull << 40);
            if (b.add(id, Side::Bid, 1, 1) == Result::Ok) live.push_back(id);
        } else {
            const auto k = r.below(live.size());
            bad += b.reduce(live[k], 1) != Result::Ok;
            live[k] = live.back();
            live.pop_back();
        }
    }
    Side s;
    std::int64_t p;
    std::uint32_t q;
    for (auto id : live) bad += !b.get_order(id, s, p, q);
    ULL_CHECK(bad == 0);
    ULL_CHECK(b.order_count() == live.size());
}

void against_reference(std::uint64_t seed, std::uint32_t instruments, std::uint64_t n) {
    GeneratorConfig cfg;
    cfg.seed = seed;
    cfg.instruments = instruments;
    cfg.levels = 512;
    cfg.max_live_per_instrument = 3000;
    Generator g(cfg);
    BookSet books(instruments, cfg.base_price, cfg.levels, cfg.max_live_per_instrument);
    std::vector<RefBook> refs(instruments);
    std::uint64_t rejects = 0, first_bad = 0;
    bool all_equal = true;
    Msg m;
    for (std::uint64_t i = 1; i <= n; ++i) {
        g.next(m);
        const Result res = books.apply(m);
        rejects += (res != Result::Ok && res != Result::Ignored);
        if (m.type == MsgType::Heartbeat || m.type == MsgType::Trade) continue;
        refs[m.instrument].apply(m);
        if (!equal(books[m.instrument], refs[m.instrument])) {
            if (all_equal) first_bad = i;
            all_equal = false;
            break;
        }
    }
    std::printf(
        "book vs reference: seed %llu, %u instrument(s), %llu messages, first mismatch %llu\n",
        static_cast<unsigned long long>(seed), instruments, static_cast<unsigned long long>(n),
        static_cast<unsigned long long>(first_bad));
    ULL_CHECK(all_equal);
    ULL_CHECK(rejects == 0);
}

// Negative control: the comparison must notice a one-lot difference.
void comparison_detects_difference() {
    L3Book b(100, 8, 4);
    RefBook r;
    Msg m;
    m.type = MsgType::Add;
    m.order_id = 1;
    m.side = Side::Bid;
    m.price = 101;
    m.qty = 5;
    b.apply(m);
    r.apply(m);
    ULL_CHECK(equal(b, r));
    b.reduce(1, 1);
    ULL_CHECK(!equal(b, r));
}

}  // namespace

int main() {
    comparison_detects_difference();
    worked_example();
    errors_and_priority();
    hash_deletion_stress();
    against_reference(1, 1, 100'000);
    against_reference(2, 3, 100'000);
    ULL_TEST_MAIN_END();
}
