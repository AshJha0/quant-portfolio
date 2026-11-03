// Spec cases for qp::exec::Book (the same cases as the Python
// tests/test_book_spec.py; the differential test covers the rest).
#include <cassert>
#include <cstdio>
#include <vector>

#include "qp/exec/book.hpp"

using namespace qp::exec;

#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static NewOrder lim(std::int64_t id, Side s, std::int64_t q, std::int64_t p) {
    NewOrder o;
    o.id = id;
    o.side = s;
    o.qty = q;
    o.price = p;
    return o;
}

int main() {
    std::vector<Fill> f;
    {  // price-time priority
        Book b;
        b.add(lim(1, Side::Sell, 5, 101), 1, f);
        b.add(lim(2, Side::Sell, 5, 100), 2, f);
        b.add(lim(3, Side::Sell, 5, 100), 3, f);
        f.clear();
        b.add(lim(9, Side::Buy, 12, 101), 10, f);
        CHECK(f.size() == 3 && f[0].maker_id == 2 && f[1].maker_id == 3 && f[2].maker_id == 1 &&
              f[2].qty == 2);
    }
    {  // iceberg refill within one sweep
        Book b;
        auto ice = lim(1, Side::Sell, 10, 100);
        ice.display_qty = 2;
        b.add(ice, 1, f);
        b.add(lim(2, Side::Sell, 3, 100), 2, f);
        f.clear();
        auto t = lim(3, Side::Buy, 7, 100);
        t.tif = Tif::Ioc;
        b.add(t, 3, f);
        CHECK(f.size() == 3 && f[0].maker_id == 1 && f[1].maker_id == 2 && f[2].maker_id == 1 &&
              f[2].qty == 2);
    }
    {  // FOK counts hidden
        Book b;
        auto h = lim(1, Side::Sell, 4, 101);
        h.hidden = true;
        b.add(h, 1, f);
        auto k = lim(2, Side::Buy, 5, 101);
        k.tif = Tif::Fok;
        CHECK(b.add(k, 2, f).reject == Reject::FokUnfillable);
        k.qty = 4;
        f.clear();
        CHECK(b.add(k, 3, f).accepted && f.size() == 1);
    }
    {  // midpoint pegs cross at an integer midpoint
        Book b;
        b.add(lim(1, Side::Buy, 5, 100), 1, f);
        b.add(lim(2, Side::Sell, 5, 104), 2, f);
        NewOrder p;
        p.id = 3;
        p.side = Side::Buy;
        p.qty = 4;
        p.peg = Peg::Mid;
        b.add(p, 3, f);
        p.id = 4;
        p.side = Side::Sell;
        p.qty = 3;
        f.clear();
        b.add(p, 4, f);
        CHECK(f.size() == 1 && f[0].taker_id == 4 && f[0].maker_id == 3 && f[0].price == 102);
    }
    {  // auction
        Book b;
        b.start_auction();
        b.add(lim(1, Side::Buy, 5, 103), 1, f);
        b.add(lim(2, Side::Buy, 5, 101), 1, f);
        b.add(lim(3, Side::Sell, 4, 100), 1, f);
        b.add(lim(4, Side::Sell, 4, 102), 1, f);
        f.clear();
        b.uncross(9, 100, f);
        CHECK(f.size() == 2 && f[0].price == 102 && f[0].qty == 4 && f[1].qty == 1);
        CHECK(b.phase() == Phase::Continuous && *b.best_bid() == 101);
    }
    std::puts("exec-core book tests passed");
    return 0;
}
