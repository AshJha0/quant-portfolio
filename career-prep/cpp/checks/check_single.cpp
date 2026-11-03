// Single-threaded checks for C12, C15, C18 and C20. Built with the address
// and undefined-behaviour sanitisers, so a leak, double free or overflow
// fails the run even when every CHECK passes.
#include <list>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "concepts.hpp"
#include "fixed_price.hpp"
#include "object_pool.hpp"
#include "unique_ptr.hpp"

namespace {

int g_live = 0;
struct Tracked {
    int v;
    explicit Tracked(int x) : v(x) {
        if (x < 0) throw std::invalid_argument("negative");
        ++g_live;
    }
    ~Tracked() { --g_live; }
};

void check_unique_ptr() {
    {
        auto a = cp::make_unique<Tracked>(1);
        CHECK(g_live == 1 && a && a->v == 1);
        cp::UniquePtr<Tracked> b = std::move(a);
        CHECK(!a && b && (*b).v == 1 && g_live == 1);
        b = cp::make_unique<Tracked>(2);  // move-assign frees the old object
        CHECK(g_live == 1 && b->v == 2);
        auto& self = b;
        b = std::move(self);  // self-move keeps the object
        CHECK(b && b->v == 2 && g_live == 1);
        Tracked* raw = b.release();
        CHECK(!b && g_live == 1);
        b.reset(raw);
        b.reset();
        CHECK(g_live == 0);
    }
    CHECK(g_live == 0);
    static_assert(!std::is_copy_constructible_v<cp::UniquePtr<int>>);
    static_assert(std::is_nothrow_move_constructible_v<cp::UniquePtr<int>>);
    static_assert(sizeof(cp::UniquePtr<int>) == sizeof(int*));
    PASS("C12");
}

void check_object_pool() {
    cp::ObjectPool<Tracked, 4> pool;
    std::vector<Tracked*> got;
    for (int i = 0; i < 4; ++i) got.push_back(pool.create(i));
    CHECK(pool.create(9) == nullptr);  // exhausted
    CHECK(g_live == 4 && pool.live() == 4);
    pool.destroy(got[1]);
    CHECK(g_live == 3);
    Tracked* again = pool.create(7);
    CHECK(again == got[1] && again->v == 7);  // LIFO reuse of the freed slot
    bool threw = false;
    pool.destroy(again);
    try {
        pool.create(-1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw && pool.live() == 3);
    CHECK(pool.create(5) != nullptr);  // the slot came back after the throw
    for (int i : {0, 2, 3}) pool.destroy(got[static_cast<std::size_t>(i)]);
    CHECK(pool.live() == 1);
    PASS("C15");
}

void check_price() {
    using cp::Price;
    CHECK(Price::parse("101.25")->ticks() == 1012500);
    CHECK(Price::parse("-0.0001")->ticks() == -1);
    CHECK(Price::parse("7")->str() == "7.0000");
    CHECK(Price::parse("0.1")->str() == "0.1000");
    CHECK(!Price::parse("1.23456"));  // finer than a tick
    CHECK(!Price::parse("1.2.3") && !Price::parse("") && !Price::parse("-") && !Price::parse(".") &&
          !Price::parse("1e3"));
    CHECK(!Price::parse("9223372036854775807"));  // overflows once scaled
    // ten additions of 0.1 are exactly 1, unlike double
    Price acc = Price::from_ticks(0);
    for (int i = 0; i < 10; ++i) acc = *acc.checked_add(*Price::parse("0.1"));
    CHECK(acc == *Price::parse("1"));
    double d = 0;
    for (int i = 0; i < 10; ++i) d += 0.1;
    CHECK(d != 1.0);
    CHECK(Price::parse("-12.5")->str() == "-12.5000");
    CHECK(!Price::from_ticks(INT64_MAX).checked_add(Price::from_ticks(1)));
    CHECK(!Price::from_ticks(INT64_MAX / 2 + 1).checked_mul(2));
    CHECK(Price::from_ticks(INT64_MIN).str() == "-922337203685477.5808");
    CHECK(*Price::parse("1.5") < *Price::parse("1.5001") && *Price::parse("1.5") == *Price::parse("1.50"));
    PASS("C18");
}

struct NoPlus {};
void check_concepts() {
    static_assert(cp::SummableRange<std::vector<int>>);
    static_assert(cp::SummableRange<std::list<std::string>>);
    static_assert(!cp::SummableRange<std::vector<NoPlus>>);
    static_assert(!cp::SummableRange<int>);
    CHECK(cp::sum_all(std::vector<int>{1, 2, 3, 4}) == 10);
    CHECK(cp::sum_all(std::list<std::string>{"a", "b"}) == "ab");
    static_assert(alignof(cp::PaddedCounter) == 64 && sizeof(cp::PaddedCounter) == 64);
    cp::PaddedCounter arr[2];
    CHECK(reinterpret_cast<char*>(&arr[1]) - reinterpret_cast<char*>(&arr[0]) == 64);
    PASS("C20");
}

}  // namespace

int main() {
    check_unique_ptr();
    check_object_pool();
    check_price();
    check_concepts();
    return 0;
}
