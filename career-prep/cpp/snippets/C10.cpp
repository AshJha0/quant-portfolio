#include "out.hpp"
template <class... T>
constexpr auto sum(T... t) { return (t + ... + 0); }
template <class... T>
constexpr auto sub_left(T... t) { return (... - t); }
template <class... T>
constexpr auto sub_right(T... t) { return (t - ...); }
constexpr int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
static_assert(fib(10) == 55);
int main() {
    out(sum(1, 2, 3));
    out(sub_left(1, 2, 3));
    out(sub_right(1, 2, 3));
    out(fib(20));
}
