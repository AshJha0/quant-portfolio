#include "out.hpp"
struct A { char c; double d; char e; };
struct B { double d; char c; char e; };
struct alignas(64) C { int x; };
struct E {};
struct F { [[no_unique_address]] E e; int x; };
int main() {
    out(static_cast<long long>(sizeof(A)));
    out(static_cast<long long>(sizeof(B)));
    out(static_cast<long long>(sizeof(C)));
    out(static_cast<long long>(sizeof(E)));
    out(static_cast<long long>(sizeof(F)));
}
