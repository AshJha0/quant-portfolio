#include <vector>
#include "out.hpp"
struct A {
    virtual ~A() = default;
    virtual const char* name() const { return "A"; }
};
struct B : A {
    const char* name() const override { return "B"; }
};
void by_value(A a) { out(a.name()); }
void by_ref(const A& a) { out(a.name()); }
int main() {
    B b;
    by_value(b);
    by_ref(b);
    A a = b;
    out(a.name());
    std::vector<A> v{b};
    out(v[0].name());
}
