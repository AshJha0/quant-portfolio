#include <string>
#include "out.hpp"
struct P {
    std::string n;
    explicit P(const char* s) : n(s) { out(n); }
    ~P() { out("~" + n); }
};
struct S {
    P b;
    P a;
    S() : a("a"), b("b") {}  // the order written here does not matter
};
int main() {
    S s;
    P x("x");
}
