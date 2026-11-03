#include <vector>
#include "out.hpp"
struct T {
    T() { out("ctor"); }
    T(const T&) { out("copy"); }
    T(T&&) noexcept { out("move"); }
    T& operator=(const T&) = default;
    ~T() { out("dtor"); }
};
T make() { return T{}; }
int main() {
    std::vector<T> v;
    v.reserve(2);
    v.push_back(make());
    v.emplace_back();
    T a = v[0];
}
