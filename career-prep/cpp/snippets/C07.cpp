#include <utility>
#include "out.hpp"
struct T {
    T() = default;
    T(const T&) { out("copy"); }
    T(T&&) noexcept { out("move"); }
};
int main() {
    const T a;
    T b = std::move(a);
    T c = std::move(b);
    T d = T(T(T()));
    out("end");
}
