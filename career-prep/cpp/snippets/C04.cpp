#include <cstdint>
#include <type_traits>
#include "out.hpp"
int main() {
    unsigned u = 1;
    int i = -1;
    out(i < u ? "less" : "not-less");
    unsigned char c = 255;
    c += 1;
    out(static_cast<long long>(c));
    auto x = 'a' + 1;
    out(std::is_same_v<decltype(x), int> ? 1 : 0);
    std::uint32_t z = 0;
    out(static_cast<long long>(z - 1));
}
