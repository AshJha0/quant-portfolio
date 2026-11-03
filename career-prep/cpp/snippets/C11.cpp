#include <algorithm>
#include <vector>
#include "out.hpp"
int main() {
    std::vector<int> v{1, 2, 3, 2, 5};
    auto it = std::remove(v.begin(), v.end(), 2);
    out(static_cast<long long>(v.size()));
    out(static_cast<long long>(it - v.begin()));
    v.erase(it, v.end());
    for (int x : v) out(x);
    std::vector<int> w{4, 4, 1};
    out(static_cast<long long>(std::erase(w, 4)));
    out(static_cast<long long>(w.size()));
}
