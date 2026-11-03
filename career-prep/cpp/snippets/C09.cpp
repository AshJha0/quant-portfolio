#include <memory>
#include "out.hpp"
int main() {
    auto p = std::make_shared<int>(1);
    auto q = p;
    std::weak_ptr<int> w = p;
    out(p.use_count());
    q.reset();
    out(p.use_count());
    out(w.expired() ? 1 : 0);
    p.reset();
    out(w.expired() ? 1 : 0);
    auto u = std::make_unique<int>(7);
    std::shared_ptr<int> s = std::move(u);
    out(u ? 1 : 0);
    out(s.use_count());
}
