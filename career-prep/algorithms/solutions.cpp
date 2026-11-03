// Reference C++20 solutions for the algorithms pack.
//
// Same problems and the same input encoding as solutions.py. The program
// reads one case per line from stdin:
//
//   <ID> <number of arrays> <len_1> v ... <len_2> v ... ...
//
// and writes the answer array for that case as one line of space-separated
// integers (an empty line for an empty answer). The test
// career-prep/tests/test_algorithms_pack.py compiles this file with
// -Wall -Wextra -Werror and the address and undefined-behaviour sanitisers,
// feeds it seeded random cases and compares every line with the Python
// solution and with a slow oracle.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

using i64 = std::int64_t;
using Vec = std::vector<i64>;
using Args = std::vector<Vec>;

Vec a01(const Args& a) {
    const i64 target = a[1][0];
    std::unordered_map<i64, i64> seen;
    i64 count = 0;
    for (i64 x : a[0]) {
        if (auto it = seen.find(target - x); it != seen.end()) count += it->second;
        ++seen[x];
    }
    return {count};
}

Vec a02(const Args& a) {
    const Vec& xs = a[0];
    i64 best = xs[0], cur = xs[0];
    for (std::size_t i = 1; i < xs.size(); ++i) {
        cur = std::max(xs[i], cur + xs[i]);
        best = std::max(best, cur);
    }
    return {best};
}

Vec a03(const Args& a) {
    Vec tails;
    for (i64 x : a[0]) {
        auto it = std::lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) tails.push_back(x);
        else *it = x;
    }
    return {static_cast<i64>(tails.size())};
}

Vec a04(const Args& a) {
    const Vec& xs = a[0];
    const auto k = static_cast<std::size_t>(a[1][0]);
    std::deque<std::size_t> dq;
    Vec out;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        while (!dq.empty() && xs[dq.back()] <= xs[i]) dq.pop_back();
        dq.push_back(i);
        if (dq.front() + k <= i) dq.pop_front();
        if (i + 1 >= k) out.push_back(xs[dq.front()]);
    }
    return out;
}

Vec a05(const Args& a) {
    Vec v = a[0];
    auto k = static_cast<std::size_t>(a[1][0] - 1);
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return {v[k]};
}

i64 merge_count(Vec& v, Vec& tmp, std::size_t lo, std::size_t hi) {
    if (hi - lo <= 1) return 0;
    std::size_t mid = lo + (hi - lo) / 2;
    i64 inv = merge_count(v, tmp, lo, mid) + merge_count(v, tmp, mid, hi);
    std::size_t i = lo, j = mid, o = lo;
    while (i < mid && j < hi) {
        if (v[i] <= v[j]) tmp[o++] = v[i++];
        else {
            inv += static_cast<i64>(mid - i);
            tmp[o++] = v[j++];
        }
    }
    while (i < mid) tmp[o++] = v[i++];
    while (j < hi) tmp[o++] = v[j++];
    std::copy(tmp.begin() + static_cast<std::ptrdiff_t>(lo), tmp.begin() + static_cast<std::ptrdiff_t>(hi),
              v.begin() + static_cast<std::ptrdiff_t>(lo));
    return inv;
}

Vec a06(const Args& a) {
    Vec v = a[0], tmp(v.size());
    return {merge_count(v, tmp, 0, v.size())};
}

Vec a07(const Args& a) {
    std::vector<std::pair<i64, i64>> iv;
    for (std::size_t i = 0; i + 1 < a[0].size(); i += 2) iv.emplace_back(a[0][i], a[0][i + 1]);
    std::sort(iv.begin(), iv.end());
    std::vector<std::pair<i64, i64>> out;
    for (auto [s, e] : iv) {
        if (!out.empty() && s <= out.back().second) out.back().second = std::max(out.back().second, e);
        else out.emplace_back(s, e);
    }
    Vec flat;
    for (auto [s, e] : out) {
        flat.push_back(s);
        flat.push_back(e);
    }
    return flat;
}

Vec a08(const Args& a) {
    const Vec& xs = a[0];
    Vec out(xs.size(), -1);
    std::vector<std::size_t> st;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        while (!st.empty() && xs[st.back()] < xs[i]) {
            out[st.back()] = xs[i];
            st.pop_back();
        }
        st.push_back(i);
    }
    return out;
}

Vec a09(const Args& a) {
    const Vec& h = a[0];
    std::vector<std::size_t> st;
    i64 best = 0;
    for (std::size_t i = 0; i <= h.size(); ++i) {
        i64 cur = i < h.size() ? h[i] : 0;
        while (!st.empty() && h[st.back()] >= cur) {
            i64 height = h[st.back()];
            st.pop_back();
            std::size_t left = st.empty() ? 0 : st.back() + 1;
            best = std::max(best, height * static_cast<i64>(i - left));
        }
        st.push_back(i);
    }
    return {best};
}

Vec a10(const Args& a) {
    const Vec& h = a[0];
    std::size_t lo = 0, hi = h.size();  // half-open [lo, hi)
    i64 lmax = 0, rmax = 0, water = 0;
    while (lo < hi) {
        if (h[lo] <= h[hi - 1]) {
            lmax = std::max(lmax, h[lo]);
            water += lmax - h[lo++];
        } else {
            rmax = std::max(rmax, h[hi - 1]);
            water += rmax - h[--hi];
        }
    }
    return {water};
}

Vec a11(const Args& a) {
    std::unordered_map<i64, std::size_t> last;
    std::size_t start = 0, best = 0;
    for (std::size_t i = 0; i < a[0].size(); ++i) {
        auto it = last.find(a[0][i]);
        if (it != last.end() && it->second >= start) start = it->second + 1;
        last[a[0][i]] = i;
        best = std::max(best, i - start + 1);
    }
    return {static_cast<i64>(best)};
}

Vec a12(const Args& a) {
    const Vec &s = a[0], &t = a[1];
    Vec prev(t.size() + 1), cur(t.size() + 1);
    for (std::size_t j = 0; j <= t.size(); ++j) prev[j] = static_cast<i64>(j);
    for (std::size_t i = 1; i <= s.size(); ++i) {
        cur[0] = static_cast<i64>(i);
        for (std::size_t j = 1; j <= t.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (s[i - 1] != t[j - 1] ? 1 : 0)});
        std::swap(prev, cur);
    }
    return {prev[t.size()]};
}

Vec a13(const Args& a) {
    const Vec &s = a[0], &t = a[1];
    Vec prev(t.size() + 1, 0), cur(t.size() + 1, 0);
    for (i64 x : s) {
        for (std::size_t j = 0; j < t.size(); ++j)
            cur[j + 1] = x == t[j] ? prev[j] + 1 : std::max(prev[j + 1], cur[j]);
        std::swap(prev, cur);
    }
    return {prev[t.size()]};
}

Vec a14(const Args& a) {
    const auto cap = static_cast<std::size_t>(a[2][0]);
    Vec dp(cap + 1, 0);
    for (std::size_t i = 0; i < a[0].size(); ++i) {
        auto w = static_cast<std::size_t>(a[0][i]);
        for (std::size_t c = cap + 1; c-- > w;) dp[c] = std::max(dp[c], dp[c - w] + a[1][i]);
    }
    return {dp[cap]};
}

Vec a15(const Args& a) {
    const auto amount = static_cast<std::size_t>(a[1][0]);
    Vec dp(amount + 1, 0);
    dp[0] = 1;
    for (i64 c : a[0])
        for (auto x = static_cast<std::size_t>(c); x <= amount; ++x) dp[x] += dp[x - static_cast<std::size_t>(c)];
    return {dp[amount]};
}

Vec a16(const Args& a) {
    const auto amount = static_cast<std::size_t>(a[1][0]);
    const i64 inf = a[1][0] + 1;
    Vec dp(amount + 1, inf);
    dp[0] = 0;
    for (std::size_t x = 1; x <= amount; ++x)
        for (i64 c : a[0])
            if (static_cast<std::size_t>(c) <= x) dp[x] = std::min(dp[x], dp[x - static_cast<std::size_t>(c)] + 1);
    return {dp[amount] < inf ? dp[amount] : -1};
}

Vec a17(const Args& a) {
    const Vec& xs = a[0];
    std::size_t lo = 0, hi = xs.size();
    while (lo < hi) {
        std::size_t mid = lo + (hi - lo) / 2;  // no (lo + hi) overflow
        if (xs[mid] < a[1][0]) lo = mid + 1;
        else hi = mid;
    }
    return {static_cast<i64>(lo)};
}

Vec a18(const Args& a) {
    std::priority_queue<i64> low;
    std::priority_queue<i64, Vec, std::greater<>> high;
    Vec out;
    for (i64 x : a[0]) {
        if (low.empty() || x <= low.top()) low.push(x);
        else high.push(x);
        if (low.size() > high.size() + 1) {
            high.push(low.top());
            low.pop();
        } else if (high.size() > low.size()) {
            low.push(high.top());
            high.pop();
        }
        out.push_back(low.top());
    }
    return out;
}

Vec a19(const Args& a) {
    std::map<i64, i64> cnt;
    for (i64 x : a[0]) ++cnt[x];
    std::vector<std::pair<i64, i64>> v;  // (-count, value)
    for (auto [x, c] : cnt) v.emplace_back(-c, x);
    std::sort(v.begin(), v.end());
    Vec out;
    for (std::size_t i = 0; i < v.size() && static_cast<i64>(i) < a[1][0]; ++i) out.push_back(v[i].second);
    return out;
}

Vec a20(const Args& a) {
    const auto n = static_cast<std::size_t>(a[0][0]);
    std::vector<std::vector<std::pair<std::size_t, i64>>> adj(n);
    for (std::size_t i = 0; i + 2 < a[1].size(); i += 3)
        adj[static_cast<std::size_t>(a[1][i])].emplace_back(static_cast<std::size_t>(a[1][i + 1]), a[1][i + 2]);
    Vec dist(n, -1);
    using Item = std::pair<i64, std::size_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> pq;
    pq.emplace(0, static_cast<std::size_t>(a[0][1]));
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (dist[u] != -1) continue;
        dist[u] = d;
        for (auto [v, w] : adj[u])
            if (dist[v] == -1) pq.emplace(d + w, v);
    }
    return dist;
}

std::size_t find(std::vector<std::size_t>& p, std::size_t x) {
    while (p[x] != x) {
        p[x] = p[p[x]];
        x = p[x];
    }
    return x;
}

Vec a21(const Args& a) {
    const auto n = static_cast<std::size_t>(a[0][0]);
    std::vector<std::size_t> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = i;
    i64 comps = a[0][0];
    for (std::size_t i = 0; i + 1 < a[1].size(); i += 2) {
        auto ru = find(p, static_cast<std::size_t>(a[1][i])), rv = find(p, static_cast<std::size_t>(a[1][i + 1]));
        if (ru != rv) {
            p[ru] = rv;
            --comps;
        }
    }
    return {comps};
}

Vec a22(const Args& a) {
    const auto n = static_cast<std::size_t>(a[0][0]);
    std::vector<std::vector<std::size_t>> adj(n);
    std::vector<std::size_t> indeg(n, 0);
    for (std::size_t i = 0; i + 1 < a[1].size(); i += 2) {
        adj[static_cast<std::size_t>(a[1][i])].push_back(static_cast<std::size_t>(a[1][i + 1]));
        ++indeg[static_cast<std::size_t>(a[1][i + 1])];
    }
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>> pq;
    for (std::size_t i = 0; i < n; ++i)
        if (indeg[i] == 0) pq.push(i);
    Vec out;
    while (!pq.empty()) {
        auto u = pq.top();
        pq.pop();
        out.push_back(static_cast<i64>(u));
        for (auto v : adj[u])
            if (--indeg[v] == 0) pq.push(v);
    }
    return out.size() == n ? out : Vec{};
}

Vec a23(const Args& a) {
    const auto n = static_cast<std::size_t>(a[0][0]);
    std::vector<std::tuple<i64, std::size_t, std::size_t>> es;
    for (std::size_t i = 0; i + 2 < a[1].size(); i += 3)
        es.emplace_back(a[1][i + 2], static_cast<std::size_t>(a[1][i]), static_cast<std::size_t>(a[1][i + 1]));
    std::sort(es.begin(), es.end());
    std::vector<std::size_t> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = i;
    i64 total = 0;
    std::size_t used = 0;
    for (auto [w, u, v] : es) {
        auto ru = find(p, u), rv = find(p, v);
        if (ru != rv) {
            p[ru] = rv;
            total += w;
            ++used;
        }
    }
    return {used + 1 == n ? total : -1};
}

constexpr int DR[4] = {1, -1, 0, 0};
constexpr int DC[4] = {0, 0, 1, -1};

Vec a24(const Args& a) {
    const i64 rows = a[0][0], cols = a[0][1];
    const Vec& g = a[1];
    if (g.front() != 0 || g.back() != 0) return {-1};
    Vec dist(g.size(), -1);
    dist[0] = 0;
    std::queue<i64> q;
    q.push(0);
    while (!q.empty()) {
        i64 u = q.front();
        q.pop();
        for (int d = 0; d < 4; ++d) {
            i64 r = u / cols + DR[d], c = u % cols + DC[d];
            if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
            auto v = static_cast<std::size_t>(r * cols + c);
            if (g[v] == 0 && dist[v] == -1) {
                dist[v] = dist[static_cast<std::size_t>(u)] + 1;
                q.push(static_cast<i64>(v));
            }
        }
    }
    return {dist.back()};
}

Vec a25(const Args& a) {
    const auto cap = static_cast<std::size_t>(a[0][0]);
    std::list<std::pair<i64, i64>> order;  // front = most recent
    std::unordered_map<i64, std::list<std::pair<i64, i64>>::iterator> where;
    Vec out;
    const Vec& ops = a[1];
    for (std::size_t i = 0; i + 2 < ops.size(); i += 3) {
        i64 k = ops[i + 1];
        auto it = where.find(k);
        if (ops[i] == 0) {
            if (it == where.end()) out.push_back(-1);
            else {
                order.splice(order.begin(), order, it->second);
                out.push_back(it->second->second);
            }
        } else {
            if (it != where.end()) {
                it->second->second = ops[i + 2];
                order.splice(order.begin(), order, it->second);
            } else {
                order.emplace_front(k, ops[i + 2]);
                where[k] = order.begin();
                if (order.size() > cap) {
                    where.erase(order.back().first);
                    order.pop_back();
                }
            }
        }
    }
    return out;
}

Vec a26(const Args& a) {
    const Vec &text = a[0], &pat = a[1];
    std::vector<std::size_t> fail(pat.size(), 0);
    for (std::size_t i = 1, k = 0; i < pat.size(); ++i) {
        while (k && pat[i] != pat[k]) k = fail[k - 1];
        if (pat[i] == pat[k]) ++k;
        fail[i] = k;
    }
    Vec out;
    for (std::size_t i = 0, k = 0; i < text.size(); ++i) {
        while (k && text[i] != pat[k]) k = fail[k - 1];
        if (text[i] == pat[k]) ++k;
        if (k == pat.size()) {
            out.push_back(static_cast<i64>(i + 1 - k));
            k = fail[k - 1];
        }
    }
    return out;
}

Vec a27(const Args& a) {
    std::unordered_map<i64, i64> seen{{0, 1}};
    i64 s = 0, count = 0;
    for (i64 x : a[0]) {
        s += x;
        if (auto it = seen.find(s - a[1][0]); it != seen.end()) count += it->second;
        ++seen[s];
    }
    return {count};
}

Vec a28(const Args& a) {
    const auto n = static_cast<std::size_t>(a[0][0]);
    Vec tree(n + 1, 0);
    auto prefix = [&](std::size_t i) {
        i64 s = 0;
        for (; i > 0; i -= i & (~i + 1)) s += tree[i];
        return s;
    };
    Vec out;
    const Vec& ops = a[1];
    for (std::size_t j = 0; j + 2 < ops.size(); j += 3) {
        auto x = static_cast<std::size_t>(ops[j + 1]);
        if (ops[j] == 0) {
            for (std::size_t i = x + 1; i <= n; i += i & (~i + 1)) tree[i] += ops[j + 2];
        } else {
            out.push_back(prefix(static_cast<std::size_t>(ops[j + 2]) + 1) - prefix(x));
        }
    }
    return out;
}

Vec a29(const Args& a) {
    const i64 rows = a[0][0], cols = a[0][1];
    const Vec& g = a[1];
    std::vector<char> seen(g.size(), 0);
    i64 count = 0;
    for (std::size_t s = 0; s < g.size(); ++s) {
        if (!g[s] || seen[s]) continue;
        ++count;
        seen[s] = 1;
        std::vector<i64> st{static_cast<i64>(s)};
        while (!st.empty()) {
            i64 u = st.back();
            st.pop_back();
            for (int d = 0; d < 4; ++d) {
                i64 r = u / cols + DR[d], c = u % cols + DC[d];
                if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
                auto v = static_cast<std::size_t>(r * cols + c);
                if (g[v] && !seen[v]) {
                    seen[v] = 1;
                    st.push_back(static_cast<i64>(v));
                }
            }
        }
    }
    return {count};
}

Vec a30(const Args& a) {
    i64 lo = a[0][0], best = 0;
    for (i64 x : a[0]) {
        best = std::max(best, x - lo);
        lo = std::min(lo, x);
    }
    return {best};
}

Vec a31(const Args& a) {
    const Vec& p = a[0];
    i64 cash = 0, hold = -p[0];
    for (std::size_t i = 1; i < p.size(); ++i) {
        i64 c2 = std::max(cash, hold + p[i] - a[1][0]);
        hold = std::max(hold, cash - p[i]);
        cash = c2;
    }
    return {cash};
}

Vec a32(const Args& a) {
    // price level -> FIFO queue of resting quantities
    std::map<i64, std::deque<i64>, std::greater<>> bids;
    std::map<i64, std::deque<i64>> asks;
    i64 traded = 0;
    auto match = [&traded](auto& book, i64& qty, auto crosses) {
        while (qty > 0 && !book.empty() && crosses(book.begin()->first)) {
            auto& q = book.begin()->second;
            i64 fill = std::min(qty, q.front());
            traded += fill;
            qty -= fill;
            if ((q.front() -= fill) == 0) q.pop_front();
            if (q.empty()) book.erase(book.begin());
        }
    };
    const Vec& ops = a[0];
    for (std::size_t i = 0; i + 2 < ops.size(); i += 3) {
        i64 price = ops[i + 1], qty = ops[i + 2];
        if (ops[i] == 0) {
            match(asks, qty, [price](i64 p) { return p <= price; });
            if (qty > 0) bids[price].push_back(qty);
        } else {
            match(bids, qty, [price](i64 p) { return p >= price; });
            if (qty > 0) asks[price].push_back(qty);
        }
    }
    return {traded, bids.empty() ? -1 : bids.begin()->first, asks.empty() ? -1 : asks.begin()->first};
}

Vec a33(const Args& a) {
    const Vec &px = a[0], &qty = a[1];
    const auto k = static_cast<std::size_t>(a[2][0]);
    i64 num = 0, den = 0;
    Vec out;
    for (std::size_t i = 0; i < px.size(); ++i) {
        num += px[i] * qty[i];
        den += qty[i];
        if (i >= k) {
            num -= px[i - k] * qty[i - k];
            den -= qty[i - k];
        }
        if (i + 1 >= k) out.push_back(num / den);  // all inputs positive, so / is floor
    }
    return out;
}

const std::map<std::string, std::function<Vec(const Args&)>> kSolutions = {
    {"A01", a01}, {"A02", a02}, {"A03", a03}, {"A04", a04}, {"A05", a05}, {"A06", a06}, {"A07", a07},
    {"A08", a08}, {"A09", a09}, {"A10", a10}, {"A11", a11}, {"A12", a12}, {"A13", a13}, {"A14", a14},
    {"A15", a15}, {"A16", a16}, {"A17", a17}, {"A18", a18}, {"A19", a19}, {"A20", a20}, {"A21", a21},
    {"A22", a22}, {"A23", a23}, {"A24", a24}, {"A25", a25}, {"A26", a26}, {"A27", a27}, {"A28", a28},
    {"A29", a29}, {"A30", a30}, {"A31", a31}, {"A32", a32}, {"A33", a33},
};

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string id;
        std::size_t k = 0;
        if (!(in >> id >> k)) continue;
        Args args(k);
        for (auto& v : args) {
            std::size_t len = 0;
            in >> len;
            v.resize(len);
            for (auto& x : v) in >> x;
        }
        auto it = kSolutions.find(id);
        if (it == kSolutions.end()) {
            std::cerr << "unknown problem " << id << '\n';
            return 2;
        }
        Vec out = it->second(args);
        for (std::size_t i = 0; i < out.size(); ++i) std::cout << (i ? " " : "") << out[i];
        std::cout << '\n';
    }
    return 0;
}
