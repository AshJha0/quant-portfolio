"""Reference Python solutions for the algorithms pack.

Every problem takes a list of integer arrays and returns one integer array,
so that the same seeded cases can be fed to this file, to the C++ solutions
in ``solutions.cpp`` and to the slow oracles in
``career-prep/tests/test_algorithms_pack.py``. The encoding of each
problem's input is stated in ``questions.md``.
"""

from __future__ import annotations

import heapq
from bisect import bisect_left
from collections import Counter, OrderedDict, deque
from collections.abc import Callable

Args = list[list[int]]


def a01_pair_count(a: Args) -> list[int]:
    xs, (target,) = a
    seen: Counter[int] = Counter()
    count = 0
    for x in xs:
        count += seen[target - x]
        seen[x] += 1
    return [count]


def a02_max_subarray(a: Args) -> list[int]:
    (xs,) = a
    best = cur = xs[0]
    for x in xs[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return [best]


def a03_lis(a: Args) -> list[int]:
    (xs,) = a
    tails: list[int] = []
    for x in xs:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return [len(tails)]


def a04_window_max(a: Args) -> list[int]:
    xs, (k,) = a
    dq: deque[int] = deque()
    out: list[int] = []
    for i, x in enumerate(xs):
        while dq and xs[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(xs[dq[0]])
    return out


def a05_kth_smallest(a: Args) -> list[int]:
    xs, (k,) = a
    arr = list(xs)
    lo, hi, want = 0, len(arr) - 1, k - 1
    while True:
        # Lomuto partition around the middle element (deterministic pivot)
        mid = (lo + hi) // 2
        arr[mid], arr[hi] = arr[hi], arr[mid]
        p = lo
        for i in range(lo, hi):
            if arr[i] < arr[hi]:
                arr[i], arr[p] = arr[p], arr[i]
                p += 1
        arr[p], arr[hi] = arr[hi], arr[p]
        if p == want:
            return [arr[p]]
        if p < want:
            lo = p + 1
        else:
            hi = p - 1


def a06_inversions(a: Args) -> list[int]:
    (xs,) = a

    def sort(v: list[int]) -> tuple[list[int], int]:
        if len(v) <= 1:
            return v, 0
        m = len(v) // 2
        left, cl = sort(v[:m])
        right, cr = sort(v[m:])
        merged: list[int] = []
        i = j = 0
        inv = cl + cr
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv += len(left) - i
                j += 1
        merged += left[i:] + right[j:]
        return merged, inv

    return [sort(list(xs))[1]]


def a07_merge_intervals(a: Args) -> list[int]:
    (flat,) = a
    iv = sorted(zip(flat[::2], flat[1::2], strict=True))
    out: list[list[int]] = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [v for pair in out for v in pair]


def a08_next_greater(a: Args) -> list[int]:
    (xs,) = a
    out = [-1] * len(xs)
    stack: list[int] = []
    for i, x in enumerate(xs):
        while stack and xs[stack[-1]] < x:
            out[stack.pop()] = x
        stack.append(i)
    return out


def a09_histogram(a: Args) -> list[int]:
    (h,) = a
    stack: list[int] = []
    best = 0
    for i in range(len(h) + 1):
        cur = h[i] if i < len(h) else 0
        while stack and h[stack[-1]] >= cur:
            height = h[stack.pop()]
            left = stack[-1] + 1 if stack else 0
            best = max(best, height * (i - left))
        stack.append(i)
    return [best]


def a10_rain(a: Args) -> list[int]:
    (h,) = a
    lo, hi = 0, len(h) - 1
    lmax = rmax = water = 0
    while lo <= hi:
        if h[lo] <= h[hi]:
            lmax = max(lmax, h[lo])
            water += lmax - h[lo]
            lo += 1
        else:
            rmax = max(rmax, h[hi])
            water += rmax - h[hi]
            hi -= 1
    return [water]


def a11_longest_distinct(a: Args) -> list[int]:
    (xs,) = a
    last: dict[int, int] = {}
    start = best = 0
    for i, x in enumerate(xs):
        if last.get(x, -1) >= start:
            start = last[x] + 1
        last[x] = i
        best = max(best, i - start + 1)
    return [best]


def a12_edit_distance(a: Args) -> list[int]:
    s, t = a
    prev = list(range(len(t) + 1))
    for i in range(1, len(s) + 1):
        cur = [i] + [0] * len(t)
        for j in range(1, len(t) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (s[i - 1] != t[j - 1]))
        prev = cur
    return [prev[-1]]


def a13_lcs(a: Args) -> list[int]:
    s, t = a
    prev = [0] * (len(t) + 1)
    for x in s:
        cur = [0]
        for j, y in enumerate(t):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return [prev[-1]]


def a14_knapsack(a: Args) -> list[int]:
    w, v, (cap,) = a
    dp = [0] * (cap + 1)
    for wi, vi in zip(w, v, strict=True):
        for c in range(cap, wi - 1, -1):
            dp[c] = max(dp[c], dp[c - wi] + vi)
    return [dp[cap]]


def a15_coin_ways(a: Args) -> list[int]:
    coins, (amount,) = a
    dp = [1] + [0] * amount
    for c in coins:
        for x in range(c, amount + 1):
            dp[x] += dp[x - c]
    return [dp[amount]]


def a16_coin_min(a: Args) -> list[int]:
    coins, (amount,) = a
    inf = amount + 1
    dp = [0] + [inf] * amount
    for x in range(1, amount + 1):
        for c in coins:
            if c <= x and dp[x - c] + 1 < dp[x]:
                dp[x] = dp[x - c] + 1
    return [dp[amount] if dp[amount] < inf else -1]


def a17_lower_bound(a: Args) -> list[int]:
    xs, (x,) = a
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return [lo]


def a18_running_median(a: Args) -> list[int]:
    (xs,) = a
    low: list[int] = []  # max-heap via negation
    high: list[int] = []
    out: list[int] = []
    for x in xs:
        if not low or x <= -low[0]:
            heapq.heappush(low, -x)
        else:
            heapq.heappush(high, x)
        if len(low) > len(high) + 1:
            heapq.heappush(high, -heapq.heappop(low))
        elif len(high) > len(low):
            heapq.heappush(low, -heapq.heappop(high))
        out.append(-low[0])
    return out


def a19_top_k(a: Args) -> list[int]:
    xs, (k,) = a
    cnt = Counter(xs)
    return [x for x, _ in sorted(cnt.items(), key=lambda p: (-p[1], p[0]))[:k]]


def a20_dijkstra(a: Args) -> list[int]:
    (n, src), edges = a
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in zip(edges[::3], edges[1::3], edges[2::3], strict=True):
        adj[u].append((v, w))
    dist = [-1] * n
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if dist[u] != -1:
            continue
        dist[u] = d
        for v, w in adj[u]:
            if dist[v] == -1:
                heapq.heappush(pq, (d + w, v))
    return dist


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def a21_components(a: Args) -> list[int]:
    (n,), edges = a
    parent = list(range(n))
    comps = n
    for u, v in zip(edges[::2], edges[1::2], strict=True):
        ru, rv = _find(parent, u), _find(parent, v)
        if ru != rv:
            parent[ru] = rv
            comps -= 1
    return [comps]


def a22_topo(a: Args) -> list[int]:
    (n,), edges = a
    indeg = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(edges[::2], edges[1::2], strict=True):
        adj[u].append(v)
        indeg[v] += 1
    pq = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(pq)
    out: list[int] = []
    while pq:
        u = heapq.heappop(pq)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(pq, v)
    return out if len(out) == n else []


def a23_mst(a: Args) -> list[int]:
    (n,), edges = a
    es = sorted(zip(edges[2::3], edges[::3], edges[1::3], strict=True))
    parent = list(range(n))
    total, used = 0, 0
    for w, u, v in es:
        ru, rv = _find(parent, u), _find(parent, v)
        if ru != rv:
            parent[ru] = rv
            total += w
            used += 1
    return [total if used == n - 1 else -1]


def a24_grid_path(a: Args) -> list[int]:
    (rows, cols), cells = a
    if cells[0] or cells[-1]:
        return [-1]
    dist = [-1] * (rows * cols)
    dist[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        r, c = divmod(u, cols)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = nr * cols + nc
                if not cells[v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
    return [dist[-1]]


def a25_lru(a: Args) -> list[int]:
    (cap,), ops = a
    cache: OrderedDict[int, int] = OrderedDict()
    out: list[int] = []
    for op, k, v in zip(ops[::3], ops[1::3], ops[2::3], strict=True):
        if op == 0:
            if k in cache:
                cache.move_to_end(k)
                out.append(cache[k])
            else:
                out.append(-1)
        else:
            cache[k] = v
            cache.move_to_end(k)
            if len(cache) > cap:
                cache.popitem(last=False)
    return out


def a26_kmp(a: Args) -> list[int]:
    text, pat = a
    fail = [0] * len(pat)
    k = 0
    for i in range(1, len(pat)):
        while k and pat[i] != pat[k]:
            k = fail[k - 1]
        if pat[i] == pat[k]:
            k += 1
        fail[i] = k
    out: list[int] = []
    k = 0
    for i, x in enumerate(text):
        while k and x != pat[k]:
            k = fail[k - 1]
        if x == pat[k]:
            k += 1
        if k == len(pat):
            out.append(i - k + 1)
            k = fail[k - 1]
    return out


def a27_subarray_sum(a: Args) -> list[int]:
    xs, (k,) = a
    seen: Counter[int] = Counter({0: 1})
    s = count = 0
    for x in xs:
        s += x
        count += seen[s - k]
        seen[s] += 1
    return [count]


def a28_fenwick(a: Args) -> list[int]:
    (n,), ops = a
    tree = [0] * (n + 1)

    def add(i: int, d: int) -> None:
        i += 1
        while i <= n:
            tree[i] += d
            i += i & -i

    def prefix(i: int) -> int:  # sum of [0, i)
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    out: list[int] = []
    for op, x, y in zip(ops[::3], ops[1::3], ops[2::3], strict=True):
        if op == 0:
            add(x, y)
        else:
            out.append(prefix(y + 1) - prefix(x))
    return out


def a29_islands(a: Args) -> list[int]:
    (rows, cols), cells = a
    seen = [False] * (rows * cols)
    count = 0
    for start in range(rows * cols):
        if cells[start] and not seen[start]:
            count += 1
            seen[start] = True
            stack = [start]
            while stack:
                r, c = divmod(stack.pop(), cols)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    v = nr * cols + nc
                    if 0 <= nr < rows and 0 <= nc < cols and cells[v] and not seen[v]:
                        seen[v] = True
                        stack.append(v)
    return [count]


def a30_one_trade(a: Args) -> list[int]:
    (p,) = a
    lo, best = p[0], 0
    for x in p[1:]:
        best = max(best, x - lo)
        lo = min(lo, x)
    return [best]


def a31_trades_with_fee(a: Args) -> list[int]:
    p, (fee,) = a
    cash, hold = 0, -p[0]
    for x in p[1:]:
        cash, hold = max(cash, hold + x - fee), max(hold, cash - x)
    return [cash]


def a32_matching(a: Args) -> list[int]:
    (ops,) = a
    bids: list[tuple[int, int, int]] = []  # (-price, seq, qty)
    asks: list[tuple[int, int, int]] = []  # (price, seq, qty)
    traded = 0
    for seq, (side, price, qty) in enumerate(zip(ops[::3], ops[1::3], ops[2::3], strict=True)):
        if side == 0:  # buy
            while qty and asks and asks[0][0] <= price:
                ap, s, aq = heapq.heappop(asks)
                fill = min(qty, aq)
                traded += fill
                qty -= fill
                if aq > fill:
                    heapq.heappush(asks, (ap, s, aq - fill))
            if qty:
                heapq.heappush(bids, (-price, seq, qty))
        else:
            while qty and bids and -bids[0][0] >= price:
                bp, s, bq = heapq.heappop(bids)
                fill = min(qty, bq)
                traded += fill
                qty -= fill
                if bq > fill:
                    heapq.heappush(bids, (bp, s, bq - fill))
            if qty:
                heapq.heappush(asks, (price, seq, qty))
    best_bid = -bids[0][0] if bids else -1
    best_ask = asks[0][0] if asks else -1
    return [traded, best_bid, best_ask]


def a33_vwap_window(a: Args) -> list[int]:
    px, qty, (k,) = a
    num = den = 0
    out: list[int] = []
    for i, (p, q) in enumerate(zip(px, qty, strict=True)):
        num += p * q
        den += q
        if i >= k:
            num -= px[i - k] * qty[i - k]
            den -= qty[i - k]
        if i >= k - 1:
            out.append(num // den)
    return out


SOLUTIONS: dict[str, Callable[[Args], list[int]]] = {
    name[:3].upper(): fn for name, fn in sorted(globals().items()) if name[:1] == "a" and name[1:3].isdigit()
}
