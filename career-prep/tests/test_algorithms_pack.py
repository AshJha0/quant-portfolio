"""Algorithms pack: Python and C++ solutions against slow, obviously-correct oracles.

For each problem a seeded generator draws small random cases (plus hand
edge cases). Each case goes to three places: the Python reference in
``career-prep/algorithms/solutions.py``, the C++ reference in
``solutions.cpp`` (compiled with sanitisers), and a brute-force oracle
written here independently. All three must agree, and the worked example
in the key must match the oracle.
"""

from __future__ import annotations

import heapq
import importlib.util
import itertools
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cxx_build import compile_cpp, run

PACK = Path(__file__).resolve().parents[1] / "algorithms"
_spec = importlib.util.spec_from_file_location("cp_alg_solutions", PACK / "solutions.py")
assert _spec and _spec.loader
sol = importlib.util.module_from_spec(_spec)
sys.modules["cp_alg_solutions"] = sol
_spec.loader.exec_module(sol)

Args = list[list[int]]
R = random.Random


# ---------------------------------------------------------------- oracles


def _o01(a: Args) -> list[int]:
    xs, (t,) = a
    return [sum(1 for i, j in itertools.combinations(range(len(xs)), 2) if xs[i] + xs[j] == t)]


def _o02(a: Args) -> list[int]:
    (xs,) = a
    return [max(sum(xs[i:j]) for i in range(len(xs)) for j in range(i + 1, len(xs) + 1))]


def _o03(a: Args) -> list[int]:
    (xs,) = a
    best = 0
    for mask in range(1 << len(xs)):
        sub = [xs[i] for i in range(len(xs)) if mask >> i & 1]
        if all(x < y for x, y in itertools.pairwise(sub)):
            best = max(best, len(sub))
    return [best]


def _o04(a: Args) -> list[int]:
    xs, (k,) = a
    return [max(xs[i : i + k]) for i in range(len(xs) - k + 1)]


def _o05(a: Args) -> list[int]:
    xs, (k,) = a
    return [sorted(xs)[k - 1]]


def _o06(a: Args) -> list[int]:
    (xs,) = a
    return [sum(1 for i, j in itertools.combinations(range(len(xs)), 2) if xs[i] > xs[j])]


def _o07(a: Args) -> list[int]:
    (flat,) = a
    # paint the closed integer intervals on a fine grid (2x to keep touching ends joined)
    covered = set()
    for s, e in zip(flat[::2], flat[1::2], strict=True):
        covered |= set(range(2 * s, 2 * e + 1))
    runs: list[list[int]] = []
    for p in sorted(covered):
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return [x // 2 for run in runs for x in run]


def _o08(a: Args) -> list[int]:
    (xs,) = a
    return [next((y for y in xs[i + 1 :] if y > x), -1) for i, x in enumerate(xs)]


def _o09(a: Args) -> list[int]:
    (h,) = a
    return [max([min(h[i:j]) * (j - i) for i in range(len(h)) for j in range(i + 1, len(h) + 1)], default=0)]


def _o10(a: Args) -> list[int]:
    (h,) = a
    return [sum(max(0, min(max(h[: i + 1]), max(h[i:])) - h[i]) for i in range(len(h)))]


def _o11(a: Args) -> list[int]:
    (xs,) = a
    n = len(xs)
    return [max((j - i for i in range(n) for j in range(i, n + 1) if len(set(xs[i:j])) == j - i), default=0)]


def _o12(a: Args) -> list[int]:
    s, t = a

    def d(i: int, j: int, memo: dict[tuple[int, int], int]) -> int:
        if i == 0 or j == 0:
            return i + j
        if (i, j) not in memo:
            memo[i, j] = min(
                d(i - 1, j, memo) + 1, d(i, j - 1, memo) + 1, d(i - 1, j - 1, memo) + (s[i - 1] != t[j - 1])
            )
        return memo[i, j]

    return [d(len(s), len(t), {})]


def _is_subseq(sub: tuple[int, ...], t: list[int]) -> bool:
    it = iter(t)
    return all(x in it for x in sub)


def _o13(a: Args) -> list[int]:
    s, t = a
    for r in range(len(s), 0, -1):
        if any(_is_subseq(c, t) for c in itertools.combinations(s, r)):
            return [r]
    return [0]


def _o14(a: Args) -> list[int]:
    w, v, (cap,) = a
    best = 0
    for mask in range(1 << len(w)):
        idx = [i for i in range(len(w)) if mask >> i & 1]
        if sum(w[i] for i in idx) <= cap:
            best = max(best, sum(v[i] for i in idx))
    return [best]


def _multisets(coins: list[int], amount: int) -> list[list[int]]:
    """All count vectors c with sum c_i * coins_i == amount."""
    if not coins:
        return [[]] if amount == 0 else []
    out = []
    for k in range(amount // coins[0] + 1):
        out += [[k, *rest] for rest in _multisets(coins[1:], amount - k * coins[0])]
    return out


def _o15(a: Args) -> list[int]:
    coins, (amount,) = a
    return [len(_multisets(coins, amount))]


def _o16(a: Args) -> list[int]:
    coins, (amount,) = a
    sols = _multisets(coins, amount)
    return [min(sum(s) for s in sols) if sols else -1]


def _o17(a: Args) -> list[int]:
    xs, (x,) = a
    return [next((i for i, y in enumerate(xs) if y >= x), len(xs))]


def _o18(a: Args) -> list[int]:
    (xs,) = a
    return [sorted(xs[: i + 1])[i // 2] for i in range(len(xs))]


def _o19(a: Args) -> list[int]:
    xs, (k,) = a
    vals = sorted(set(xs), key=lambda v: (-xs.count(v), v))
    return vals[:k]


def _o20(a: Args) -> list[int]:
    (n, src), e = a
    inf = 10**18
    dist = [inf] * n
    dist[src] = 0
    for _ in range(n):  # Bellman-Ford
        for u, v, w in zip(e[::3], e[1::3], e[2::3], strict=True):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return [d if d < inf else -1 for d in dist]


def _reach(n: int, e: list[int]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in zip(e[::2], e[1::2], strict=True):
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _o21(a: Args) -> list[int]:
    (n,), e = a
    adj = _reach(n, e)
    seen: set[int] = set()
    comps = 0
    for s in range(n):
        if s not in seen:
            comps += 1
            todo = [s]
            while todo:
                u = todo.pop()
                if u not in seen:
                    seen.add(u)
                    todo += adj[u]
    return [comps]


def _o22(a: Args) -> list[int]:
    (n,), e = a
    edges = set(zip(e[::2], e[1::2], strict=True))
    # lexicographically smallest valid permutation, found greedily by a different route:
    # repeatedly take the smallest vertex with no remaining incoming edge.
    left = set(range(n))
    out: list[int] = []
    while left:
        free = [v for v in sorted(left) if not any((u, v) in edges for u in left)]
        if not free:
            return []
        out.append(free[0])
        left.remove(free[0])
    return out


def _o23(a: Args) -> list[int]:
    (n,), e = a
    es = list(zip(e[::3], e[1::3], e[2::3], strict=True))
    best = -1
    for sub in itertools.combinations(es, n - 1):
        if _o21([[n], [x for u, v, _ in sub for x in (u, v)]]) == [1]:
            w = sum(x[2] for x in sub)
            best = w if best == -1 else min(best, w)
    return [best]


def _o24(a: Args) -> list[int]:
    (rows, cols), g = a
    if g[0] or g[-1]:
        return [-1]
    # Bellman-Ford style relaxation on the grid
    inf = 10**9
    d = [inf] * (rows * cols)
    d[0] = 0
    for _ in range(rows * cols):
        for u in range(rows * cols):
            if g[u] or d[u] == inf:
                continue
            r, c = divmod(u, cols)
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and not g[nr * cols + nc]:
                    d[nr * cols + nc] = min(d[nr * cols + nc], d[u] + 1)
    return [d[-1] if d[-1] < inf else -1]


def _o25(a: Args) -> list[int]:
    (cap,), ops = a
    store: dict[int, int] = {}
    stamp: dict[int, int] = {}
    out: list[int] = []
    for t, (op, k, v) in enumerate(zip(ops[::3], ops[1::3], ops[2::3], strict=True)):
        if op == 0:
            out.append(store.get(k, -1))
            if k in store:
                stamp[k] = t
        else:
            store[k] = v
            stamp[k] = t
            if len(store) > cap:
                old = min(store, key=stamp.__getitem__)
                del store[old], stamp[old]
    return out


def _o26(a: Args) -> list[int]:
    text, pat = a
    m = len(pat)
    return [i for i in range(len(text) - m + 1) if text[i : i + m] == pat]


def _o27(a: Args) -> list[int]:
    xs, (k,) = a
    n = len(xs)
    return [sum(1 for i in range(n) for j in range(i + 1, n + 1) if sum(xs[i:j]) == k)]


def _o28(a: Args) -> list[int]:
    (n,), ops = a
    arr = [0] * n
    out = []
    for op, x, y in zip(ops[::3], ops[1::3], ops[2::3], strict=True):
        if op == 0:
            arr[x] += y
        else:
            out.append(sum(arr[x : y + 1]))
    return out


def _o29(a: Args) -> list[int]:
    (rows, cols), g = a
    # label propagation until stable
    label = [i if g[i] else -1 for i in range(rows * cols)]
    changed = True
    while changed:
        changed = False
        for u in range(rows * cols):
            if label[u] < 0:
                continue
            r, c = divmod(u, cols)
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and label[nr * cols + nc] >= 0:
                    m = min(label[u], label[nr * cols + nc])
                    if m < label[u]:
                        label[u] = m
                        changed = True
    return [len({x for x in label if x >= 0})]


def _o30(a: Args) -> list[int]:
    (p,) = a
    return [max([0] + [p[j] - p[i] for i in range(len(p)) for j in range(i + 1, len(p))])]


def _o31(a: Args) -> list[int]:
    p, (fee,) = a
    n = len(p)

    def best(i: int, holding: bool, memo: dict[tuple[int, bool], int]) -> int:
        if i == n:
            return 0
        if (i, holding) not in memo:
            stay = best(i + 1, holding, memo)
            act = (p[i] - fee if holding else -p[i]) + best(i + 1, not holding, memo)
            memo[i, holding] = max(stay, act)
        return memo[i, holding]

    return [best(0, False, {})]


def _o32(a: Args) -> list[int]:
    (ops,) = a
    book: list[list[int]] = []  # [side, price, seq, qty], scanned linearly
    traded = 0
    for seq, (side, price, qty) in enumerate(zip(ops[::3], ops[1::3], ops[2::3], strict=True)):
        while qty:
            opp = [o for o in book if o[0] != side and (o[1] <= price if side == 0 else o[1] >= price)]
            if not opp:
                break
            o = min(opp, key=lambda o: (o[1] if side == 0 else -o[1], o[2]))
            fill = min(qty, o[3])
            traded += fill
            qty -= fill
            o[3] -= fill
            if o[3] == 0:
                book.remove(o)
        if qty:
            book.append([side, price, seq, qty])
    bids = [o[1] for o in book if o[0] == 0]
    asks = [o[1] for o in book if o[0] == 1]
    return [traded, max(bids, default=-1), min(asks, default=-1)]


def _o33(a: Args) -> list[int]:
    px, q, (k,) = a
    return [
        sum(p * x for p, x in zip(px[i : i + k], q[i : i + k], strict=True)) // sum(q[i : i + k])
        for i in range(len(px) - k + 1)
    ]


# ------------------------------------------------------------- generators


def _ints(r: R, n: int, lo: int, hi: int) -> list[int]:
    return [r.randint(lo, hi) for _ in range(n)]


def _g01(r: R) -> Args:
    return [_ints(r, r.randint(0, 9), -5, 5), [r.randint(-6, 6)]]


def _g_nonempty(r: R) -> Args:
    return [_ints(r, r.randint(1, 10), -9, 9)]


def _g04(r: R) -> Args:
    xs = _ints(r, r.randint(1, 10), -9, 9)
    return [xs, [r.randint(1, len(xs))]]


def _g07(r: R) -> Args:
    flat: list[int] = []
    for _ in range(r.randint(1, 6)):
        s = r.randint(0, 12)
        flat += [s, s + r.randint(0, 4)]
    return [flat]


def _g09(r: R) -> Args:
    return [_ints(r, r.randint(0, 9), 0, 7)]


def _g11(r: R) -> Args:
    return [_ints(r, r.randint(0, 10), 0, 4)]


def _g_two_seqs(r: R) -> Args:
    return [_ints(r, r.randint(0, 7), 0, 2), _ints(r, r.randint(0, 7), 0, 2)]


def _g14(r: R) -> Args:
    n = r.randint(0, 8)
    return [_ints(r, n, 1, 6), _ints(r, n, 0, 9), [r.randint(0, 15)]]


def _g_coins(r: R) -> Args:
    return [r.sample(range(1, 10), r.randint(1, 4)), [r.randint(0, 25)]]


def _g17(r: R) -> Args:
    return [sorted(_ints(r, r.randint(0, 10), 0, 9)), [r.randint(-1, 10)]]


def _g19(r: R) -> Args:
    xs = _ints(r, r.randint(1, 12), 0, 5)
    return [xs, [r.randint(1, len(set(xs)))]]


def _g20(r: R) -> Args:
    n = r.randint(1, 6)
    e: list[int] = []
    for _ in range(r.randint(0, 12)):
        e += [r.randrange(n), r.randrange(n), r.randint(0, 9)]
    return [[n, r.randrange(n)], e]


def _g_undirected(r: R) -> Args:
    n = r.randint(1, 8)
    e: list[int] = []
    for _ in range(r.randint(0, 8)):
        e += [r.randrange(n), r.randrange(n)]
    return [[n], e]


def _g22(r: R) -> Args:
    n = r.randint(1, 7)
    e: list[int] = []
    for _ in range(r.randint(0, 9)):
        u, v = r.randrange(n), r.randrange(n)
        if r.random() < 0.85 and u > v:  # mostly acyclic, sometimes not
            u, v = v, u
        if u != v or r.random() < 0.1:
            e += [u, v]
    return [[n], e]


def _g23(r: R) -> Args:
    n = r.randint(1, 6)
    e: list[int] = []
    for _ in range(r.randint(0, 9)):
        e += [r.randrange(n), r.randrange(n), r.randint(0, 9)]
    return [[n], e]


def _g_grid(r: R) -> Args:
    rows, cols = r.randint(1, 5), r.randint(1, 5)
    return [[rows, cols], [int(r.random() < 0.35) for _ in range(rows * cols)]]


def _g25(r: R) -> Args:
    ops: list[int] = []
    for _ in range(r.randint(1, 20)):
        ops += [r.randint(0, 1), r.randint(0, 5), r.randint(0, 99)]
    return [[r.randint(1, 3)], ops]


def _g26(r: R) -> Args:
    return [_ints(r, r.randint(0, 16), 0, 1), _ints(r, r.randint(1, 4), 0, 1)]


def _g27(r: R) -> Args:
    return [_ints(r, r.randint(0, 10), -3, 3), [r.randint(-3, 3)]]


def _g28(r: R) -> Args:
    n = r.randint(1, 8)
    ops: list[int] = []
    for _ in range(r.randint(1, 15)):
        if r.random() < 0.5:
            ops += [0, r.randrange(n), r.randint(-9, 9)]
        else:
            x = r.randrange(n)
            ops += [1, x, r.randint(x, n - 1)]
    return [[n], ops]


def _g30(r: R) -> Args:
    return [_ints(r, r.randint(1, 9), 1, 20)]


def _g31(r: R) -> Args:
    return [_ints(r, r.randint(1, 9), 1, 20), [r.randint(0, 4)]]


def _g32(r: R) -> Args:
    ops: list[int] = []
    for _ in range(r.randint(1, 15)):
        ops += [r.randint(0, 1), r.randint(98, 102), r.randint(1, 5)]
    return [ops]


def _g33(r: R) -> Args:
    n = r.randint(1, 10)
    return [_ints(r, n, 90, 110), _ints(r, n, 1, 9), [r.randint(1, n)]]


CASES: dict[str, tuple[Callable[[R], Args], Callable[[Args], list[int]], list[Args]]] = {
    "A01": (_g01, _o01, [[[], [0]], [[1, 1, 1], [2]]]),
    "A02": (_g_nonempty, _o02, [[[-3, -1, -2]], [[5]]]),
    "A03": (_g_nonempty, _o03, [[[2, 2, 2]], [[3, 1, 2]]]),
    "A04": (_g04, _o04, [[[4], [1]], [[1, 3, 3, 2], [4]]]),
    "A05": (_g04, _o05, [[[7, 7, 7], [2]]]),
    "A06": (_g_nonempty, _o06, [[[3, 2, 1]], [[1, 1]]]),
    "A07": (_g07, _o07, [[[1, 2, 2, 3]], [[5, 5]]]),
    "A08": (_g_nonempty, _o08, [[[2, 2, 3]]]),
    "A09": (_g09, _o09, [[[]], [[2, 1, 5, 6, 2, 3]]]),
    "A10": (_g09, _o10, [[[]], [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]]]),
    "A11": (_g11, _o11, [[[]], [[1, 1, 1]]]),
    "A12": (_g_two_seqs, _o12, [[[], []], [[1, 2], []]]),
    "A13": (_g_two_seqs, _o13, [[[], [1]]]),
    "A14": (_g14, _o14, [[[], [], [5]], [[6], [9], [5]]]),
    "A15": (_g_coins, _o15, [[[1, 2, 5], [5]], [[2], [3]]]),
    "A16": (_g_coins, _o16, [[[2], [3]], [[1, 3, 4], [6]]]),
    "A17": (_g17, _o17, [[[], [3]], [[1, 2, 2, 2, 3], [2]]]),
    "A18": (_g_nonempty, _o18, [[[5, 15, 1, 3]]]),
    "A19": (_g19, _o19, [[[1, 1, 2, 2, 3], [2]]]),
    "A20": (_g20, _o20, [[[3, 0], [0, 1, 4, 0, 2, 1, 2, 1, 2]]]),
    "A21": (_g_undirected, _o21, [[[1], []], [[4], [0, 1, 2, 3]]]),
    "A22": (_g22, _o22, [[[2], [0, 1, 1, 0]], [[3], [2, 0]], [[1], [0, 0]]]),
    "A23": (_g23, _o23, [[[1], []], [[3], [0, 1, 1]]]),
    "A24": (_g_grid, _o24, [[[1, 1], [0]], [[2, 2], [0, 1, 1, 0]]]),
    "A25": (_g25, _o25, [[[2], [1, 1, 10, 1, 2, 20, 0, 1, 0, 1, 3, 30, 0, 2, 0, 0, 3, 0]]]),
    "A26": (_g26, _o26, [[[0, 0, 0, 0], [0, 0]], [[1], [1, 1]]]),
    "A27": (_g27, _o27, [[[0, 0, 0], [0]]]),
    "A28": (_g28, _o28, [[[1], [1, 0, 0, 0, 0, 5, 1, 0, 0]]]),
    "A29": (_g_grid, _o29, [[[2, 2], [1, 0, 0, 1]]]),
    "A30": (_g30, _o30, [[[5, 4, 3]], [[7, 1, 5, 3, 6, 4]]]),
    "A31": (_g31, _o31, [[[1, 3, 2, 8, 4, 9], [2]]]),
    "A32": (_g32, _o32, [[[0, 100, 5, 1, 101, 3, 1, 99, 7]]]),
    "A33": (_g33, _o33, [[[100, 102, 101], [1, 3, 2], [2]]]),
}

N_RANDOM = 150


def _all_cases() -> list[tuple[str, Args]]:
    out: list[tuple[str, Args]] = []
    for pid, (gen, _, edges) in CASES.items():
        r = random.Random(f"alg-{pid}")
        out += [(pid, c) for c in edges]
        out += [(pid, gen(r)) for _ in range(N_RANDOM)]
    return out


def _encode(pid: str, args: Args) -> str:
    parts = [pid, str(len(args))]
    for arr in args:
        parts += [str(len(arr)), *map(str, arr)]
    return " ".join(parts)


QUESTION = re.compile(r"^- \*\*(A\d\d)\*\*", re.M)
KEY_EXAMPLE = re.compile(r"^- \*\*(A\d\d)\*\* .*?Example: `([^`]*)` -> `([^`]*)`", re.M)


def test_question_key_solution_and_check_sets_agree() -> None:
    q = QUESTION.findall((PACK / "questions.md").read_text("utf-8"))
    k = QUESTION.findall((PACK / "keys" / "answers.md").read_text("utf-8"))
    assert len(q) == len(set(q)) == 33
    assert set(q) == set(k) == set(sol.SOLUTIONS) == set(CASES)


def test_questions_file_holds_no_solutions() -> None:
    text = (PACK / "questions.md").read_text("utf-8")
    assert "Example:" not in text and "->" not in text
    for _, inp, _ in KEY_EXAMPLE.findall((PACK / "keys" / "answers.md").read_text("utf-8")):
        assert inp not in text


def _parse_example(s: str) -> Args:
    return [[int(x) for x in grp.split()] if grp.strip() else [] for grp in s.split("|")]


def test_every_key_example_matches_the_oracle() -> None:
    found = dict(
        (pid, (i, o)) for pid, i, o in KEY_EXAMPLE.findall((PACK / "keys" / "answers.md").read_text("utf-8"))
    )
    assert set(found) == set(CASES)
    for pid, (inp, outp) in found.items():
        args = _parse_example(inp)
        want = [int(x) for x in outp.split()]
        assert CASES[pid][1](args) == want, pid
        assert sol.SOLUTIONS[pid](args) == want, pid


@pytest.mark.parametrize("pid", sorted(CASES))
def test_python_solution_matches_oracle(pid: str) -> None:
    gen, oracle, edges = CASES[pid]
    r = random.Random(f"alg-{pid}")
    for args in edges + [gen(r) for _ in range(N_RANDOM)]:
        assert sol.SOLUTIONS[pid]([list(x) for x in args]) == oracle(args), (pid, args)


def test_cpp_solution_matches_python_on_every_case() -> None:
    exe = compile_cpp([PACK / "solutions.cpp"])
    cases = _all_cases()
    out = run(exe, "\n".join(_encode(pid, a) for pid, a in cases) + "\n").split("\n")
    assert len(out) == len(cases) + 1 and out[-1] == ""
    bad = []
    for (pid, args), line in zip(cases, out, strict=False):
        got = [int(x) for x in line.split()]
        if got != sol.SOLUTIONS[pid]([list(x) for x in args]):
            bad.append((pid, args, got))
    assert not bad, bad[:5]


def test_scale_large_inputs_run_in_the_stated_complexity() -> None:
    """Large cases the oracles cannot touch: Python vs C++ only, still exact."""
    r = random.Random("alg-scale")
    n = 20000
    xs = [r.randint(-(10**6), 10**6) for _ in range(n)]
    cases: list[tuple[str, Args]] = [
        ("A02", [xs]),
        ("A03", [xs]),
        ("A06", [xs]),
        ("A04", [xs, [137]]),
        ("A18", [xs[:5000]]),
        ("A27", [[r.randint(-2, 2) for _ in range(n)], [3]]),
    ]
    exe = compile_cpp([PACK / "solutions.cpp"])
    out = run(exe, "\n".join(_encode(p, a) for p, a in cases) + "\n").split("\n")
    for (pid, args), line in zip(cases, out, strict=False):
        assert [int(x) for x in line.split()] == sol.SOLUTIONS[pid](args), pid


def test_heap_based_kth_and_topo_cross_checks() -> None:
    # two independent extra checks on solutions whose oracles share an idea with them
    r = random.Random(5)
    for _ in range(200):
        xs = [r.randint(0, 50) for _ in range(r.randint(1, 30))]
        k = r.randint(1, len(xs))
        assert sol.SOLUTIONS["A05"]([xs, [k]]) == [heapq.nsmallest(k, xs)[-1]]
