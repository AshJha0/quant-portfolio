# Algorithms and data structures: knowledge check (questions only)

Thirty-three problems, written fresh for this repository. They cover the
patterns that come up again and again in engineering rounds at trading
firms: hashing, two pointers, monotonic stacks and queues, heaps, binary
search, dynamic programming, graphs and union-find, string matching, and
two small market problems (an order book and a rolling VWAP). The key,
with complexities and a worked example for each, is in
[`keys/answers.md`](keys/answers.md).

How to work a problem: state the brute force first and its cost, then find
the better idea, then write it in Python or C++ in under 25 minutes. Then
compare with [`solutions.py`](solutions.py) and
[`solutions.cpp`](solutions.cpp).

How it is checked: `career-prep/tests/test_algorithms_pack.py` draws about
150 seeded random small cases per problem plus hand-picked edge cases, and
requires the Python solution, the C++ solution (built with
`-Wall -Wextra -Werror` and the address and undefined-behaviour sanitisers)
and a slow brute-force oracle written separately in the test to agree on
every one. It also runs large inputs through both fast solutions.

Input convention: every problem takes a list of integer arrays and returns
one integer array, so one test harness can drive all three implementations.
"Sequences of symbols" are integer arrays. Graph vertices are numbered
`0..n-1`; edge lists are flattened (`u0 v0 u1 v1 ...`, or with a weight,
`u0 v0 w0 ...`). A grid of `rows x cols` cells is one flat row-major array.

## Arrays, hashing and two pointers

- **A01** Given an array `xs` and a target `t`, count the pairs of positions `i < j` with `xs[i] + xs[j] == t`. Do it in one pass.
- **A02** Largest sum of a non-empty contiguous subarray (values may all be negative).
- **A03** Length of the longest strictly increasing subsequence. Aim for better than quadratic.
- **A04** Sliding window maximum: for window length `k`, output the maximum of each of the `n - k + 1` windows.
- **A05** The `k`-th smallest element (1-based) of an unsorted array, in expected linear time.
- **A06** Count inversions: pairs `i < j` with `xs[i] > xs[j]`.
- **A07** Merge overlapping closed intervals `[s, e]` (given flat as `s0 e0 s1 e1 ...`, unsorted). Intervals that share an endpoint merge. Output sorted and flat.
- **A08** Next greater element: for each position, the first later value strictly larger, or `-1`.
- **A09** Largest rectangle under a histogram of non-negative bar heights (0 for no bars).
- **A10** Trapping rain water: total water held between bars of the given heights.
- **A11** Length of the longest contiguous run with no repeated value.

## Dynamic programming

- **A12** Edit distance (insert, delete, substitute, each cost 1) between two sequences.
- **A13** Length of the longest common subsequence of two sequences.
- **A14** 0/1 knapsack: weights `w`, values `v`, capacity `C`. Best total value.
- **A15** Number of ways to make `amount` from unlimited coins of the given distinct denominations (order does not matter; `amount = 0` has one way).
- **A16** Fewest coins to make `amount`, or `-1` if impossible. Why does the greedy method fail for coins `1, 3, 4`?

## Search, heaps and order statistics

- **A17** Lower bound: in a sorted array, the first index whose value is `>= x` (the length if none). Write it without integer overflow in C++.
- **A18** Running median: after each new value, output the lower median of everything seen so far.
- **A19** The `k` most frequent values, most frequent first, ties broken by the smaller value.

## Graphs

- **A20** Single-source shortest paths with non-negative weights on a directed graph. Input `[n, src]` and the weighted edge list. Output the distance to each vertex, `-1` if unreachable.
- **A21** Number of connected components of an undirected graph `[n]` with the given edges.
- **A22** Topological order of a directed graph `[n]`: output the lexicographically smallest valid order, or an empty array if there is a cycle.
- **A23** Weight of a minimum spanning tree of an undirected weighted graph `[n]`, or `-1` if the graph is disconnected.
- **A24** Shortest path length, moving up, down, left or right, from the top-left to the bottom-right cell of a grid (`1` = wall), or `-1`. Input `[rows, cols]` and the cells.
- **A29** Number of islands (4-connected groups of `1` cells) in a grid. Input `[rows, cols]` and the cells.

## Design, strings and ranges

- **A25** LRU cache of capacity `cap`. Operations come as triples `op key value`: `op = 0` is `get(key)` (output the value or `-1`), `op = 1` is `put(key, value)`. A `get` hit or a `put` makes a key most recent. Each operation must be O(1).
- **A26** All start positions of a pattern in a text, in linear time.
- **A27** Number of contiguous subarrays summing to exactly `k` (values may be negative).
- **A28** Point updates and range sums on an array of length `n`, initially zero. Operations are triples: `0 i d` adds `d` at `i`; `1 l r` outputs the sum over `[l, r]`. Each operation in O(log n).

## Market problems

- **A30** Given daily prices, the best profit from at most one buy followed by one later sell (0 if none is profitable).
- **A31** Same prices, now any number of non-overlapping round trips, each paying a fee `fee` once on the sale. Best profit.
- **A32** A price-time priority limit order book. Orders come as triples `side price qty` (`side 0` buy, `1` sell). An incoming order trades against the best opposite price first, and within a price against the oldest order first, as long as prices cross; any rest joins the book. Output `[total quantity traded, best bid, best ask]` at the end (`-1` for an empty side).
- **A33** Rolling VWAP: given trade prices, quantities and a window `k`, output for each full window of `k` consecutive trades the volume-weighted average price, rounded down to an integer.
