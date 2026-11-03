# Algorithms and data structures: answer key

Questions are in [`../questions.md`](../questions.md). Each entry gives the
idea, the cost (n is the input length unless stated), and a worked example
in the pack's input encoding (arrays separated by `|`). The full code is in
[`../solutions.py`](../solutions.py) and [`../solutions.cpp`](../solutions.cpp).

Every example below is recomputed by the slow oracle in
`career-prep/tests/test_algorithms_pack.py`, and both solutions are compared
with that oracle on about 150 seeded random cases per problem. If this
prose and the test ever disagree, the test is the arbiter. Standard
references for the ideas: Cormen, Leiserson, Rivest and Stein,
*Introduction to Algorithms* (3rd ed., 2009); Sedgewick and Wayne,
*Algorithms* (4th ed., 2011). The problems and their wording are original.

## Arrays, hashing and two pointers

- **A01** Walk once, keeping a count of each value seen so far; at each `x` add `seen[t - x]`. O(n) time, O(n) space. The sorted two-pointer method is O(n log n) and needs care with duplicates. Example: `1 5 3 3 2 4 | 6` -> `3` (1+5, 3+3, 2+4).
- **A02** Kadane: the best subarray ending at `i` either extends the best ending at `i - 1` or starts fresh, `cur = max(x, cur + x)`. O(n), O(1). Start from `xs[0]`, not 0, or an all-negative input returns 0. Example: `-2 1 -3 4 -1 2 1 -5 4` -> `6`.
- **A03** Patience sorting: `tails[L]` is the smallest tail of an increasing subsequence of length `L + 1`; each value replaces the first tail `>= x` (`lower_bound`, which gives strictness). O(n log n). Example: `10 9 2 5 3 7 101 18` -> `4` (2, 3, 7, 18).
- **A04** Monotonic deque of indices whose values decrease from front to back. Pop smaller values from the back before pushing; drop the front when it leaves the window. Each index enters and leaves once, so O(n). Example: `1 3 -1 -3 5 3 6 7 | 3` -> `3 3 5 5 6 7`.
- **A05** Quickselect: partition, then recurse into one side only. Expected O(n), worst O(n^2) with bad pivots; median-of-medians gives O(n) worst case. `std::nth_element` in C++. Example: `7 10 4 3 20 15 | 3` -> `7`.
- **A06** Merge sort; when an element of the right half is taken before the remaining `m - i` elements of the left half, it forms `m - i` inversions. O(n log n). A Fenwick tree over ranks also works. Example: `2 4 1 3 5` -> `3` ((2,1), (4,1), (4,3)).
- **A07** Sort by start; extend the last merged interval while `s <= end`. O(n log n). Example: `1 3 8 10 2 6 10 12` -> `1 6 8 12`.
- **A08** Monotonic stack of indices still waiting for a larger value; a new value resolves every smaller one on top. O(n). Example: `2 1 2 4 3` -> `4 2 4 -1 -1`.
- **A09** Monotonic increasing stack. When a bar is popped, the new top and the current index are its nearest smaller bars on each side, so its widest rectangle is known. A sentinel 0 at the end flushes the stack. O(n). Example: `2 1 5 6 2 3` -> `10` (bars 5 and 6, width 2).
- **A10** Two pointers. Water over a bar is `min(max left, max right) - h`. Move the lower side inward; its own running maximum is the binding one. O(n), O(1). Example: `0 1 0 2 1 0 1 3 2 1 2 1` -> `6`.
- **A11** Sliding window with the last index of each value; on a repeat inside the window move the start just past it. O(n). Example: `1 2 3 1 2 3 4 1` -> `4`.

## Dynamic programming

- **A12** `d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + [s_i != t_j])`, with `d[i][0] = i`, `d[0][j] = j`. O(nm) time, O(m) space with two rows. Example: `1 2 3 4 5 6 | 1 3 4 7 6` -> `2` (delete 2, substitute 5 by 7).
- **A13** `L[i][j] = L[i-1][j-1] + 1` on a match, else `max(L[i-1][j], L[i][j-1])`. O(nm). Example: `1 2 3 2 4 1 2 | 2 4 3 1 2 1` -> `4`.
- **A14** `dp[c]` = best value within capacity `c`; for each item loop `c` downwards so the item is used at most once. O(nC), pseudo-polynomial (the problem is NP-hard in general). Example: `1 3 4 5 | 1 4 5 7 | 7` -> `9` (weights 3 and 4).
- **A15** Loop coins on the outside, amounts inside, `dp[x] += dp[x - c]`: each combination is counted once. Swapping the loops counts ordered sequences instead. O(len(coins) * amount). Example: `1 2 5 | 5` -> `4`.
- **A16** `dp[x] = 1 + min over c of dp[x - c]`. O(len(coins) * amount). Greedy takes 4 + 1 + 1 (three coins) for 6, but 3 + 3 uses two: greedy is only safe for "canonical" coin systems. Example: `1 3 4 | 6` -> `2`.

## Search, heaps and order statistics

- **A17** Half-open `[lo, hi)`, `mid = lo + (hi - lo) / 2` (the C++ form avoids the `lo + hi` overflow that broke many published binary searches; Bloch, "Nearly all binary searches and mergesorts are broken", Google Research blog, 2006). O(log n). Example: `1 2 4 4 4 7 | 4` -> `2`.
- **A18** Two heaps: a max-heap for the lower half, a min-heap for the upper half, sizes differing by at most one, lower half never smaller. The lower median is the top of the max-heap. O(log n) per value. Example: `5 15 1 3 8` -> `5 5 5 3 5`.
- **A19** Count with a hash map, then sort by `(-count, value)`: O(n + u log u) for `u` distinct values; a size-`k` heap gives O(n + u log k); bucket sort by count gives O(n). Example: `4 1 1 2 2 2 3 4 | 2` -> `2 1`.

## Graphs

- **A20** Dijkstra with a binary heap and lazy deletion (skip a popped vertex already settled). O((V + E) log V). Correct only for non-negative weights; with negative weights use Bellman-Ford, which is what the test's oracle runs. Example: `5 0 | 0 1 4 0 2 1 2 1 2 1 3 1 2 3 5` -> `0 3 1 4 -1`.
- **A21** Union-find with path compression (and union by rank for the full bound): each union that joins two roots removes one component. Nearly O(E + V) (inverse Ackermann). Example: `5 | 0 1 1 2 3 4` -> `2`.
- **A22** Kahn's algorithm with a min-heap instead of a queue gives the lexicographically smallest order. If fewer than `n` vertices come out, the rest lie on or behind a cycle. O((V + E) log V). Example: `4 | 3 1 2 1 1 0` -> `2 3 1 0`.
- **A23** Kruskal: sort edges by weight, add each one that joins two different union-find sets. The cut property makes it optimal. O(E log E). Disconnected means fewer than `n - 1` edges were added. Example: `4 | 0 1 1 1 2 2 0 2 3 2 3 4 0 3 5` -> `7` (1 + 2 + 4).
- **A24** Breadth-first search: in an unweighted graph the first time a cell is reached is by a shortest path. O(rows * cols). Example: `3 3 | 0 0 0 1 1 0 0 0 0` -> `4`.
- **A29** Flood fill (DFS with an explicit stack, so a large grid does not overflow the call stack) from each unvisited land cell; count the starts. O(rows * cols). Example: `3 4 | 1 1 0 0 0 1 0 1 1 0 0 1` -> `3`.

## Design, strings and ranges

- **A25** Hash map from key to a node of a doubly linked list ordered by recency (`OrderedDict` in Python, `std::list` plus `splice` in C++). Get and put move the node to the front; overflow evicts the back. All O(1). Example: `2 | 1 1 10 1 2 20 0 1 0 1 3 30 0 2 0 0 3 0 0 1 0` -> `10 -1 30 10` (putting 3 evicts 2, because the get made 1 recent).
- **A26** Knuth-Morris-Pratt: the failure table gives, for each prefix of the pattern, the length of its longest proper border; on a mismatch fall back along it instead of rewinding the text. O(n + m). Overlapping matches are found. Example: `1 0 1 0 1 1 0 1 | 1 0 1` -> `0 2 5`.
- **A27** Prefix sums with a hash count: a subarray `(i, j]` sums to `k` exactly when `S_j - S_i = k`. Seed the count with `S = 0`. O(n). The sliding window only works for non-negative values. Example: `1 2 3 -2 2 | 3` -> `4`.
- **A28** Fenwick (binary indexed) tree: index `i` covers the last `i & -i` positions. Update and prefix query each touch O(log n) nodes; range sum is a difference of two prefixes (Fenwick, "A new data structure for cumulative frequency tables", Software: Practice and Experience 24(3), 1994). Example: `5 | 0 1 3 0 3 4 1 0 4 0 1 -1 1 1 3` -> `7 6`.

## Market problems

- **A30** Track the lowest price so far; the best sale today is `p - low`. O(n). Example: `7 1 5 3 6 4` -> `5`.
- **A31** Two states per day: `cash` (flat) and `hold` (long one unit). `cash' = max(cash, hold + p - fee)`, `hold' = max(hold, cash - p)`. O(n), O(1). Example: `1 3 2 8 4 9 | 2` -> `8` (buy 1 sell 8, buy 4 sell 9: 5 + 3).
- **A32** Two sorted maps from price to a FIFO queue of resting quantities (bids descending, asks ascending), or two heaps keyed by `(price, arrival)`. An incoming order consumes the best level while it crosses, oldest first, then rests. O(log L) per level touched, plus O(1) per fill. Production books add order ids, cancels and an intrusive list per level; see the C++ matching engine in this repository. Example: `1 101 3 1 102 2 0 100 5 0 102 4 1 99 6` -> `9 -1 99`.
- **A33** Keep running sums of `p * q` and `q` over the window; add the new trade and subtract the one leaving. O(n). Use integer (or fixed-point) prices so the subtraction does not accumulate rounding error. Example: `100 102 101 99 | 1 3 2 4 | 2` -> `101 101 99`.
