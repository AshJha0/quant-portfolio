# Mock round 4: answer key

Questions are in [`../round-4.md`](../round-4.md). Each graded answer
is recomputed by [`../verify.py`](../verify.py).

## Graded

- **M4.1** `89`. f(n) = f(n - 1) + f(n - 2) with f(1) = 1, f(2) = 2: the Fibonacci numbers, f(10) = 89.
- **M4.2** `20`. Each probe halves the remaining range; 2^20 = 1,048,576 > 1,000,000, and 2^19 is not enough.
- **M4.3** `8`. 3 beats 1, 1, 2; 4 beats 1, 2; 5 beats 2; 9 beats 2, 6. A merge sort counts them in O(n log n).
- **M4.4** `10`. Capacity goes 1, 2, 4, ..., 1024: ten doublings. Total copies are below 2n, so appends are amortised O(1).

## Discussion rubric

- **M4.5** Two heaps with lazy deletion, or an order-statistic tree, giving O(log N) per update. Strong answers state the complexity and test against a brute-force sort on random input.
- **M4.6** Measure before changing: profile, check allocation and I/O, look at the distribution of latency, not the mean. Strong answers form a hypothesis and confirm it with a benchmark that fails before the fix.
