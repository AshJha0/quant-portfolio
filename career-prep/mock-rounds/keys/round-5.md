# Mock round 5: answer key

Questions are in [`../round-5.md`](../round-5.md). Each graded answer
is recomputed by [`../verify.py`](../verify.py).

## Graded

- **M5.1** `24`. a at 0, padding to 8, b at 8, c at 16, padding to a multiple of 8: 24.
- **M5.2** `16`. b at 0, c at 8, a at 12, padded to 16. Order members by decreasing alignment.
- **M5.3** `125`. 8,000 bytes / 64.
- **M5.4** `904`. 5000 mod 1024 = 5000 - 4 x 1024 = 904. The mask works only because the capacity is a power of two.

## Discussion rubric

- **M5.5** Two cores writing different variables on the same cache line force the line to bounce between them. Put head and tail on separate lines (alignas(64)) and cache the other side's index locally.
- **M5.6** Release on the producer's store publishes the slot write; acquire on the consumer's load sees it. That is all the queue needs, and on x86 it compiles to plain moves, while seq_cst stores need a fence.
