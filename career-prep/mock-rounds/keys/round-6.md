# Mock round 6: answer key

Questions are in [`../round-6.md`](../round-6.md). Each graded answer
is recomputed by [`../verify.py`](../verify.py).

## Graded

- **M6.1** `1253/125`. (1,000 + 3,012 + 6,012) / 1,000 = 10.024.
- **M6.2** `195/2`. Market volume is 2,000,000 / 390 a minute, your share 10% of it, so 50,000 / (200,000 / 390) = 97.5 minutes.
- **M6.3** `13`. 6 plus the expected value of two unseen dice, 7.
- **M6.4** `2`. Realised 600 x 0.03 = 18; unrealised 400 x (99.98 - 100.02) = -16; total 2.

## Discussion rubric

- **M6.5** Wide enough to cover the uncertainty in your estimate and the chance the other side knows more, narrow enough to trade. Strong answers update the quote after each fill (Glosten-Milgrom thinking).
- **M6.6** Compare with a benchmark of no participation: interval VWAP, mark-outs after your fills, and similar stocks on the day. Strong answers separate impact from drift and say what they would change in the schedule.
