# Mock rounds

Six timed rounds, one per role family. Each has four graded questions with a
single numeric answer and two discussion questions, in the proportions a
first technical round usually has. The keys are kept apart in
[`keys/`](keys/) so you can sit a round cold; do not open them until the
timer stops.

| Round | For | Time | Questions |
| --- | --- | --- | --- |
| 1 | Quant researcher | 45 min | [round-1.md](round-1.md) |
| 2 | Pricing quant, strats | 45 min | [round-2.md](round-2.md) |
| 3 | Risk quant | 40 min | [round-3.md](round-3.md) |
| 4 | Quant developer | 45 min | [round-4.md](round-4.md) |
| 5 | Low-latency developer | 40 min | [round-5.md](round-5.md) |
| 6 | Trader, execution, market making | 40 min | [round-6.md](round-6.md) |

## How to sit a round

Set a timer for the stated time. Work on paper, no calculator, and say your
reasoning out loud (record yourself if you can; it is the closest thing to a
panel). Give a number for every graded question, even a rough one: a
bounded estimate with its assumption scores better than a blank. Spend the
last ten minutes on the two discussion questions.

## Debrief

When the timer stops, mark the graded answers against the key for the
round, then read the rubric for the discussion questions:
[round 1](keys/round-1.md), [round 2](keys/round-2.md),
[round 3](keys/round-3.md), [round 4](keys/round-4.md),
[round 5](keys/round-5.md), [round 6](keys/round-6.md).

Score yourself in three columns: correct, correct method but arithmetic
slip, and wrong method. Only the third column needs more study; the study
paths in [`../paths/`](../paths/) say which pack covers each topic. Every
graded answer is recomputed by [`verify.py`](verify.py), and
`career-prep/tests/test_final_packs.py` fails if a key and its check
disagree.
