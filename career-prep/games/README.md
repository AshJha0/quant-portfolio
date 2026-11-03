# Games: mental maths and market making

- `mental_maths.py`: a seeded drill of two-digit products, squares,
  percentages, fractions to decimals and simple expected values. Same seed,
  same drill, so two people can race on one sheet.
  `python mental_maths.py --seed 7 --n 20` prints questions then answers;
  add `--interactive` to type answers and get a score and time.
- `market_making.py`: quote a bid and ask on the sum of three dice against a
  counterparty who has seen one die. Shows adverse selection in numbers:
  a zero-width quote at 10.5 loses 1.5 a round on average; a quote of 8 / 13
  is never hit. `python market_making.py --rounds 10 --seed 1`.

The checks in `career-prep/tests/test_games.py` recompute every drill
answer from the question text with a separate parser, enumerate the 216
dice outcomes for E[V] = 10.5 and Var[V] = 35/4, and simulate the two
quoting rules above.
