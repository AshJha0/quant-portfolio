# Case and behavioural: answer key

Questions are in [`../questions.md`](../questions.md). Graded answers are in
backticks; [`../verify.py`](../verify.py) recomputes each one by a
different route.

## Trading cases

- **C01** `636.830651176`. Short calls have delta -1,000 N(d1) with
  d1 = (0.05 + 0.02) / 0.2 = 0.35, N(d1) = 0.63683. Buy 636.83 shares (637
  in practice). Say also that this hedge only holds for small moves: the
  position is short gamma.
- **C02** `0.572942450071`. Parity says C - P = S - K e^{-rT} = 4.8771; the
  market has C - P = 5.45, so the call is rich. Sell the call, buy the put
  and the stock, borrow K e^{-rT}: you receive 10.45 + 95.1229 and pay
  5.00 + 100, netting 0.5729 today, and at expiry the position is worth
  exactly zero whatever the stock does.
- **C03** `1/10`. Kelly: f = p - q = 0.55 - 0.45. Growth at that stake is
  0.55 ln 1.1 + 0.45 ln 0.9, about 0.5% a bet; staking twice as much gives
  roughly zero growth.
- **C04** `501/5`. The ask equals E[value | a buy] (Glosten and Milgrom
  1985). P(buy | 101) = 0.2 + 0.8 x 0.5 = 0.6 and P(buy | 99) = 0.4, so
  P(101 | buy) = 0.6, and the ask is 0.6 x 101 + 0.4 x 99 = 100.2.
- **C05** `1000`. Ten slices of 10,000: 10 x 10^-6 x 10^8 = 1,000. Halving
  the slice size halves the cost, which is why impact-only schedules spread
  orders out; risk aversion pulls the other way (Almgren and Chriss 2000).
- **C06** `75022.4646909`. Portfolio sigma = 20,000 sqrt(1 + 1 + 2 x 0.3)
  = 20,000 sqrt(2.6); times z_0.99 = 2.3263. Compare with 93,054 if the two
  were perfectly correlated.

## What the behavioural prompts test

There is no key, but there is a rubric. A strong answer is specific (a real
project, a number), owns the mistake without blame, says what changed
afterwards (a test, a check, a process), and is short: two minutes, then
stop and let the panel ask. For B05 in particular, the best answers name the
evidence that changed their mind, which is the habit a research desk most
wants to see.
