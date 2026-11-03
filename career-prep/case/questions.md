# Case and behavioural: knowledge check (questions only)

Six trading cases with a numeric answer, then six behavioural prompts. The
cases are the short, talk-it-through problems a desk uses to see how you
reason under a clock: state the assumption, get a number, say what would
change it. Answers are in [`keys/answers.md`](keys/answers.md); each graded
answer is recomputed by [`verify.py`](verify.py) by a different route and
`career-prep/tests/test_final_packs.py` fails if they disagree.

Allow 10 minutes a case. Reference texts: Hull, *Options, Futures, and
Other Derivatives*; Glosten and Milgrom (1985); Kelly (1956); Almgren and
Chriss (2000).

## Trading cases

- **C01** You sell 1,000 one-year at-the-money calls on a stock at 100
  (rate 5% continuously compounded, volatility 20%, no dividends). How many
  shares do you buy to be delta-neutral? Give the exact number before
  rounding.
- **C02** The same stock and rate. A one-year call struck at 100 trades at
  10.45 and the put at 5.00. What riskless profit per contract does the
  cheapest arbitrage lock in today, in present-value terms?
- **C03** A repeated bet wins with probability 0.55 and pays even money.
  What fraction of capital should you stake each time to maximise
  long-run growth?
- **C04** You make a market in an asset worth 99 or 101 with equal
  probability. One trader in five knows the value and buys only if it is
  101; the rest buy or sell at random. Where do you set the ask so that you
  break even on a buy order?
- **C05** You must buy 100,000 shares in 10 equal slices. Temporary impact
  costs 10^-6 dollars a share for every share in the same slice (so a slice
  of n shares costs 10^-6 n^2). What is the total impact cost in dollars?
- **C06** Two positions of $1m each, daily volatility 2% each, correlation
  0.3, normal P&L. What is the one-day 99% VaR of the pair, in dollars?

## Behavioural prompts (not graded)

- **B01** Tell me about a time a model or system you built gave a wrong
  answer in use. How did you find out, and what did you change?
- **B02** Describe a disagreement with a trader or senior colleague about a
  number. How was it settled?
- **B03** Walk me through a project where you had to cut scope to ship.
  What did you cut and why?
- **B04** When have you said "I don't know" in a high-pressure setting?
- **B05** What is a result from your own work you no longer believe, and
  what convinced you?
- **B06** Why this desk, and what will you have built in your first six
  months?
