# Learn, by level

This folder is the front door to the teaching material. It sorts what you need
to know into five levels, L0 to L4, and gives each level the same four parts:
a plain idea, a formal statement, a worked example and a set of checks. The
559 questions and answers stay in one appendix,
[LEARN.md](../LEARN.md), so that links into it keep working; every question
there now carries a level tag, and each level folder lists its own questions.

## The levels

| Level | Who it is for | You can do this at the end | Start here |
|---|---|---|---|
| L0 | New to markets and to this repository | Say what each area of the portfolio is for and read a risk number | [L0/README.md](L0/README.md) |
| L1 | Comfortable with school maths and some code | Compute returns, volatility and a historical VaR by hand and match the code | [L1/README.md](L1/README.md) |
| L2 | Ready to price and hedge | Price a European option, a bond and a CDS, and explain each Greek or sensitivity | [L2/README.md](L2/README.md) |
| L3 | Ready for the models behind the models | Derive the Black-Scholes equation with Ito's lemma, and reason about market impact | [L3/README.md](L3/README.md) |
| L4 | Working on a desk or in model validation | Judge a backtest, a capital number or a research result, and say what would change your mind | [L4/README.md](L4/README.md) |

The levels are a reading order, not a ranking of people. Most readers enter at
L1 or L2 and drop back a level when a question stops making sense.

## Chapters

| Chapter | Level | What it covers |
|---|---|---|
| [Stochastic calculus](L3/stochastic-calculus.md) | L3 | Brownian motion, Ito's lemma, the Black-Scholes equation from a hedge |
| [Rates](L2/rates.md) | L2 | Bond prices, yield, duration and convexity |
| [Credit](L2/credit.md) | L2 | Hazard rates, survival, the CDS legs and the credit triangle |
| [Microstructure](L3/microstructure.md) | L3 | Spreads, the Roll estimator, Kyle's lambda and the square-root impact law |
| [Dated swaps, curve building and bond futures](L2/dated-swaps-and-bond-futures.md) | L2 | SOFR compounded in arrears, dated bootstrap, conversion factors, credit index defaults |
| [Counterparty risk and capital](L2/counterparty-and-capital.md) | L2 | CVA and SA-CCR, with the Basel Committee's Example 1 |
| [Rates volatility, P&L explain and historical stress](L3/rates-vol-and-pnl-explain.md) | L3 | SABR (Hagan and Obloj), risk-theoretical P&L, the 2015 franc shock |
| [Market making, routing and the risk book](L3/market-making-and-routing.md) | L3 | Avellaneda-Stoikov, GLFT, smart order routing as a linear program, RFQ markup, netting |
| [Trading systems](L3/trading-systems.md) | L3 | One backtest core, the kill switch, a feed handler that admits gaps, settlement calendars |
| [Adjoint algorithmic differentiation](L4/aad.md) | L4 | The tape, one reverse sweep for all Greeks, and what it costs in practice |

## How the pieces fit

- The level pages and chapters teach. They are short on purpose.
- [LEARN.md](../LEARN.md) is the appendix: Parts I to III (orientation and
  learning paths) and the 559-question exercise room.
- The [notebooks](../notebooks/README.md) are where you type the code
  yourself.
- [CAREER_PREP.md](../CAREER_PREP.md) covers probability puzzles, mental
  maths and market-making games.

## How the numbers are kept honest

Every number in a worked example on these pages is recomputed by
`docs/learn/tools/test_worked_examples.py` from the repository's own code
(or, for pure arithmetic, from the formula on the page). Every question in
the appendix tagged `numeric` cites a file, a test or a module, and
`docs/learn/tools/lint_learn.py` fails if one does not, if a cited path does
not exist, if a question has no level tag or folded answer, or if a prose line
runs past 400 characters. `docs/learn/tools/build_index.py --check` fails if a
level's question list is stale.

Run all three from the repository root:

```
python -m pytest docs/learn/tools
```

The level of each question was first assigned by a rule (the set's level,
one lower for "what is" questions, one higher for derivations), then read.
If a question feels mislevelled, change its tag in LEARN.md and rerun
`python docs/learn/tools/build_index.py`.
