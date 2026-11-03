# research/tsmom: cross-asset time-series momentum (W23 part 3, study F)

Each asset is long after a year in which its excess return was positive and
short after one in which it was negative, scaled to 40 % ex-ante volatility,
and the portfolio is the equal-weight mean (Moskowitz, Ooi and Pedersen
2012). The assets are the FX panel's financed excess returns and the French
`Mkt-RF`. A known premium: a positive training result reproduces the
literature.

| File | What it does |
|---|---|
| `tsmom.py` | Signal, EWMA ex-ante volatility, weights, turnover, per-asset and portfolio returns with Newey-West t-statistics; `assemble_assets` joins return blocks. Inputs pass the hold-out lock. |
| `tests/test_tsmom.py` | The docstring's worked example (+8 % and -8 %), no look-ahead, the EWMA against a hand recursion, turnover, the lock. |

The confirmatory test is drafted in
`research/prereg/2026-11-03-tsmom-cross-asset.md` (unsigned; it needs a
simulated power result before signing). Until it is signed and opens the
lock, every result here is labelled exploratory and sees only months before
2016.

Difference from the paper, stated with every result: volatility is an EWMA
of monthly squared returns (centre of mass 12 months); the paper uses daily
returns with a 60-day centre of mass.
