# Microstructure: spreads, information and impact (L3)

## Plain idea

Prices move for two reasons when you trade. You cross a spread, because the
market maker who takes the other side wants paying for the risk and for the
chance that you know something. And your order pushes the price, because it
uses up liquidity and because others read it as information. The first cost
is visible in the quote. The second is not, and it grows with the size of the
order relative to what the market normally trades.

## Formal statement

1. **Roll (1984).** If trades bounce randomly between bid and ask around an
   efficient price, consecutive price changes are negatively correlated, and
   the effective spread is `s = 2 sqrt(-cov(dp_t, dp_{t-1}))`. The estimator
   is undefined when the covariance is positive.
2. **Kyle (1985).** With one informed trader, noise trading of standard
   deviation `sigma_u` and a value uncertainty of standard deviation
   `sigma_v`, the single-period equilibrium price moves
   `lambda = sigma_v / (2 sigma_u)` per unit of order flow. Price impact is
   linear in order size, and deeper noise trading means smaller impact.
3. **The square-root law.** Empirically, the impact of a parent order of
   size `Q` in a market with daily volume `V` and daily volatility `sigma` is
   close to `Y sigma sqrt(Q / V)`, with `Y` of order one (Almgren, Thum,
   Hauptmann and Li, 2005; Bouchaud, Bonart, Donier and Gould, *Trades,
   Quotes and Prices*, 2018, ch. 12).
4. **Almgren-Chriss (2000).** Splitting the order over time trades impact
   cost against price risk; the optimal schedule front-loads more as risk
   aversion rises.

Reference: Harris, *Trading and Exchanges* (2003), ch. 13-14; Kyle (1985),
*Econometrica* 53(6); Roll (1984), *Journal of Finance* 39(4). The
repository's execution code is
`python/equity/08-algo-execution/src/eq_algo/almgren_chriss.py`
(`ac_trajectory`, `power_law_trajectory`, `compare_impact_laws`), with the
FX counterpart in `python/fx/08-algo-execution`.

## Worked example

**Roll.** Daily price changes have a first-order autocovariance of
`-0.0001` (prices in dollars). The implied effective spread is
`2 sqrt(0.0001) = $0.02`, two cents.

**Kyle.** With `sigma_v = $2` and `sigma_u = 1,000,000` shares, `lambda` is
`2 / (2 x 1,000,000) = $0.000001` per share: a 50,000-share imbalance moves the
price 5 cents.

**Square root.** A stock with 2 % daily volatility; you buy 1 % of daily
volume. With `Y = 1` the impact is `0.02 x sqrt(0.01) = 0.2 %`, or 20 bp.
Buying 4 % of volume costs 40 bp, not 80 bp: four times the size, twice the
cost per share.

Source: all three are arithmetic from the formulas above, checked in
`docs/learn/tools/test_worked_examples.py::test_microstructure`. The
repository's own Almgren-Chriss and power-law numbers are in
`python/equity/08-algo-execution/docs/VALIDATION.md`.

## Where the repository goes further

- Optimal schedules under linear and power-law impact, scored against each
  other: `compare_impact_laws`, and the
  [execution notebook](../../notebooks/04-optimal-execution-exercise.ipynb).
- Implementation shortfall and benchmark slippage: `eq_algo.benchmarks`.
- Order types, market making and venue routing are being added in parallel
  workstreams (W25-W30); this chapter will link to them once they are merged.

## Checks

1. Why does the Roll estimator fail in a trending market?
2. In Kyle's model, what happens to `lambda` if noise trading doubles?
3. Under the square-root law, what is the total cost (not per share) of
   doubling the order?

Answers: trends add positive autocorrelation that offsets the bid-ask bounce,
so the covariance can turn positive; it halves; it rises by a factor
`2 x sqrt(2) ~= 2.83`.
