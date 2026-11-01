# Desk guide: running a pairs book with this library

This guide maps the library onto a statistical-arbitrage workflow. Every
number the library produces today comes from synthetic data with planted
cointegrated pairs; no real-data result exists yet (Phase 4). Read the
limits and scenarios below as a description of how the tools would be used,
not as evidence that the strategy earns money.

What is coded and what is desk process:

| control | status |
|---|---|
| Engle-Granger gate, half-life gate [1, 126] days | coded (`_fit_formation`) |
| entry/exit/stop/time-stop state machine with re-entry arming | coded (`generate_signals`) |
| next-period fill, close-auction variant with cost | coded (`ExecutionConfig`) |
| commission, slippage, square-root impact, borrow, financing | coded (`CostModel`) |
| walk-forward re-fit with frozen parameters | coded |
| book drawdown kill switch, sector limits, leverage cap | not coded; desk process |
| borrow locate, recall handling, per-name borrow fees | not coded; desk process |
| weekly cointegration re-test of open pairs | not coded; run `engle_granger` by hand |

---

## 1. Research, paper, size-up

**Research.** Universe: liquid names with sector tags. Run the funnel as in
`run_pipeline` §2: same-sector candidates, return-correlation or SSD screen,
Engle-Granger at 5% with N=2 critical values, OU gates (mean reverting,
half-life in [1, 126] days). Per pair keep beta, alpha, kappa, mu, sigma,
half-life, ADF statistic and the walk-forward result with full costs.
Log every configuration tried in a `TrialLog` and report the deflated
Sharpe ratio against that count.

**Paper.** Trade the signal on paper against live prices for one to three
months. Check that realised fill prices match the t+1 fill model, that
borrow is available, and that realised impact matches the square-root
model at your size.

**Size-up.** Start at a quarter of target gross per pair. Judge live results
against the walk-forward distribution over seeds, not the in-sample one
(VALIDATION §4 shows the gap).

| number | consumer | cadence |
|---|---|---|
| per-pair z, half-life, days in trade against the time stop | trader | daily |
| per-pair attribution (`PortfolioResult.attribution()`) | PM | daily |
| Sharpe with Lo SE, drawdown, turnover, cost drag by component | risk | weekly |
| deflated Sharpe and PBO against the trial log | research, risk | each new variant |
| borrow fee and utilisation per short leg | borrow desk, trader | daily |

## 2. Risk limits (desk process, not coded)

Scaled to the pipeline book: $1mm gross per pair, 6 pair slots, $6mm
capital.

- Per pair: gross at most $1mm; |z| stop at 4 with no re-entry until |z| < 2
  (coded); time stop at 3 half-lives (coded).
- Per sector: at most 3 pairs sharing a sector leg.
- Book: halve gross at a drawdown of 1.5% of capital and go flat at 3%.
  These triggers are not in the code. Set them above the walk-forward
  drawdown distribution in VALIDATION §4.
- Leverage: cap gross by stressed spread volatility, not trailing
  volatility.
- Cointegration health: re-test open pairs weekly on the trailing formation
  window; two failures at 10% stop new entries.

## 3. Capacity

The pipeline reports the walk-forward Sharpe at impact coefficients
Y = 0, 0.3, 0.7, 1.0 and at $1mm and $10mm gross per pair (VALIDATION §7).
The ADV of $20mm per name is an assumption because the synthetic data has
no volume. On real data, replace it with each name's measured ADV and
re-run the table.

## 4. Borrow desk

Before entry, confirm a locate and a fee for every short leg and set
`CostModel.borrow_bps` per pair. A recall is an involuntary exit at an
uncontrolled price. None of this is modelled.

## 5. Scenarios

**Crowded unwind (August 2007).** Many pairs hit stops together with no
news. Cut gross first. Under the independence assumption (METHODOLOGY A8)
this pattern is very unlikely, so treat it as that assumption failing.

**Short squeeze (January 2021).** One short leg rallies, the borrow fee
spikes and z says add. The model has no state for a cornered float. Respect
the |z| = 4 stop.

**Merger.** The target leg pins to the offer price and the spread jumps and
does not revert. Retire the pair on the corporate-action news. The
synthetic regime-break case (VALIDATION §8) is a small version of this.

**Halt or long gap.** `align_pair` refuses gaps longer than 5 days.

**Delisting or split.** Use split-adjusted prices. `align_pair` trims the
history after a leg stops trading, so a delisted leg's final return is not
booked; on real data, book it from a delisting-return source.

## 6. Model governance

Per pair, keep the formation window, fitted parameters, ADF statistic
against the N=2 critical value, signal rules and cost assumptions. Every
number in the docs regenerates from `python examples/run_pipeline.py`; the
test suite is the regression gate for any code change.
