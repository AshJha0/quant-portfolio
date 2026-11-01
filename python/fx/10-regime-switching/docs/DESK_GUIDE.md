# Desk Guide: Using a RORO Regime Filter on an FX Desk

This guide describes how a currency desk could consume a regime filter of
this kind. Every number quoted comes from synthetic data in which the
generator plants the crash drift; nothing here has been run on real data
yet. Read section 6 before relying on any control listed here.

## 1. Outputs and consumers

| output | consumer | cadence |
|---|---|---|
| filtered regime probabilities | PM, risk manager | daily close |
| committed regime label (after hysteresis) | carry-book sizing, overlay | daily close, filled during the next day |
| expected durations, transition row | PM | at each flip |
| per-regime P&L attribution | risk committee | weekly |
| detection lag and oracle gap | model governance | monthly |

## 2. Daily workflow

1. At the close of day t, features update from spot closes and deposit
   rates. Filtered probabilities are recomputed. The HMM refits on a
   schedule (every 63 business days in the pipeline, 21 by default),
   warm-started, with a competing cold fit every fourth refit.
2. Hysteresis keeps or flips the committed regime (challenger at least
   0.70 for 2 days, or incumbent below 0.30).
3. Orders fill at a modelled t+1 price, p_t + f (p_{t+1} - p_t) with
   f = 0.5 by default. A close-auction variant fills at p_t and pays an
   auction cost.
4. The ledger books spot, ACT/365 carry and pip costs; net = spot +
   carry - cost holds row by row.

## 3. Usage patterns

* Carry-book throttle. Scale an existing carry book by p(risk_on). On
  synthetic data the filtered book beats static carry by 0.59 Sharpe
  [-0.19, 1.37] across 8 seeds, and the whole difference is the planted
  crash drift (VALIDATION.md section 4). Whether high-vol FX states carry
  a negative expected return out of sample is the question Phase 4 must
  answer on real data before this use is justified.
* Tail-hedge overlay. A filter reacts after jumps (peg breaks), so gap
  risk needs options or limits, not the filter.
* Risk dashboard. The features themselves (vol, carry momentum, haven
  strength, USD-pair correlation, EM spread, dollar factor) are readable
  as expanding z-scores.

## 4. Historical episodes to test in Phase 4

These are candidate real-data checks. None has been run.

* 2008: carry unwind followed by a dollar-funding squeeze.
* March 2020: a short squeeze in which havens also sold off against USD.
* 2013 taper tantrum: an EM-led unwind with calm G10 features.
* 2022: a slow dollar rally at modest daily vol.
* January 2015 SNB: a peg break, which is a jump rather than a regime.

## 5. Hysteresis is a policy choice

Enter and exit thresholds and the confirmation days trade detection lag
against false alarms. On 8 synthetic seeds the fast setting (0.70/0.30,
2 days) detects 95% of flips with a 2.7-day mean lag; the conservative
setting (0.90/0.10, 5 days) detects 87% with a 5.3-day lag and a lower
Sharpe (VALIDATION.md section 5).

## 6. What exists in code and what does not

In code and tested:

* filtered-only detection, hysteresis with confirmation days,
  expanding refits with periodic cold restarts;
* the t+1 fill model and the close-auction variant;
* per-regime statistics, oracle gap decomposition, detection-lag report,
  random-timing benchmark.

Not implemented (procedures a desk would need to add):

* event-calendar overrides (freezing upgrades before central-bank
  meetings, PM-imposed risk_off);
* review of refits that relabel history;
* kill criteria on detection rate or false-alarm cost;
* scaling a book by p(risk_on) instead of switching books;
* per-country limits for idiosyncratic EM crises.

## 7. Limits

The filter addresses systemic risk-on/risk-off moves in a diversified
currency book. It does not address country-specific crashes, peg breaks,
slow carry drawdowns without a vol spike, or execution in gapping markets.
