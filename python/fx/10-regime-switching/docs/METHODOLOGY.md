# Methodology: FX Regime Switching (Risk-On / Risk-Off)

Every result in this project comes from synthetic data. The generator
plants regimes with known drifts, vols and correlations, and the edge the
strategy shows is the drift it plants. No real-data result exists yet.

## 1. The regime story being modelled

Currency markets are usually described in terms of risk-on and risk-off,
and the carry trade is the position most exposed to the switch.

* Risk-on: funding is cheap and vol is low. Carry currencies (AUD, NZD,
  EM) drift up against USD, havens (JPY, CHF) drift down, and the rate
  differential accrues daily.
* Risk-off: crowded carry positions unwind. Carry and EM currencies fall,
  JPY and CHF are bid, vol rises and pairwise correlations of USD pairs
  rise.
* USD squeeze: everything falls against USD, havens included.

The carry-crash literature (for example Brunnermeier, Nagel and Pedersen,
"Carry Trades and Currency Crashes", 2008) reports that carry returns are
negatively skewed and that unwinds coincide with funding stress and vol
spikes. Whether high-vol states have negative expected carry returns out
of sample is an empirical question. The synthetic generator here assumes
they do (risk-off drift -25% a year for G10 carry and -40% for EM), so the
synthetic results cannot answer it.

## 2. Pipeline

```
returns + deposit rates (synthetic RORO generator, known truth)
  -> features.py   7 FX features, expanding-window (PIT) z-scores
  -> pca.py        PCA; PC1 of the currency panel is the RORO axis
  -> gmm.py/hmm.py GMM (BIC evidence, HMM initialisation) and Gaussian HMM
  -> detection.py  expanding refit, filtered probabilities, labeling,
                   hysteresis
  -> strategy.py   regime books, vol targeting, carry, pip costs
  -> backtest.py   walk-forward ledger, net = spot + carry - cost
  -> risk.py       per-regime stats, oracle gap, detection lag,
                   random-timing benchmark
```

## 3. Model choice

| model | captures | misses |
|---|---|---|
| vol threshold | the main symptom, easy to explain | correlation and haven signals, persistence, arbitrary threshold |
| correlation threshold | the one-factor market | same problems as the vol rule |
| GMM | the multivariate signature of each state | dynamics: each day is classified alone, so labels flicker |
| Gaussian HMM (used) | the GMM signature plus a transition matrix | fat tails, time-varying transition hazards, k must be chosen |

The filtered HMM probability p(s_t | x_1..t) combines today's evidence
with yesterday's belief through the transition matrix, so one bad day
moves it a little and several move it a lot.

EM finds local optima. Each fit starts from GMM-seeded initialisations.
Refits are warm-started from the previous model, and every fourth refit
a cold fit competes with the warm one; the higher training likelihood
wins. When EM stops at `max_iter`, the reported likelihood is recomputed
for the returned parameters.

## 4. Filtered, never smoothed

* Filtered p(s_t | x_1..t) uses data up to t and is the only probability
  that trades.
* Smoothed p(s_t | x_1..T) uses the whole sample and is research only.

Mutation tests perturb data after t and require every output at or
before t to be unchanged (`tests/test_detection.py`,
`tests/test_backtest.py::test_no_lookahead_mutation`).

## 5. Features

| feature | construction |
|---|---|
| `avg_vol` | mean rolling realised vol of G10-vs-USD legs, annualised |
| `carry_ret` | rolling return of the rank-carry basket (top-3 long, bottom-3 short) |
| `haven_rs` | rolling (JPY+CHF)/2 minus (AUD+NZD)/2 return |
| `usd_corr` | average pairwise rolling correlation of the risk-block USD pairs |
| `em_g10` | rolling EM basket minus G10 basket return |
| `usd_str` | minus the rolling average return of all currencies vs USD |
| `fwd_ts` | CIP forward-point proxy, mean (r_ccy - r_USD) of the carry longs |

All features are standardised with expanding-window z-scores. Windows
default to 8 days (vol), 5 days (returns) and 12 days (correlation).
The HMM uses five columns by default (`avg_vol`, `carry_ret`, `haven_rs`,
`usd_corr`, `em_g10`). The windows and the column set were chosen on
design seeds only (section 8).

## 6. Labeling and the number of states

HMM state indices are arbitrary. Each refit maps them from the state
means in feature space: `risk_on` is the lowest `avg_vol` mean; with
k = 3 the other two split on `haven_rs` (havens rallying is `risk_off`,
havens falling is `usd_squeeze`).

On raw iid returns BIC recovers the true k (tested). On autocorrelated
rolling-window features BIC is unreliable, so k = 3 is an assumption.
The pipeline prints the BIC table as evidence only.

## 7. Strategy, execution and costs

| regime | book |
|---|---|
| `risk_on` | rank-carry: long top-3 yielders, short bottom-3, dollar-neutral |
| `risk_off` | long JPY and CHF against AUD, NZD, MXN, ZAR, BRL |
| `usd_squeeze` | long USD against every currency |

Books are vol-targeted to 10% a year with a trailing 63-day covariance,
leverage capped at 4x, with a vol floor for degenerate covariances.

Execution. Signals use data up to the close of day t. Orders fill at a
modelled t+1 price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a
close-auction variant fills at p_t and pays an auction cost. Over day
t+1 the old book earns p_fill/p_t - 1 = f R and the new book earns
p_{t+1}/p_fill - 1 = (1 + R)/(1 + f R) - 1, where R is the simple return
of the currency against USD. f = 1 is a fill at the next close.

Carry. Each book accrues w . (r_ccy - r_USD) x days / 365 over the
calendar gap between closes (ACT/365, so a weekend accrues three days),
split between the old and new book in the proportion f : 1 - f. This is
the same day count as the FX pairs project.

Costs. Half-spreads are given in market pips of each pair's quote
(AUDUSD, USDJPY, USDMXN and so on). The cost per unit of notional is
pips x pip size / quote level, with pip size 0.01 for USDJPY and 0.0001
otherwise, and the quote level taken from the simulated path at the
decision close. At reference levels this is 0.5-1.7 bp for G10 legs and
3-10 bp for EM legs. The close-auction mode adds its auction cost per
unit of traded notional.

Hysteresis. A switch needs the challenger's filtered probability at
least 0.70, or the incumbent's below 0.30, for 2 consecutive days.

## 8. Research protocol

Configurations are chosen on design seeds and judged on fresh seeds:

1. Twelve configurations (2 window sets x 3 HMM column sets x 2 risk-off
   baskets) are the ones tried during development. Each is logged in a
   `TrialLog` and run on design seeds 1001-1003.
2. The best mean Sharpe on design seeds fixes the configuration. PBO is
   computed by CSCV across the twelve.
3. The fixed configuration is evaluated on seeds 2001-2008. Each seed is
   run twice: with the planted drifts, and with zero drift in every state
   (same states and shocks). The zero-drift panel is the vol-clustering
   null: vol and correlation regimes without any link to expected returns.
4. Benchmarks: an oracle that knows the true state, static carry, and the
   strategy's own regime path shifted to random times (same occupancy and
   spell lengths, so equal exposure to each book).
5. Statistics come from `research_stats.py`: Lo (2002) Sharpe standard
   errors with a Newey-West lag window, the deflated Sharpe ratio at the
   logged trial count, PBO, circular block-bootstrap CIs, and Student-t
   CIs across seeds.

## 9. Assumptions register

1. Gaussian emissions. FX returns are fat-tailed. If violated, one large
   day can force a false flip and tail risk inside a state is understated.
2. Constant transition probabilities. Real flip hazards cluster around
   events. If violated, filtered priors are wrong around scheduled events.
3. k is known. With k too small distinct crash types merge; with k too
   large the labeler maps noise to books.
4. Markov property. Duration-dependent hazards bias durations; filtered
   inference degrades gradually.
5. Stationary features under expanding standardisation. A structural
   break makes z-scores drift.
6. Deposit rates stand in for forwards (CIP). In a squeeze the
   cross-currency basis widens and carry on funding legs is overstated.
7. Spreads are constant in pips. In risk-off they widen several times,
   so switching costs at the flip are understated.
8. Returns are currency-vs-USD legs; a long AUD leg is long AUDUSD, a
   long JPY leg is short USDJPY.
9. The synthetic drift link between state and expected return is assumed,
   not estimated.

## 10. Conventions

* Returns in the data: daily log returns of currency vs USD; positive
  means the currency appreciated. P&L uses simple returns.
* Rates: annualised decimals; carry ACT/365.
* Vol: annualised with sqrt(252); realised vol uses ddof=0.
* Max drawdown: additive cumulative sum of daily returns.
* All randomness flows through seeded `numpy.random.Generator` objects.
