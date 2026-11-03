# Pre-registration: FX carry + value + momentum combination, cross-currency panel

Status: DRAFT, unsigned, not yet gated. It lives here, not in
`research/prereg/`, until the W23 part 1 panel (`research/panel/`) is merged
and the power numbers below are replaced by a simulated power file. At
integration it moves to `research/prereg/fx-combination-panel.md` and is
then frozen. Everything above the Signature heading is hashed.

This is the "properly powered cross-currency carry + momentum test" that
owner decision D4 asked for, with value added as the third known premium.

## Economic prior: who pays

- Carry: investors in high-rate currencies are paid for bearing global FX
  crash risk (Lustig, Roussanov and Verdelhan 2011, RFS 24(11); Menkhoff,
  Sarno, Schmeling and Schrimpf 2012, JF 67(2)).
- Value: real exchange rates revert slowly towards purchasing-power parity,
  and the slow capital that would close the gap is paid for waiting
  (Menkhoff, Sarno, Schmeling and Schrimpf 2017, RFS 30(4); Asness,
  Moskowitz and Pedersen 2013, JF 68(3)).
- Momentum: information diffuses slowly and limits to arbitrage are largest
  in less liquid currencies (Menkhoff et al. 2012, JFE 106(3)).

All three are known premia. A positive result is labelled "known premium,
replicated", not new alpha. Post-publication decay is expected (McLean and
Pontiff 2016, JF 71(1)).

## Hypothesis

H1: the cross-sectional Fama-MacBeth slope of next-month financed excess
returns on the combined score is positive over the 2016-01 to 2026-10
hold-out on the W6c universe.

H0: the mean slope is at most zero.

## Data

- Signals and returns: the W23 part 1 panel (`research/panel/`), read only
  through `research/combination/panel_seam.py`: carry (forward discount),
  value (5-year change in the CPI-based real exchange rate, OECD CPI and
  FRED H.10), momentum (12-1 month excess return), next-month financed
  excess returns (W6 `excess_returns.py`), and the point-in-time `live`
  mask (W6c universe, 30+ currencies).
- In-sample: 1999-01 to 2015-12. Hold-out: 2016-01 to 2026-10, scored once.

## Parameters (fixed, one variant)

`research/combination/combine.py`, `CombinationSpec()` defaults:
cross-sectional z-scores clipped at 3, equal-weight mean of the three
signals with at least 2 present, dollar-neutral unit-gross raw weights,
EWMA covariance half-life 12 months shrunk 50 % to its diagonal, 10 %
annual vol target, gross cap 3, name limit 0.25, tiered costs from
`research/lib/costs.py` at 1x. No cost-QP sizing in the primary test (the
sizer is a secondary report). Trial count: 1.

## Primary metric and decision rule

Fama-MacBeth mean slope of next-month excess returns on the combined score
(`research/lib/panel.py`, `fama_macbeth`, Newey-West lag 3), one-sided
t-test at 5 %. Reject H0 if t > 1.645. Published whatever it is.

## Power (W6 gate)

Analytic gate (`research/lib/power_gate.py`): power =
`Phi(IC sqrt(N_eff) sqrt(12) sqrt(10.83) - 1.645)`.

| monthly IC (ASSUMPTION) | N_eff | power |
|---|---|---|
| 0.04 | 15 | 0.548, blocked |
| 0.06 | 15 | 0.842, passes |

The IC and N_eff must be replaced by the in-sample estimates from the
panel, and `prereg.py` requires a simulated power file under
`research/power/` named here before signing. Neither exists yet: this draft
cannot be signed in Round 11.

## Secondary (reported, not tested)

Vol-targeted portfolio net Sharpe (Lo SE), turnover, holding period,
capacity curve and AUM at which net Sharpe halves (`research/capacity/`,
square-root law with Y = 1 ASSUMPTION and BIS 2022 ADV), factor attribution
on DOL and HML_FX (W23a), cost-QP sized variant.

## Signature

signed_by:
signed_on:
sha256:
