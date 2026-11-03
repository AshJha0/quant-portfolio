# Rates volatility, P&L explain and historical stress (L3)

## Plain idea

Three tools a rates or options desk uses every day.

- **SABR** gives the implied volatility of a forward at any strike from four
  numbers, so a desk can quote a whole smile from a few liquid points and
  move it consistently when the forward moves.
- **P&L explain** splits the day's profit into what the risk numbers said
  should happen (time decay plus sensitivities times market moves) and what
  is left over. A large leftover means the risk numbers are missing
  something.
- **Historical stress** replays a known bad episode on today's book, using
  the moves in the published data series rather than round numbers.

## Formal statement

1. **SABR (Hagan, Kumar, Lesniewski and Woodward 2002).** The forward and its
   volatility follow `dF = alpha F^beta dW`, `d alpha = nu alpha dZ`, with
   `corr(dW, dZ) = rho`. An asymptotic expansion gives a Black implied vol
   for each strike. The repository offers Hagan's formula and Obloj's (2008)
   correction, which is more accurate away from the money; the two agree at
   the money. Code: `qp_core.vol.sabr` (`sabr_vol`, `sabr_calibrate`), with a
   PDE in `qp_core.vol.sabr_pde` for the arbitrage-free version.
2. **Risk-theoretical P&L.** For factor moves `dx`, deltas `d`, gammas `G`
   and time effects `theta`,
   `RTPL = theta + d . dx + 0.5 dx' G dx`, and the unexplained P&L is
   `HPL - RTPL`, with HPL the full-revaluation (hypothetical) P&L. The FRTB
   P&L attribution test compares these two series day by day (Basel
   Committee, *Minimum capital requirements for market risk*, 2019, the
   P&L attribution test in MAR32). Code: `qp_core.risk.pnl_explain` (`explain`).
3. **Historical scenario.** For a window `[t0, t1]` the shock to a rate is
   `x(t1) - x(t0)` and to an FX rate `ln(x(t1) / x(t0))`, applied as
   `q + dx` or `q exp(dx)`. Code: `qp_core.scenarios.historical`, built on
   frozen copies of the Fed, Bank of England, ECB and CBOE series.

## Worked example

**SABR.** `F = K = 3 %`, `T = 5`, `alpha = 0.02`, `beta = 0.5`,
`rho = -0.3`, `nu = 0.4`. Both variants give 12.1209 % at the money. At a
1 % strike, Obloj gives 28.346 % and Hagan 28.182 %, a gap of 0.16 vol
points. Small at one strike, but enough to matter for a wing price, which is
why the code makes the choice an explicit argument.

**Explain.** A book has delta 100 and gamma 40 to one factor that moves by
0.1, a theta of -1, and a full-revaluation P&L of 9.25. The risk side is
`100 x 0.1 + 0.5 x 40 x 0.01 = 10.2`, the RTPL is `10.2 - 1 = 9.2`, and the
unexplained P&L is 0.05.

**Stress.** The Swiss National Bank removed the EURCHF floor on 15 January
2015. The ECB fixing went from 1.2010 to 1.0280, a log move of
`ln(1.0280 / 1.2010) = -0.1555`. Applying `scenario("chf_2015")` to a
snapshot quoting 1.2010 gives 1.0280.

Source: recomputed from the package code in
`docs/learn/tools/test_worked_examples.py::test_rates_vol_and_explain`.

## Where the repository goes further

- A swaption cube with SABR per expiry and tenor:
  `python/equity/04-fixed-income/src/fi_rates/swaptions.py`.
- How a smile moves when spot moves (sticky strike, sticky moneyness, sticky
  delta)
  and vega by bucket: `qp_core.risk.smile_dynamics`,
  `qp_core.risk.vega_grid`.
- Par-rate and zero-rate risk, and the Jacobian between them:
  `qp_core.risk.jacobian` and `fi_rates.risk.par`.
- Reverse stress (how large must the episode be to lose a given amount):
  `reverse_stress_scale`.

## Checks

1. Why does `beta` matter less than it seems when fitting one smile?
2. A desk's unexplained P&L is small on average but large on the days the
   market jumps. Which term is probably missing?
3. Why is the FX shock a log move rather than a difference?

Answers: `beta` and `rho` both tilt the smile, so for one expiry they are
nearly interchangeable; desks fix `beta` and fit the rest; a higher-order or
cross term (cross-gamma, or vol moves the vega does not capture), which only
shows up when moves are large; a log move applied as `q exp(dx)` keeps the
rate positive and gives the same shock whichever way round the pair is quoted,
up to sign.
