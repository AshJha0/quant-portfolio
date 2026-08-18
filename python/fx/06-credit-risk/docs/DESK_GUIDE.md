# Desk Guide — Who Uses This, How, and What the Limits Look Like

## 1. Consumers and daily workflow

| Desk / function | What they take from this project | Cadence |
|---|---|---|
| **Country risk / credit** | Scorecard PD + rating band per sovereign → country limits; binning tables for the annual review pack | Annual refit, quarterly monitoring (PSI, realised-vs-predicted), event-driven overrides |
| **EM trading desk** | Pre-trade check: does the trade fit within the counterparty's PFE limit and the country limit? Rating-band PD feeds spread guidance | Every trade > threshold |
| **Treasury / operations** | Settlement-line management: gross Herstatt exposure per counterparty vs settlement limit; CLS vs non-CLS routing decisions (the 6-trade demo: USD 180m gross → 66m netted → target: everything CLS-eligible into CLS) | Daily, per value date |
| **CVA desk** | EE profile and CVA per counterparty netting set; hedges CVA with sovereign CDS / FX options; flags wrong-way trades (long-USD forwards vs the EM sovereign itself) for the WWR add-on | Daily batch + new-trade incremental |
| **Capital / finance** | EL for provisioning; standardized RW for regulatory RWA; Vasicek K for internal capital allocation and limit sizing (the AAA wedge: 0 regulatory vs 0.34 per 100 internal) | Monthly |
| **Model risk / validation** | VALIDATION.md evidence pack: OOT metrics, bootstrap CIs, contagion-year stress, autocorrelation caveat | Annual model review |

A morning run: `run_pipeline.py`-equivalent batch produces (i) refreshed PDs
and ratings per country, (ii) per-counterparty settlement exposure for
today's value dates, (iii) EE/PFE/CVA per netting set, (iv) exceptions:
limit breaches, PSI > 0.10, ratings migrating ≥ 2 notches.

## 2. Limits governance

- **Country limits** (owned by country-risk committee): notional cap per
  rating band, e.g. band B ⇒ cap X, CCC ⇒ X/4, C ⇒ exit-only. Model PD is an
  *input*; the committee can override with documented rationale (peg watch,
  election risk). Every override is logged and back-tested annually.
- **Settlement lines** (treasury): cap on same-day gross non-PvP principal
  per counterparty. CLS members get economic relief (their PvP legs count
  zero); non-CLS counterparties consume the line at full bought-principal.
  Escalation: line breach blocks release of the outgoing payment, not the
  trade booking — the payment queue is where Herstatt risk is actually
  controlled.
- **PFE limits** (credit): 99% peak PFE per netting set ≤ limit, sized off
  the counterparty band (BB sovereign: limit ~ f(K_sov = 14.1 per 100)).
  Un-netted counterparties (no ISDA/CSA — common for sovereigns, who resist
  collateral annexes) are measured gross: the 3.54m-vs-0 netting demo is the
  argument credit officers take into ISDA negotiations.
- **Escalation ladder**: 85% of any limit → desk head notified; 100% →
  auto-block new risk-additive trades; contagion trigger (regional crisis
  flag) → all regional limits cut 50% pending committee review — because the
  model's own validation shows calibration fails 2.6x in contagion years.

## 3. Realistic scenarios (history, encoded in this project)

1. **Herstatt 1974 (the anchor).** Counterparties paid DEM to Bankhaus
   Herstatt during the European morning; German regulators withdrew its
   licence at 15:30 CET, before USD legs settled in New York. Full principal
   lost on the paid legs. In this project: pay-EUR/receive-USD window =
   17.5 h at full principal (`test_window_herstatt_1974_direction`); the CLS
   comparison shows PvP eliminating exactly that exposure. CLS itself exists
   because of this event (est. 2002).
2. **Russia 1998.** A sovereign defaulting on *domestic-currency* debt (GKOs)
   while Basel weighted it favourably — the reason our capital block refuses
   the 0% sovereign floor internally (AAA row: standardized 0.00 vs Vasicek
   0.34 per 100). Also a Guidotti-ratio poster child: short-term debt far
   above reserves in vintage data. And a settlement lesson: forward RUB/USD
   hedges failed as Russian banks (counterparties on the hedges) defaulted
   together with the sovereign — wrong-way risk (VALIDATION §4.4).
3. **Argentina 2001 / 2019.** 2001: currency-board peg masked risk until it
   broke (Assumption A4) — scorecard inputs looked stable while devaluation
   risk accumulated; the model's peg dummy adds risk but cannot time the
   break ⇒ mandatory overlay on peg countries. 2019: capital controls
   (cepo) — trades neither defaulted nor settled: convertibility risk means a
   solvent counterparty cannot deliver; settlement lines must distinguish
   country-of-currency from country-of-counterparty.
4. **2022 RUB sanctions.** Settlement failure *without* default: sanctioned
   payment rails meant USD legs could not clear against RUB legs; CLS
   suspended RUB. Model lesson: `PAYMENT_SYSTEM_HOURS_UTC` assumes the rail
   exists — a sanctions scenario zeroes the receive-side finality, turning
   every open trade into full-principal exposure regardless of PD. Treasury's
   control is the settlement line + same-day kill-switch on payment release,
   not the PD model.
5. **2020-style global contagion (planted in the data).** The out-of-time
   2020 year defaults at 16.7% vs 6.4% predicted: limits governance (regional
   50% cut on contagion trigger) exists precisely because the PD model is
   documented to under-call systemic waves.

## 4. P&L and attribution

CVA desk marks CVA = 4.4 bp (BB example) into the trade at inception and
hedges the PD leg with sovereign CDS, the EE leg with FX options; daily CVA
P&L attributes to (i) spread/PD moves, (ii) FX spot/vol moves on EE, (iii)
new trades/netting changes. Country-risk EL feeds loan-loss provisioning.
Settlement risk carries no P&L — it is a pure tail-loss control, which is why
its limit is a hard block rather than a priced charge.

## 5. Model governance summary

Annual: refit + re-bin, shadow-rating benchmark comparison, override
back-test, VALIDATION.md refresh with new OOT year. Quarterly: PSI, band
migration matrix, realised-vs-predicted by band. Event-driven: contagion
trigger, peg-break watchlist, sanctions rail review. Change control: any
band-boundary or PD-midpoint change requires model-risk sign-off (it rescales
every CVA and limit downstream — Assumption A9).
