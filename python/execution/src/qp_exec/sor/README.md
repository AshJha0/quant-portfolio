# Smart order router and multi-venue simulation (W27)

`qp_exec.sor` routes a parent order across several lit order books and
several dark venues, and measures what that routing costs: fees, spread,
fill rates, adverse selection, information leakage and gaming. Everything
runs on one seeded simulator, so every number below can be reproduced.

**What this is and is not.** The market is stylised. There is no free
dark-pool data, so venue rules follow published descriptions and the
behavioural parameters (who arrives, how often, how often they firm up) are
assumptions written in the code. Read every number here as an illustration
of a mechanism, not as an estimate for any real venue. Fee schedules are
shaped like public US equity schedules but are not any named exchange's
current schedule (sources at the top of `python/execution/venues/fee_schedules.toml`).
Trade-through checks are properties of this simulator, not a claim of
Regulation NMS compliance.

## Pieces

| Module | What it does |
|---|---|
| `fees.py` | Maker-taker, inverted and flat lit schedules; dark fees; latencies. |
| `market.py` | Efficient price with drift regimes and permanent impact; lit venues are W25 `ReferenceBook`s quoted by a market maker; Poisson takers, some informed. |
| `dark.py` | Midpoint pool, conditional pool, periodic (batch) auction pool; counterparties; the predator. |
| `allocation.py` | Cont-Kukanov order placement as an LP, plus closed-form special cases. |
| `router.py` | The router: sweep, allocate, post, cancel, learn; optional randomisation and IOIs. |
| `tca.py` | Per-venue fill rate, effective spread, markouts; markout-learned venue scores. |
| `gaming.py` | The predator's detector, held-out AUC and bootstrap CI. |
| `studies.py` | The W27b measurements below. |

## 1. Allocation across venues

**Plain idea.** Crossing the spread now is certain and costs half the spread
plus a taker fee. Resting is cheaper (we earn the half spread and maybe a
rebate) but fills only if enough flow arrives to eat through the queue in
front of us. Unfilled quantity costs something later. Split the order to
minimise the expected total.

**Formal statement.** Cont and Kukanov (2017, *Quantitative Finance* 17(1))
choose a market order `M` and limit orders `L_k` to minimise
`E[(h+f)M + Σ c_k min(L_k, D_k) + λ_u (S−A)^+ + λ_o (A−S)^+]`, where `D_k`
is the flow that reaches us behind queue `Q_k` and `A` the filled amount.
We add an adverse-selection charge to each `c_k` (lit:
`c_k = −(h + r_k) + a_k`; dark midpoint: `c_k = fee_k + a_k`) and impose
**no oversubscription**, `M + Σ L_k ≤ S`. That is a departure from the
paper, stated in `allocation.py`: it makes the `λ_o` term vanish and turns
the sample-average problem into an exact LP whenever every `c_k ≤ λ_u`.
The router raises `λ_u` linearly from a floor to its horizon value, so it
is patient early and urgent late.

**Worked example.** One venue, `h = 0.5`, `f = 0.3`, `r = 0.2`, `λ_u = 2`
(cents per share), flow behind the queue uniform on 0..9, `S = 8`. The
first-order condition gives the quantile rule
`F_D(L*) = (2h+f+r)/(λ_u+h+r) = 1.5/2.7 ≈ 0.556`, so `L* = 5` and
`M* = 3`. The LP returns exactly `(3, 5)` (doctest in `allocation.py`).

**Checks.** The LP agrees with cvxpy (Clarabel) to 1e-6 on random
instances and on live lit/dark decisions taken by the router; with the
single-venue quantile rule; with the greedy solution when flow is known;
and its reported cost equals the objective evaluated directly.

## 2. Lit venues and routing TCA (W27a)

Four lit venues, each a W25 price-time book with its own fee model and
latency (40 to 250 µs). The sweep takes displayed depth best price first,
ties to the lower taker fee, then the lower latency. Because each IOC
arrives after its venue's latency, a quote seen at decision time can be
gone on arrival; the TCA reports that as a lower fill rate for slow venues.

TCA definitions (`tca.py`): fill rate = filled / sent; effective spread
`2d(p − m)` with `m` the consolidated mid at the fill (the SEC Rule 605
definition); markout at τ `d(m_{t+τ} − p)`; all with standard errors.

## 3. Dark venues (W27b)

| Venue | Rule |
|---|---|
| `DARK_MID` | Continuous midpoint crossing, time priority, no floor. |
| `DARK_BLOCK` | Same, with a 20-lot venue minimum. |
| `DARK_COND` | Conditional orders: a match sends firm-up invitations; failed firm-ups leak the side and count against the participant; participants below a 60 % firm-up rate after 5 invitations are suspended. |
| `DARK_FBA` | Periodic batch auction every 100 ms at the mid (Budish, Cramton and Shim 2015). |

A minimum quantity applies to the size of each match (relaxed to the
remaining quantity when less is left), so it changes *whom we meet*, not just
what is reported. Counterparties: uninformed, informed (they know the drift
regime) and a predator that pings with 1-lot IOCs, fishes with conditionals
it rarely firms up, reads IOIs, and front-runs what it learns on the lit
venue with the lowest latency.

## 4. What the simulation measures

All numbers are means with standard errors across seeded episodes
(episode-clustered: fills inside one episode share a price path). Ticks are
cents per share. Commands in section 5; seeds `range(20)` unless stated.

**Adverse selection by counterparty (1 s markout of our dark fills).**
Informed counterparties: −2.68 ± 1.04 ticks (9 episodes with such fills);
uninformed: −0.92 ± 0.76 (19). Fills against informed flow go further
against us. Uninformed fills are also slightly negative: the router posts
more when the price is moving its way and crosses when it is not, so its
passive and dark fills are selected towards the bad states.

**Minimum quantity changes the fill distribution.** One resting dark buy of
60 lots in `DARK_MID` for 3 s, with the predator pinging:

| min qty | fills / episode | mean fill size (lots) | our lots taken by pings | 1 s markout |
|---|---|---|---|---|
| 0 | 23.4 ± 2.1 | 2.6 ± 0.2 | 19.7 ± 2.1 | +0.46 ± 0.72 |
| 5 | 4.4 ± 0.2 | 13.6 ± 0.9 | 0.15 ± 0.08 | −1.26 ± 0.53 |
| 20 | 2.5 ± 0.1 | 24.0 ± 1.7 | 0.05 ± 0.05 | −0.50 ± 0.83 |

Fewer, larger fills, and the predator's 1-lot pings are shut out (the
residual is the last lot, when the remainder falls below the minimum and the
rule relaxes). The markouts do not separate at 20 episodes.

**IOIs leak.** Router with `DARK_MID`, predator not pinging (IOI channel
only), 150 lots over 4 s:

| IOIs | predator detections of our side | shortfall (ticks) |
|---|---|---|
| none | 0 | 0.94 ± 0.35 |
| actionable | 2.0 ± 0.3 | 1.23 ± 0.43 |
| non-actionable | 4.35 ± 0.49 | 0.95 ± 0.45 |

Non-actionable IOIs reach more recipients and leak about twice as often
(the leak probabilities are assumptions in `Market.ioi`). The shortfall
differences are inside their errors at this sample size; the detections are
the measured leakage.

**Conditional orders and the firm-up penalty.** Firm-up rates on
`DARK_COND`: uninformed 0.93 ± 0.03, informed 0.92 ± 0.04, predator
0.19 ± 0.04, router 1.00. The venue suspended the predator in 20 of 20
episodes. Each invitation still tells the other side ours exists (logged in
`ConditionalPool.leaks`), which is the leakage conditionals trade for size.

**Markout learning moves flow off a toxic venue.** `DARK_MID` made toxic (90 %
informed), `DARK_BLOCK` clean, 300 lots over 8 s. Share of dark allocation
sent to the toxic venue in the second half: 0.54 ± 0.05 without learning,
0.23 ± 0.06 with it; the learned 200 ms score of the toxic venue is
−0.44 ± 0.10 ticks. Shortfall: 0.44 ± 0.61 without, 0.40 ± 0.65 with:
the reallocation is clear, the saving is not measurable at 20 episodes.

**Anti-gaming: can the predator tell we are there?** Held-out AUC of the
predator's detector, 300 episodes per class, seed 2, 95 % percentile
bootstrap intervals:

| router | detector AUC | net-imbalance AUC |
|---|---|---|
| regular (fixed size and interval, no min-qty) | 0.70 (0.64-0.76) | 0.59 |
| randomised size, timing, min-qty | 0.59 (0.53-0.66) | 0.65 |
| randomised, each dark venue in 25 % of intervals | **0.52 (0.45-0.58)** | 0.70 |

Randomising size, timing and min-qty is not enough on its own. Per-feature
AUCs (150 episodes per class) show why: no ping ever took a lot of ours, but
the parent still depletes contra liquidity in the dark, so the predator's
pings find more resting interest on the parent's side (ping feature AUC 0.55),
and the end-of-horizon clean-up sweep is a burst (timing feature 0.57).
Using each dark venue in a quarter of intervals meets the plan's < 0.55
threshold. The price is visible in the last column: more of the order goes
to lit venues (dark share 38 % → 27 % → 9 %), and a detector that only looks
at signed lit volume does better. Randomisation hides the *pattern*, never
the *demand*. Shortfall with the predator present (40 seeds): 0.18 ± 0.14,
0.37 ± 0.21 and 0.34 ± 0.20 ticks; differences inside their errors.

## 5. Reproduce

```
cd python/execution
PYTHONPATH=src python -m pytest src/qp_exec/sor            # about 100 s
QP_SLOW=1 PYTHONPATH=src python -m pytest src/qp_exec/sor -k acceptance   # AUC acceptance, about 15 min
PYTHONPATH=src python -c "from qp_exec.sor import studies as S; print(S.min_qty_effect(range(20)))"
```

`test_matches_cvxpy` and `test_live_lit_dark_split_matches_cvxpy` skip when
cvxpy is not installed.

## References

- Cont, R. and Kukanov, A. (2017). Optimal order placement in limit order markets. *Quantitative Finance* 17(1), 21-39.
- Budish, E., Cramton, P. and Shim, J. (2015). The high-frequency trading arms race: frequent batch auctions as a market design response. *Quarterly Journal of Economics* 130(4), 1547-1621.
- Mittal, H. (2008). Are you playing in a toxic dark pool? A guide to preventing information leakage. *Journal of Trading* 3(3).
- Ye, M. (2011). A glimpse into the dark: price formation, transaction costs and market share of the crossing network. Working paper.
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
- SEC Rule 605 (17 CFR 242.605) for the effective-spread definition; Regulation NMS Rule 610(c) for the access-fee cap.
