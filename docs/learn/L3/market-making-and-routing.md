# Market making, routing and the risk book (L3)

## Plain idea

A market maker earns half a spread each time someone trades against its
quote, and loses when the price moves against the inventory those trades
leave behind. So it leans its quotes away from the side that would grow the
position and widens them when there is more time for the price to move.

A broker working a client order has the mirror problem. Crossing the spread
fills for certain; resting a limit order is cheaper but may not fill. A smart
order router decides how much to send where. A dealer answering a request for
quote picks a markup that trades the chance of winning against the profit
when it wins. And a bank that runs many desks nets their positions in one
central risk book before it hedges anything outside.

## Formal statement

1. **Avellaneda-Stoikov (2008).** With mid `dS = sigma dW`, fill intensity
   `A exp(-k delta)` at distance `delta` and CARA utility with risk aversion
   `gamma`, a quadratic expansion gives the reservation price
   `r = s - q gamma sigma^2 (T - t)` and the total spread
   `gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)`. Quotes sit at
   `r -/+ spread / 2`.
2. **Gueant, Lehalle and Fernandez-Tapia (2013).** With inventory bounded to
   `{-Q, ..., Q}` the same problem is solved exactly by a matrix exponential
   of a tridiagonal matrix; far from the horizon the quotes become
   stationary, and an asymptotic formula needs no matrix at all.
3. **Cont and Kukanov (2017).** Choose a market order `M` and limit orders
   `L_k` on venues `k` to minimise the expected cost of the fills plus a
   penalty `lambda_u` per unfilled lot. With scenarios for the flow that
   reaches each queue, and the rule that working quantity never exceeds what
   is left (`M + sum L_k <= S`), this is a linear program.
4. **RFQ markup.** If the probability of winning at markup `m` is the
   logistic `p(m) = 1 / (1 + exp(-(a + b m)))`, the dealer maximises
   `m p(m)`.
5. **Central risk book.** Net exposure per instrument is the signed sum of
   the desks' fills; gross is the sum of their absolute sizes. Only the net
   needs hedging outside.

Code: `python/execution/src/qp_exec/mm/quotes.py` (`avellaneda_stoikov`,
`GLFT`, `glft_asymptotic`), `python/execution/src/qp_exec/sor/allocation.py`
(`solve_allocation`), `python/execution/src/qp_exec/rfq/quote.py`
(`optimal_markup`) and `python/execution/src/qp_exec/crb/book.py`
(`CentralRiskBook`). The SOR package has its own guide with measured
results: `python/execution/src/qp_exec/sor/README.md`.

## Worked example

**Avellaneda-Stoikov, the paper's own numbers.** `s = 100`, `gamma = 0.1`,
`sigma = 2`, `k = 1.5`, one unit of time left and a long inventory of 2.
The spread is `0.1 x 4 x 1 + 20 ln(1 + 0.1 / 1.5) = 1.6908`. The reservation
price is `100 - 2 x 0.1 x 4 = 99.2`, so the bid is 98.3546 and the ask
100.0454: both quotes move down, so the maker is more likely to sell.

**GLFT with the same parameters**, `A = 140` and `Q = 20`. Flat, the
stationary exact quotes are 99.3284 and 100.6716; the asymptotic formula
gives 99.3288 and 100.6712, within half a thousandth. Long 2, the exact
reservation price is 99.8952. It moves far less than the Avellaneda-Stoikov
one, because a stationary quote does not depend on a horizon that is one
time unit away.

**Router.** One venue, half spread `h = 0.5`, fee `f = 0.3`, rebate
`r = 0.2`, penalty `lambda_u = 2` (cents a share), flow behind the queue
uniform on 0 to 9, and 8 lots to buy. The program posts 5 lots and sends the
other 3 as a market order.

**RFQ.** With `a = 1` and `b = -200` per unit of markup, the best markup is
0.007836, from `m* = (1 + W(e^(a - 1))) / (-b)` with `W` the Lambert W
function. The markup is in the same unit as `1 / b`.

**Central risk book.** The cash desk buys 10 ES, the derivatives desk sells 7.
Net 3, gross 17: hedging the net trades 3 contracts instead of 17.

Source: every figure is recomputed from the package code in
`docs/learn/tools/test_worked_examples.py::test_market_making_and_routing`.

## Where the repository goes further

- Simulated quoting against a fill model, with inventory limits and quote
  pulling: `qp_exec.mm.sim` and `qp_exec.mm.controls`.
- The limit order book, queue-reactive flow and fill models:
  `python/execution/src/qp_exec/lob/`.
- Dark venues, anti-gaming and the routing studies, with standard errors:
  `python/execution/src/qp_exec/sor/README.md`.
- Hedging policies for the central risk book: `qp_core.hedging`.

## Checks

1. In Avellaneda-Stoikov, what happens to the spread as `gamma` goes to zero?
2. Why does GLFT quote no bid when the inventory is at `+Q`?
3. In the router, why must every `c_k` be at most `lambda_u` for the linear
   program to be exact?
4. A second desk sells 3 more ES into the book above. What is the net and
   what is the gross?

Answers: the inventory term vanishes and the spread tends to `2 / k`, the
risk-neutral maker's; a bid could only grow a position that is already at
its bound; otherwise filling a limit order costs more than leaving it
unfilled, the objective stops being convex and the auxiliary variables are no
longer pushed up to `min(L_k, D_k)`; net 0, gross 20.
