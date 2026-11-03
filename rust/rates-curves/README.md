# qp-rates-curves

Dated OIS, multi-curve and cross-currency curve bootstrap (plan workstream
W13b). It is the Rust port of `fi_rates.bootstrap.bootstrap_dated`,
`fi_rates.multicurve.bootstrap_projection_dated` and
`fx_rates.xccy.bootstrap_xccy_discount`, and the C++ port lives in
`cpp/rates-curves`. All three use the same algorithm and the same loop
order. std only.

**The idea.** Each quote (a deposit, a future, an FRA, an OIS, a term swap
or a cross-currency basis swap) adds one pillar date to the curve. The
discount factor on that date is chosen so that the quote is reproduced
exactly.

**Formal statement.**
- Curve time is `t = days/365`. `ln P` is linear in `t` between pillars,
  starting from `P(0) = 1`. After the last pillar the last forward is held
  flat.
- A forward quote satisfies `(P(s)/P(e) - 1)/tau = R`.
- A swap satisfies `sum_j D(p_j)(P(s_j)/P(e_j) - 1) = K sum_i tau_i D(p_i)`.
- A constant-notional basis swap with the spread on the foreign leg
  satisfies `V_f/X(s0) = V_d/D(s0)` (Fujii, Shimada and Takahashi 2010,
  sec. 4).
- Each pillar is solved by bisection on `ln P` in `[ln 1e-8, ln 20]`.

**Oracle.** `tests/fixture.rs` reads
`python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt`, which
was written by QuantLib 1.44. The test rebuilds three curves from it: USD
SOFR OIS with Hull-White-adjusted futures, a 3M term projection curve
discounted on OIS, and EUR under USD collateral from EURUSD basis swaps.
Every pillar must match QuantLib to 1e-12 in discount factor, and every
quote must reprice to 1e-10 in rate. The quotes are illustrative, not
market data.

    cargo test -p qp-rates-curves
