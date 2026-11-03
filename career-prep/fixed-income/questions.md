# Fixed-income maths by hand: knowledge check (questions only)

Twenty-two problems on bond pricing, duration and convexity, curve building,
swaps, a short-rate model and caplets, written fresh for this repository in
the style of Tuckman and Serrat (*Fixed Income Securities*, 3rd ed.) and
Hull (chs. 4, 7, 29, 31). Answers are in [`keys/answers.md`](keys/answers.md).

The key uses textbook identities (annuity sums, duration formulas, the
Vasicek and Black closed forms). [`verify.py`](verify.py) recomputes each
answer differently: cash flow by cash flow in exact rational arithmetic,
finite differences of the price, root-finding, a linear solve, a numerical
ODE or quadrature, with a seeded simulation for the Vasicek bond.
`career-prep/tests/test_quant_packs.py` fails if the key and a check
disagree. Rational answers are given exactly.

Conventions: face 100, annual coupons and annual compounding unless stated.

Tags: `[price]`, `[risk]` duration, convexity and DV01, `[curve]` zeros,
forwards and bootstrapping, `[swap]`, `[model]`.

## The reference bond

Five-year bond, 6% annual coupon, priced at a 5% yield.

- **F01** `[price]` Its price.
- **F02** `[risk]` Its Macaulay duration.
- **F03** `[risk]` Its modified duration.
- **F04** `[risk]` Its convexity.
- **F05** `[risk]` Its DV01 (price change per 100 face for a 1 basis point fall in yield).
- **F22** `[price]` How much does it lose if the yield jumps to 6%?

## Pricing and yields

- **F11** `[price]` Two-year bond, 4% coupon paid semiannually, yield 5% with semiannual compounding. Price?
- **F12** `[price]` Three-year bond, 5% annual coupon, price 98. Yield to maturity?
- **F13** `[price]` Ten-year zero-coupon bond at a 4% yield. Price?
- **F14** `[risk]` Its modified duration?
- **F21** `[risk]` Its convexity?
- **F15** `[risk]` A portfolio holds 40% of its value in the F13 zero and 60% in the reference bond. Its modified duration?

## Curves

- **F06** `[curve]` The two-year discount factor is 0.9. Two-year zero rate, annual compounding?
- **F10** `[curve]` What continuously compounded rate is equivalent to 5% annual?
- **F07** `[curve]` One-year zero 4%, two-year zero 5%, annual compounding. The one-year rate one year forward?
- **F08** `[curve]` A one-year zero trades at 96 and a two-year 5% annual-coupon bond at 99. Bootstrap the two-year discount factor.
- **F09** `[curve]` From the F08 curve, the two-year par yield?

## Swaps and rate derivatives

Annual discount factors 0.97, 0.94 and 0.91 for one, two and three years.

- **F17** `[swap]` The three-year annual par swap rate.
- **F18** `[swap]` Value, per 100 notional, of receiving 4% fixed against floating on that swap.
- **F16** `[swap]` A 3x6 FRA on 1,000,000 at 5% (accrual 0.25 years). At fixing the rate is 6%. Settlement amount to the buyer, paid at the start of the period?
- **F19** `[model]` Vasicek dr = 0.5 (0.05 - r) dt + 0.01 dW, r_0 = 3%. Price of a two-year zero paying 1.
- **F20** `[model]` A caplet on 1,000,000: forward rate 5%, strike 5%, Black vol 20%, fixing in one year, accrual 0.25, payment discount factor 0.94. Price?
