# L0: orientation

## Plain idea

A trading firm holds positions whose value moves with markets. Almost every
project in this repository answers one of three questions about those
positions: what is it worth now (pricing), how much could it lose (risk), and
how do we trade it without giving money away (execution). You do not need any
maths to see which question a project answers.

## Formal statement

At the end of L0 you can:

1. name, for each of the ten areas in [README.md](../../../README.md), whether
   it prices, measures risk or trades;
2. read a one-day 99 % value at risk (VaR) as "on 99 days in 100 the loss
   should be smaller than this", and say what it does not tell you (how bad
   the other day is);
3. tell a validated number from an illustrative one: this repository labels
   synthetic results, and the label matters.

## Worked example

A $1,000,000 position in one stock has a daily volatility of 2 %. If daily
returns were normal with mean zero, the 99 % one-day VaR is the 1 % quantile
of the loss:

- the 99 % point of the standard normal is 2.3263;
- VaR = 2.3263 x 0.02 x $1,000,000 = **$46,527**.

So on about 1 day in 100 the position should lose more than $46,527. It says
nothing about whether that day loses $50,000 or $500,000; expected shortfall
(L2) answers that. Real daily returns have fatter tails than the normal, so
this number is usually too small at 99 %; Set 1 Q3 in the appendix explains
why.

Source: the same formula is `var_parametric` in
`python/foundations/01-risk-metrics/src/eq_risk_metrics/var_es.py` (with the
sample mean and standard deviation in place of 0 and 2 %); the figure is
checked in
`docs/learn/tools/test_worked_examples.py::test_l0_normal_var`.

## Checks

1. Which of pricing, risk or execution does `python/equity/08-algo-execution`
   answer?
2. If the daily volatility doubles, what happens to the normal VaR above?
3. Why is "VaR was $46,527, so we cannot lose more than that" wrong?

Answers: execution; it doubles to $93,054; VaR is a quantile, not a bound.

Next: [L0 questions](QUESTIONS.md), then [LEARN.md](../../LEARN.md) Part I
and [START_HERE.md](../../START_HERE.md). When those read easily, move to
[L1](../L1/README.md).
