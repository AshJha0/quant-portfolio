# Probability and puzzles: knowledge check (questions only)

Sixty problems, written fresh for this repository, in the style of the
standard quant career-prep books (Zhou 2008; Crack, *Heard on the Street*;
Joshi, Denson and Downes). The answers and worked solutions are in
[`keys/answers.md`](keys/answers.md). Try each one first with pen and paper;
a good target is under five minutes a problem.

Every answer in the key is recomputed independently by
[`verify.py`](verify.py) (enumeration, an exact Markov-chain solve, or a
dynamic programme) and, where cheap, by a seeded simulation. The CI test
`career-prep/tests/test_probability_pack.py` fails if a key and its check
disagree.

Tags: `[ev]` expected value, `[cond]` conditioning and Bayes, `[mart]`
martingales and stopping, `[ruin]` gambler's ruin, `[coupon]`, `[order]` order
statistics, `[markov]`, `[poisson]`, `[reflect]` reflection and paths,
`[comb]` combinatorics, `[normal]`.

## Expected value and patterns

- **P01** `[ev]` What is the expected value of one roll of a fair die?
- **P02** `[ev]` Two fair dice are rolled. What is the expected value of the larger one?
- **P03** `[ev]` How many rolls of a fair die do you need, on average, to see the first 6?
- **P04** `[coupon]` How many rolls, on average, until every face of a fair die has appeared?
- **P05** `[markov]` A fair coin is flipped until HH appears. Expected number of flips?
- **P06** `[markov]` Same question for HT. Why is it different from P05?
- **P07** `[markov]` Flip a fair coin until HH or HT appears. What is the probability HH comes first?
- **P08** `[markov]` Penney's game: what is the probability that THH appears before HHH?
- **P60** `[markov]` Expected number of fair-coin flips until three heads in a row?

## Conditioning and Bayes

- **P09** `[cond]` A family has two children and at least one is a boy. Probability both are boys?
- **P10** `[cond]` Monty Hall, three doors, host always opens an empty unpicked door and offers a switch. Probability of winning if you switch?
- **P11** `[cond]` A disease has 1% prevalence. The test has 99% sensitivity and 95% specificity. You test positive. Probability you have the disease?

## Gambler's ruin and random walks

- **P12** `[ruin]` Fair +1/-1 game, start with 3, stop at 0 or 10. Probability of reaching 10?
- **P13** `[ruin]` As P12, but you win each round with probability 0.6. Probability of reaching 10?
- **P14** `[ruin]` In P12, expected number of rounds until the game stops?
- **P15** `[ruin]` A symmetric random walk starts at 0. Expected steps until it first hits +5 or -5?
- **P36** `[markov]` A symmetric walk on {0,...,4} reflects at 0 (always steps to 1) and stops at 4. Expected steps from 0?
- **P37** `[reflect]` Simple symmetric walk of 10 steps from 0. Probability its running maximum reaches at least 3?
- **P38** `[reflect]` An election: A gets 6 votes, B gets 4, counted in uniformly random order. Probability A is strictly ahead throughout the count?
- **P39** `[comb]` How many up/down paths of 10 steps start and end at 0 and never go below 0?
- **P40** `[mart]` Fair +1/-1 game from 0. Probability you hit +1 before -2?

## Coupons, birthdays and permutations

- **P16** `[coupon]` Expected number of draws (with replacement) to collect all 10 coupon types?
- **P17** `[comb]` Smallest group size for which the chance of a shared birthday exceeds 1/2 (365 equally likely days)?
- **P18** `[comb]` The probability of a shared birthday among 23 people?
- **P19** `[ev]` Expected number of fixed points of a uniformly random permutation of 6 items?
- **P20** `[comb]` Probability a uniformly random permutation of 8 items has no fixed point?

## Uniforms and order statistics

- **P21** `[order]` Expected maximum of three independent U(0,1)?
- **P22** `[order]` Expected minimum of three independent U(0,1)?
- **P23** `[order]` Expected second-smallest of five independent U(0,1)?
- **P24** `[order]` A unit stick is broken at two independent uniform points. Probability the three pieces form a triangle?
- **P25** `[ev]` A unit stick is broken at one uniform point. Expected length of the piece containing the midpoint?
- **P26** `[ev]` Expected distance between two independent U(0,1)?
- **P27** `[ev]` Draw U(0,1) values until their running sum exceeds 1. Expected number of draws?

## Poisson processes and exponentials

- **P28** `[poisson]` Arrivals at rate 3 per hour. Probability of none in 30 minutes?
- **P29** `[poisson]` Given exactly 4 arrivals in [0,1], probability all fall in [0, 1/2]?
- **P30** `[poisson]` Independent streams at rates 2 and 3. Probability the next arrival is from the first?
- **P31** `[poisson]` Expected time until the first of two independent exponentials with rates 2 and 3?
- **P32** `[poisson]` A wait is exponential with mean 10 minutes. You have waited 5. Expected remaining wait?
- **P33** `[poisson]` Buses arrive as a Poisson process with mean gap 10 minutes. You turn up at a random time. Expected length of the gap you land in?

## Markov chains

- **P34** `[markov]` Two states, P(0 to 1) = 0.3, P(1 to 0) = 0.1. Long-run fraction of time in state 1?
- **P35** `[markov]` In P34, expected return time to state 0?

## Martingales, Wald and urns

- **P41** `[mart]` Roll a die until the first 6. Expected total of all rolls, the 6 included?
- **P42** `[mart]` Polya urn: 1 red, 1 blue. Draw, return with one more of the same colour, 10 times. Probability exactly 5 reds are added?
- **P43** `[ev]` Expected number of heads before the first tail with a fair coin?
- **P44** `[ev]` Expected number of runs in 10 fair coin flips (HHTH has 3 runs)?
- **P45** `[ev]` Expected number of (overlapping) HH pairs in 10 fair flips?

## Cards and dice

- **P46** `[comb]` Two cards from a shuffled 52-card deck. Probability both are aces?
- **P47** `[comb]` Probability a 5-card hand is a flush (straight flushes included)?
- **P48** `[comb]` Probability a 5-card hand is a full house?
- **P49** `[cond]` Two dice rolled repeatedly. Probability a total of 7 appears before a total of 6?
- **P50** `[cond]` Probability of winning at craps (7 or 11 wins at once; 2, 3, 12 lose; else roll for the point before a 7)?
- **P51** `[ev]` Expected number of distinct faces seen in 6 rolls of a die?

## Betting and stopping

- **P52** `[ev]` Even-money bet won with probability 0.6. Kelly fraction of wealth to stake?
- **P53** `[mart]` You are paid the face of a die roll and may reroll once (keeping the second). Value of the game?
- **P54** `[mart]` As P53 but with up to two rerolls?
- **P55** `[mart]` Secretary problem with 10 candidates: success probability of the best cut-off rule?

## Normal distribution

- **P56** `[normal]` Expected maximum of two independent standard normals?
- **P57** `[normal]` X, Y standard normal with correlation 0.5. Probability both are positive?
- **P58** `[normal]` Expected value of |Z| for standard normal Z?
- **P59** `[reflect]` Standard Brownian motion on [0,1]. Probability its maximum reaches 1?
