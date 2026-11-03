# Career prep

Practice material for quant, risk, trading and engineering roles, built
around the code in this repository. Questions and answer keys live in
separate files, so you can work a pack without seeing the solutions. Every
numeric answer is recomputed by a script or simulation that CI runs
(`career-prep/tests/`); if a key and its check disagree, the build fails.

| Pack | What it covers | Questions | Key | Checked by |
| --- | --- | --- | --- | --- |
| [probability](probability/) | 60 problems: expected value, conditioning and Bayes, martingales and stopping, gambler's ruin, coupon collector, order statistics, Markov chains, Poisson processes, reflection | [questions](probability/questions.md) | [keys](probability/keys/answers.md) | [`verify.py`](probability/verify.py) |
| [games](games/) | Mental-maths drill generator; market-making game on the sum of three dice | run the scripts | printed at the end / in the docstring | `tests/test_games.py` |

More packs (statistics, numerics, ML, stochastic calculus, derivatives,
fixed income, C++, algorithms, system design, mock rounds and study paths)
arrive in later rounds of the v3.0.0 plan.

How to use a pack: set a timer, write answers on paper, then compare with
the key. Run `python career-prep/probability/verify.py` to see each answer
recomputed. All problems are written fresh. For more practice in the same
style, see Zhou (2008), Crack (*Heard on the Street*) and Joshi, Denson and
Downes (2008); these are cited for style only, not as sources of the text.
