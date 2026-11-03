# Career prep and learning notebooks

Two kinds of practice material sit next to the code. Both are checked by CI,
so a wrong answer key or a notebook that no longer runs fails the build.

## Learning notebooks

[`docs/notebooks/`](notebooks/README.md) holds four notebooks, each as an
exercise copy and a solution copy:

| Notebook | Topic | Library code it calls |
|---|---|---|
| 01 | Black-Scholes price, vega and implied volatility | `eq_options` |
| 02 | Value at risk and expected shortfall | `eq_var` |
| 03 | Yield-curve bootstrap | `fi_rates` |
| 04 | Almgren-Chriss optimal execution | `eq_algo` |

The plain idea comes first, then the formal statement with its textbook
source, then a worked example whose numbers are recomputed independently in
`docs/notebooks/tests/`. The exercise copies keep the checks and hide the
solutions. Both copies are generated from `docs/notebooks/build_notebooks.py`;
edit that file, not the `.ipynb` files, and run
`python docs/notebooks/build_notebooks.py --check` to confirm they match.

CI (`learning` job) runs `pytest docs/notebooks/tests` and executes every
`*-solution.ipynb` with nbmake on the pinned `python3` kernel.

## Career-prep packs

[`career-prep/`](../career-prep/README.md) holds practice packs for quant,
risk, trading and engineering roles:

- **Probability:** 60 problems, with the answer key in a separate file.
  Every answer is recomputed twice, by enumeration or an exact solve and by
  a seeded simulation (`career-prep/probability/verify.py`).
- **Games:** a mental-maths drill generator and a market-making game on the
  sum of three dice.

CI (`learning` job) runs `pytest career-prep/tests`. More packs are planned
for later rounds of the v3.0.0 plan; until they exist, they are not listed
here.

## How to use them

Set a timer, write your answers on paper, then compare with the key or the
solution notebook. For the theory behind each topic, the matching sections
of [LEARN.md](LEARN.md) go further; for the code, each project's
`docs/METHODOLOGY.md`.
