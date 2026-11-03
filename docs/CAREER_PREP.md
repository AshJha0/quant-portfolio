# Career prep and learning notebooks

Two kinds of practice material sit next to the code. Both are checked by CI,
so a wrong answer key or a notebook that no longer runs fails the build.

## Learning notebooks

[`docs/notebooks/`](notebooks/README.md) holds eight notebooks, each as an
exercise copy and a solution copy:

| Notebook | Topic | Library code it calls |
|---|---|---|
| 01 | Black-Scholes price, vega and implied volatility | `eq_options` |
| 02 | Value at risk and expected shortfall | `eq_var` |
| 03 | Yield-curve bootstrap | `fi_rates` |
| 04 | Almgren-Chriss optimal execution | `eq_algo` |
| 05 | Greeks by adjoint algorithmic differentiation | `qp_core.aad` |
| 06 | Avellaneda-Stoikov market-making quotes | `qp_exec.mm` |
| 07 | Credit valuation adjustment | `qp_core.xva` |
| 08 | Risk parity and Ledoit-Wolf shrinkage | `qp_bt.portfolio` |

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
risk, trading and engineering roles. Start with the
[study paths](../career-prep/paths/README.md): 4, 8 and 12-week plans and one
path for each of 13 roles, each ending in a timed mock round.

| Group | Packs |
|---|---|
| Maths | [probability](../career-prep/probability/questions.md) (60), [stats](../career-prep/stats/questions.md) (25), [numerics](../career-prep/numerics/questions.md) (25), [stochastic](../career-prep/stochastic/questions.md) (22) |
| Products and risk | [derivatives](../career-prep/derivatives/questions.md) (22), [fixed-income](../career-prep/fixed-income/questions.md) (22), [risk](../career-prep/risk/questions.md) (20 + 4), [portfolio](../career-prep/portfolio/questions.md) (15 + 4) |
| Research | [alpha](../career-prep/alpha/questions.md) (14 + 5), [ml](../career-prep/ml/questions.md) (25) |
| Engineering | [cpp](../career-prep/cpp/questions.md) (24), [algorithms](../career-prep/algorithms/questions.md) (33), [python](../career-prep/python/questions.md), [rust](../career-prep/rust/questions.md), [java](../career-prep/java/questions.md), [sql](../career-prep/sql/questions.md), [tools](../career-prep/tools/questions.md), [system-design](../career-prep/system-design/questions.md) (8) |
| Trading and rounds | [games](../career-prep/games/README.md), [case](../career-prep/case/questions.md) (6 + 6), [mock-rounds](../career-prep/mock-rounds/README.md) (6) |

Numbers in brackets are graded problems (+ discussion prompts). Every key
sits in a separate `keys/` file, and every numeric answer is recomputed by a
second route (a `verify.py`, a compiled solution or a simulation) in
`career-prep/tests/`, which the CI `learning` job runs.

## How to use them

Set a timer, write your answers on paper, then compare with the key or the
solution notebook. For the theory behind each topic, the matching sections
of [LEARN.md](LEARN.md) go further; for the code, each project's
`docs/METHODOLOGY.md`.
