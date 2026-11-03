# Career prep

Start with [the study paths](paths/README.md): 4, 8 and 12-week plans and
one path per role, each ending in a mock round.

Practice material for quant, risk, trading and engineering roles, built
around the code in this repository. Questions and answer keys live in
separate files, so you can work a pack without seeing the solutions. Every
numeric answer is recomputed by a script or simulation that CI runs
(`career-prep/tests/`); if a key and its check disagree, the build fails.

| Pack | What it covers | Questions | Key | Checked by |
| --- | --- | --- | --- | --- |
| [probability](probability/) | 60 problems: expected value, conditioning and Bayes, martingales and stopping, gambler's ruin, coupon collector, order statistics, Markov chains, Poisson processes, reflection | [questions](probability/questions.md) | [keys](probability/keys/answers.md) | [`verify.py`](probability/verify.py) |
| [games](games/) | Mental-maths drill generator; market-making game on the sum of three dice | run the scripts | printed at the end / in the docstring | `tests/test_games.py` |
| [cpp](cpp/) | 24 items: 11 "what does it print" snippets, 9 components to write (SPSC ring, blocking queue, object pool, spinlock, thread pool, fixed-point price, seqlock, concepts), 4 to explain | [questions](cpp/questions.md) | [keys](cpp/keys/answers.md) | `tests/test_cpp_pack.py` (snippets run; solutions under ASan, UBSan and TSan) |
| [algorithms](algorithms/) | 33 problems: hashing, monotonic stacks and queues, heaps, binary search, DP, graphs, union-find, KMP, Fenwick trees, an order book and rolling VWAP | [questions](algorithms/questions.md) | [keys](algorithms/keys/answers.md) | `tests/test_algorithms_pack.py` (Python and C++ against brute-force oracles) |
| [stats](stats/) | 25 problems: estimators and standard errors, regression, persistence, multiple testing | [questions](stats/questions.md) | [keys](stats/keys/answers.md) | [`verify.py`](stats/verify.py), `tests/test_stats_pack.py` |
| [numerics](numerics/) | 25 problems: eigenvalues and PCA, PSD repair, LU/QR/Cholesky and conditioning, root finding, quadrature, finite differences and Crank-Nicolson stability | [questions](numerics/questions.md) | [keys](numerics/keys/answers.md) | [`verify.py`](numerics/verify.py), `tests/test_numerics_pack.py` |
| [ml](ml/) | 25 problems: purged and embargoed cross-validation, overlapping labels, regularisation, tree ensembles, leakage, feature importance | [questions](ml/questions.md) | [keys](ml/keys/answers.md) | [`verify.py`](ml/verify.py), `tests/test_ml_pack.py` |
| [stochastic](stochastic/) | 22 problems: Brownian motion, Ito calculus, the processes used in pricing | [questions](stochastic/questions.md) | [keys](stochastic/keys/answers.md) | [`verify.py`](stochastic/verify.py), `tests/test_quant_packs.py` |
| [derivatives](derivatives/) | 22 problems: option pricing, Greeks, trees, simple exotics | [questions](derivatives/questions.md) | [keys](derivatives/keys/answers.md) | [`verify.py`](derivatives/verify.py), `tests/test_quant_packs.py` |
| [fixed-income](fixed-income/) | 22 problems by hand: bond pricing, duration and convexity, curve building, swaps, a short-rate model, caplets | [questions](fixed-income/questions.md) | [keys](fixed-income/keys/answers.md) | [`verify.py`](fixed-income/verify.py), `tests/test_quant_packs.py` |
| [risk](risk/) | 20 graded problems on market and credit risk, 4 discussion prompts | [questions](risk/questions.md) | [keys](risk/keys/answers.md) | [`verify.py`](risk/verify.py), `tests/test_numeric_packs.py` |
| [portfolio](portfolio/) | 15 graded problems, 4 discussion prompts on portfolio construction | [questions](portfolio/questions.md) | [keys](portfolio/keys/answers.md) | [`verify.py`](portfolio/verify.py), `tests/test_numeric_packs.py` |
| [alpha](alpha/) | 14 graded problems, 5 discussion prompts: signal evaluation, significance, overfitting | [questions](alpha/questions.md) | [keys](alpha/keys/answers.md) | [`verify.py`](alpha/verify.py), `tests/test_numeric_packs.py` |
| [python](python/) | 14 "what does this print" snippets and 6 coding tasks | [questions](python/questions.md) | [keys](python/keys/answers.md) | `tests/test_python_pack.py` |
| [sql](sql/) | 10 query tasks on a trades/quotes/closes set (SQLite), 3 discussion prompts | [questions](sql/questions.md) | [keys](sql/keys/answers.md) | `tests/test_sql_pack.py` |
| [rust](rust/) | 14 short programs (output or compiler error) and 3 tasks with unit tests | [questions](rust/questions.md) | [keys](rust/keys/answers.md) | `tests/test_compiled_packs.py` |
| [java](java/) | 12 short programs and tasks on the language basics behind the Java gateway | [questions](java/questions.md) | [keys](java/keys/answers.md) | `tests/test_compiled_packs.py` |
| [tools](tools/) | 12 one-line shell tasks; debugging and production-support prompts | [questions](tools/questions.md) | [keys](tools/keys/answers.md) | `tests/test_tools_pack.py` |
| [system-design](system-design/) | 8 design cases, one per component of a trading stack, each with a graded sizing question | [questions](system-design/questions.md) | [keys](system-design/keys/answers.md) | [`verify.py`](system-design/verify.py), `tests/test_final_packs.py` |
| [case](case/) | 6 trading cases with a numeric answer, 6 behavioural prompts | [questions](case/questions.md) | [keys](case/keys/answers.md) | [`verify.py`](case/verify.py), `tests/test_final_packs.py` |
| [mock-rounds](mock-rounds/) | 6 timed rounds, one per role family (4 graded, 2 discussion each) | [rounds](mock-rounds/README.md) | [keys](mock-rounds/keys/) | [`verify.py`](mock-rounds/verify.py), `tests/test_final_packs.py` |

How to use a pack: set a timer, write answers on paper, then compare with
the key. Run `python career-prep/probability/verify.py` to see each answer
recomputed. All problems are written fresh. For more practice in the same
style, see Zhou (2008), Crack (*Heard on the Street*) and Joshi, Denson and
Downes (2008); these are cited for style only, not as sources of the text.

The C++ checks need `g++` (GCC 13 or later for `-std=c++20` with `<ranges>`
and `std::jthread`). On Windows without it the tests fall back to
`wsl g++`; with neither, they are skipped locally and fail on CI.
