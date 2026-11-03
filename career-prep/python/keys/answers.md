# Python: answer key

Questions are in [`../questions.md`](../questions.md). The test runs each
snippet and compares its output with the text in backticks.

- **Y01** `[1, 2]`. The default list is created once, when `def` runs, and shared by every call. Use `acc=None` and create the list inside.
- **Y02** `[2, 2, 2]`. Each lambda looks `i` up when called, after the loop ended. Bind it now with `lambda i=i: i`.
- **Y03** `[[1, 0, 0], [1, 0, 0]]`. `* 2` repeats a reference to the same inner list. Use `[[0] * 3 for _ in range(2)]`.
- **Y04** `False True`. 0.1 and 0.2 are not exact in binary; compare with a tolerance (`math.isclose`).
- **Y05** `14 0`. The first `sum` exhausts the generator.
- **Y06** `unbound`. Assignment anywhere in a function makes the name local for the whole body (Language Reference 4.2.2), so `n += 1` reads an unbound local. Use `global` or pass the value in.
- **Y07** `2`. A `return` in `finally` replaces the pending return value (Language Reference 8.4).
- **Y08** `0 2 2`. `round` uses round-half-to-even. For money use `decimal` with an explicit rounding mode.
- **Y09** `-4 1`. Floor division rounds toward minus infinity, and `a % b` has the sign of `b`. C++, Rust and Java truncate toward zero and give `-3 -1`.
- **Y10** `{'k': [1, 2]}`. `copy.copy` copies the dict, not the list inside it; use `copy.deepcopy`.
- **Y11** `(3, 4)`. Shapes (3, 1) and (4,) broadcast to (3, 4).
- **Y12** `[0, 99, 2, 3, 4]`. Basic slicing returns a view on the same memory; use `.copy()`.
- **Y13** `[nan, nan, 0.01, -0.0198]`. `pct_change` at t uses the close at t, so a signal traded at t must be shifted by one bar. Forgetting the shift is the most common look-ahead bug.
- **Y14** `[3, 7]`. Bins are left-closed and labelled by their left edge by default: 09:00-09:59 holds 1 and 2, 10:00-10:59 holds 3 and 4. Check `closed` and `label` against how your bars are stamped.

## Discussion notes

- **YD1** Two threads cannot run Python bytecode at the same time. Ways round it: processes (`multiprocessing`, `concurrent.futures.ProcessPoolExecutor`), native code that releases the GIL (NumPy, Numba, Cython, Rust via PyO3 as in this repository's bindings), or async I/O when the work is waiting rather than computing. PEP 703 free-threaded builds exist from 3.13 but most wheels assume the GIL.
- **YD2** It calls a Python function per row. Use vectorised column operations, `groupby` with built-in aggregations, or NumPy on `.to_numpy()` arrays.
- **YD3** Pin the environment (a lock file, as `uv.lock` here), fix every random seed, record the data snapshot and code commit with the result, and keep a script that recreates each figure from raw data.
