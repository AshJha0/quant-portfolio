# research/notebooks: research notebooks (W23 part 3)

Three notebooks, each as an exercise (tasks left as stubs, checks kept) and a
solution. Both files are written by `build_notebooks.py` from one cell list,
so they cannot drift; `python research/notebooks/build_notebooks.py --check`
fails if a committed file is stale.

| Notebook | What it teaches |
|---|---|
| `01-vix-variance-premium` | Implied against forward realised variance with `research/vix/vix_rv.py`; forward window by hand. |
| `02-time-series-momentum` | The TSMOM weight by hand and the portfolio with `research/tsmom/tsmom.py`. |
| `03-purged-cross-validation` | On i.i.d. returns with overlapping 20-day labels, shuffled k-fold scores R2 of about 0.12 and purged, blocked k-fold about -0.12: the gap is leakage (Lopez de Prado 2018, ch. 7). |

All three use seeded synthetic data only: deterministic, offline, and no
hold-out data. Real-data runs go through the study modules behind
`research/panel/holdout_lock.py`.

Run the tests: `python -m pytest research/notebooks/tests`; run the
solutions in a kernel: `python -m pytest --nbmake research/notebooks/*-solution.ipynb`.
