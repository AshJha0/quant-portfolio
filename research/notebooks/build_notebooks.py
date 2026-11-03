"""Build the research notebooks from one source of truth (W23 part 3).

Same layout as ``docs/notebooks/build_notebooks.py``: each notebook exists as
``<name>-exercise.ipynb`` (tasks left as stubs) and ``<name>-solution.ipynb``,
both written from the cell lists below.

    python research/notebooks/build_notebooks.py          # rewrite the .ipynb files
    python research/notebooks/build_notebooks.py --check  # fail if they are stale

The notebooks use the research code in this repository, numpy, pandas and
seeded synthetic data, so a run is deterministic and needs no network and
no hold-out data. Real-data runs go through the study modules, behind the
hold-out lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATHS = "import _paths  # noqa: F401  (puts research/ modules on sys.path)"
STUB_FOOTER = "# Exercise: replace the ... above, then run the check cell below."

Cell = tuple[str, str, str]


def md(text: str) -> Cell:
    return ("md", text.strip("\n"), "")


def code(text: str) -> Cell:
    return ("code", text.strip("\n"), "")


def task(solution: str, stub: str) -> Cell:
    return ("task", solution.strip("\n"), stub.strip("\n"))


def check(text: str) -> Cell:
    return ("check", text.strip("\n"), "")


NOTEBOOKS: dict[str, list[Cell]] = {}

NOTEBOOKS["01-vix-variance-premium"] = [
    md("""
# The variance premium: VIX against realised variance

**Plain idea.** The VIX is a price for the next 30 days of S&P 500 variance,
quoted as a volatility. Square it and compare it with the variance that then
happened. On average the quote is higher: buyers of crash insurance pay a
premium (Carr and Wu 2009, RFS 22(3)).

**What this notebook uses.** Synthetic data only, so it runs offline. The
real-data study is `research/vix/vix_rv.py` (CBOE published VIX level and
the French daily market return; no option prices).
"""),
    code(PATHS + """
import numpy as np
import pandas as pd
import vix_rv
"""),
    md("""
## Simulated market

Daily volatility follows a slow random walk in logs. The "VIX" quotes the
true volatility plus a fixed premium of 3 vol points, so we know the answer.
"""),
    code("""
rng = np.random.default_rng(2026)
days = pd.bdate_range("2005-01-03", periods=2000)
log_vol = np.log(0.18) + np.cumsum(rng.normal(0, 0.03, len(days)))
log_vol = np.log(0.18) + 0.97 * (log_vol - np.log(0.18))
vol = pd.Series(np.exp(log_vol), index=days)
ret = pd.Series(rng.normal(0, 1, len(days)) * vol / np.sqrt(252), index=days)
vix = 100 * (vol + 0.03)
"""),
    md("""
## Task 1: forward realised variance by hand

Compute `rv_hand`: for each day, `252/21` times the sum of the squared
returns over the **next** 21 days (not including today). Use a plain loop.
"""),
    task("""
r2 = ret.to_numpy() ** 2
rv_hand = np.full(len(r2), np.nan)
for t in range(len(r2) - 21):
    rv_hand[t] = 252 / 21 * r2[t + 1:t + 22].sum()
""", """
r2 = ret.to_numpy() ** 2
rv_hand = ...
"""),
    check("""
lib = vix_rv.forward_realised_variance(ret).to_numpy()
assert np.allclose(rv_hand, lib, equal_nan=True, rtol=1e-12, atol=0)
"""),
    md("""
## Task 2: the study

Run `vix_rv.vix_study(vix, ret)` into `res` and print the mean premium in
variance units and the Mincer-Zarnowitz slope. The lock passes because all
dates are before 2016.
"""),
    task("""
res = vix_rv.vix_study(vix, ret)
print(f"mean VRP {res['mean_vrp']:.4f}  NW t {res['nw_t_vrp']:.1f}  "
      f"MZ slope {res['mincer_zarnowitz']['slope']:.2f}")
""", """
res = ...
"""),
    check("""
assert res["mean_vrp"] > 0
assert res["nw_t_vrp"] > 2
assert "no option prices" in res["label"]
"""),
    md("""
**What to take away.** The premium is positive by construction here. On real
data it is positive on average but with large negative spikes when realised
variance jumps; that tail is what the sellers are paid for.
"""),
]

NOTEBOOKS["02-time-series-momentum"] = [
    md("""
# Time-series momentum across assets

**Plain idea.** Long an asset after a year of positive excess return, short
after a year of negative, scaled to equal risk (Moskowitz, Ooi and Pedersen
2012, JFE 104(2)). The real-data study is `research/tsmom/tsmom.py`.
"""),
    code(PATHS + """
import numpy as np
import pandas as pd
import tsmom
"""),
    md("""
## Simulated assets with a planted trend

Four assets. Each has a slow AR(1) drift that persists for months, plus
noise; the trend is what TSMOM is meant to find.
"""),
    code("""
rng = np.random.default_rng(7)
months = pd.date_range("1980-01-31", periods=420, freq="ME")
drift = np.zeros((len(months), 4))
for t in range(1, len(months)):
    drift[t] = 0.95 * drift[t - 1] + rng.normal(0, 0.003, 4)
rx = pd.DataFrame(drift + rng.normal(0, 0.04, (len(months), 4)), index=months,
                  columns=["eq", "fx1", "fx2", "bond"])
"""),
    md("""
## Task 1: the weight by hand

For asset `eq` at the last month, compute `w_hand` = sign of the trailing
12-month sum times `0.40 / sigma`, with `sigma = tsmom.ewma_vol(rx)` at the
same month.
"""),
    task("""
sigma = tsmom.ewma_vol(rx)
w_hand = np.sign(rx["eq"].iloc[-12:].sum()) * 0.40 / sigma["eq"].iloc[-1]
""", """
sigma = tsmom.ewma_vol(rx)
w_hand = ...
"""),
    check("""
assert abs(w_hand - tsmom.weights(rx)["eq"].iloc[-1]) < 1e-12
"""),
    md("""
## Task 2: the portfolio

Run `tsmom.tsmom(rx)` into `res`; print the portfolio Sharpe and turnover.
"""),
    task("""
res = tsmom.tsmom(rx)
s = res["stats"]["TSMOM"]
print(f"Sharpe {s['sharpe_annual']:.2f}  NW t {s['nw_t']:.1f}  turnover {res['mean_turnover']:.2f}")
""", """
res = ...
"""),
    check("""
assert res["stats"]["TSMOM"]["sharpe_annual"] > 0
assert res["label"] == "exploratory"
"""),
    md("""
**What to take away.** With a planted trend the strategy earns money. It
loses at turning points (see the worked example in `tsmom.py`), and on
real data most of its return came in a few long trends.
"""),
]

NOTEBOOKS["03-purged-cross-validation"] = [
    md("""
# Purged cross-validation: why plain k-fold flatters a forecast

**Plain idea.** A label built from the next 20 days of returns overlaps the
labels of the 19 neighbouring days. In plain k-fold, test days sit next to
training days whose labels share returns with them, so the model is scored
on data it has partly seen. Purging drops training days whose label window
overlaps the test fold; an embargo drops a few more after it (Lopez de
Prado 2018, *Advances in Financial Machine Learning*, ch. 7).

**What this notebook shows.** On data with **no** predictability, shuffled
k-fold (the default in most ML tutorials) finds some; purged, blocked
k-fold does not. A k-nearest-neighbour regression in numpy only.
"""),
    code(PATHS + """
import numpy as np
"""),
    code("""
rng = np.random.default_rng(11)
n, h, warm = 1500, 20, 60
r = rng.normal(0, 0.01, warm + n + h)          # i.i.d.: nothing to predict
days = np.arange(warm, warm + n)
y = np.array([r[t + 1:t + 1 + h].sum() for t in days])  # overlapping forward label
# features: trailing 5-, 20- and 60-day returns, known at day t
X = np.array([[r[t - k + 1:t + 1].sum() for k in (5, 20, 60)] for t in days])


def knn_r2(tr, te, k=25):
    # k-nearest-neighbour regression: neighbours in feature space are
    # mostly neighbouring days, whose labels overlap the test label
    d = ((X[te][:, None, :] - X[tr][None, :, :]) ** 2).sum(axis=2)
    nearest = np.argsort(d, axis=1, kind="stable")[:, :k]
    pred = y[tr][nearest].mean(axis=1)
    return 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
"""),
    md("""
## Task 1: purged folds

Write `purged_train(test, h, embargo)`: all indices not in `test`, and not
within `h` days before the test fold's first day or `h + embargo` days
after its last day.
"""),
    task("""
def purged_train(test, h, embargo):
    lo, hi = test.min() - h, test.max() + h + embargo
    idx = np.arange(n)
    return idx[(idx < lo) | (idx > hi)]
""", """
def purged_train(test, h, embargo):
    ...
"""),
    code("""
shuffled = np.array_split(np.random.default_rng(3).permutation(n), 5)
plain = [knn_r2(np.setdiff1d(np.arange(n), f), np.sort(f)) for f in shuffled]
folds = np.array_split(np.arange(n), 5)
purged = [knn_r2(purged_train(f, h, 10), f) for f in folds]
print(f"shuffled k-fold mean R2 {np.mean(plain):.3f}; purged blocked {np.mean(purged):.3f}")
"""),
    check("""
f = folds[2]
tr = purged_train(f, h, 10)
assert not np.intersect1d(tr, f).size
assert tr[tr < f.min()].max() == f.min() - h - 1
assert tr[tr > f.max()].min() == f.max() + h + 10 + 1
assert np.mean(plain) > 0.05   # leakage looks like skill
assert np.mean(purged) < 0.0   # purged: no skill, as planted
"""),
    md("""
**What to take away.** The returns are i.i.d., so no forecast can work, yet
the shuffled score is above zero (about 0.12 here). That is leakage, not
skill. Purged, blocked folds score below zero (about -0.12): a model fitted
to noise does worse than the test fold's own mean, which is the honest
answer.
"""),
]


def _cell(kind: str, source: str) -> dict[str, object]:
    lines = source.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def render(name: str, variant: str) -> str:
    cells = []
    for kind, text, stub in NOTEBOOKS[name]:
        if kind == "md":
            cells.append(_cell("md", text))
        elif kind == "task" and variant == "exercise":
            cells.append(_cell("code", stub + "\n" + STUB_FOOTER))
        else:
            cells.append(_cell("code", text))
    for i, c in enumerate(cells):
        c["id"] = f"r{name[:2]}-{variant[0]}-{i:02d}"
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "qp": {"variant": variant, "source": "research/notebooks/build_notebooks.py"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def outputs() -> dict[Path, str]:
    return {HERE / f"{name}-{variant}.ipynb": render(name, variant)
            for name in NOTEBOOKS for variant in ("exercise", "solution")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build or check the research notebooks.")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text("utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, "utf-8", newline="\n")
    if stale:
        print("stale notebooks (run build_notebooks.py):", ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
