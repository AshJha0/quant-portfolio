"""Robustness checks for the fix-based FX studies, training period only.

Two questions the training results do not answer on their own.

1. Is the pair selection finding anything? Each fold tests 36 G10 pairs
   (Engle-Granger at 5 % AND Johansen trace rank >= 1) on three years of
   data. Some pairs pass by chance. ``selection_calibration`` puts next to
   each fold's selected count the count a null permutation gives, and the
   count that survives a Benjamini-Hochberg false discovery rate cut.
2. Does the result survive the fix not being tradeable? The studies trade at
   the ECB 14:15 CET fix at 1 bp. ``cost_event_stress`` re-scores the pairs
   v2 and momentum rules at 1, 3 and 5 bp, and with event days (a jump
   larger than ``EVENT_SIGMA`` trailing standard deviations in any currency,
   or a calendar gap of more than ``GAP_DAYS`` days between fixes) either
   dropped or scored adversely (gains on those days forfeited, losses kept).

Nothing here is a new registration or a new decision rule. The outputs are
sensitivity tables written to ``research/results`` with ``kind`` stress or
calibration, and every run is appended to ``results/trials.jsonl``.

Run: ``python research/studies/robustness.py [calibration|stress|all] [--perms N]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "research" / "tools"), str(REPO / "data" / "fetch")]

import forward_studies as fs  # noqa: E402
import trials  # noqa: E402
from fx_pairs.cointegration import engle_granger  # noqa: E402
from fx_studies import pairs_fold_masks, usd_prices  # noqa: E402
from models import johansen_trace, training_only  # noqa: E402
from qp_core.stats import lo_sharpe_se, sharpe_ratio  # noqa: E402

COSTS_BP = (1.0, 3.0, 5.0)
EVENT_SIGMA = 5.0      # |daily log return| above this many trailing sds is an event
EVENT_WINDOW = 250     # trailing rows for that sd (ex-ante: rows before the day)
EVENT_MIN_ROWS = 60    # no event flag until this many trailing rows exist
GAP_DAYS = 4           # calendar days between consecutive fixes above this is a gap
FDR_Q = 0.05
N_PERMS = 200


# ---- selection calibration -------------------------------------------------

def bh_reject(pvalues: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg (1995) step-up: reject the k smallest p-values,
    k the largest i with p_(i) <= i q / m. NaN p-values are never rejected."""
    p = np.asarray(pvalues, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    out = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[ok], kind="stable")]
    below = p[order] <= q * np.arange(1, m + 1) / m
    if below.any():
        out[order[: np.flatnonzero(below)[-1] + 1]] = True
    return out


def eg_pvalue(stat: float) -> float:
    """MacKinnon (1994, 2010) approximate p-value of an Engle-Granger tau for
    two series with a constant (statsmodels ``mackinnonp``, N = 2)."""
    from statsmodels.tsa.adfvalues import mackinnonp

    return float(mackinnonp(stat, regression="c", N=2)) if np.isfinite(stat) else float("nan")


def pair_tests(logp: pd.DataFrame, form: np.ndarray, lags: int) -> list[dict]:
    """Per candidate pair on the formation rows: EG decision and p-value,
    Johansen rank. The selection rule is ``eg and johansen_rank >= 1``."""
    names = list(logp.columns)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fy, fx = logp[a].to_numpy()[form], logp[b].to_numpy()[form]
            eg = engle_granger(fy, fx)
            jo = johansen_trace(np.column_stack([fy, fx]), lags)
            out.append({"pair": f"{a}/{b}", "eg": bool(eg.cointegrated), "eg_stat": float(eg.stat),
                        "eg_pvalue": eg_pvalue(eg.stat) if not eg.degenerate else float("nan"),
                        "johansen_rank": int(jo.rank)})
    return out


def null_levels(logp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Null data: draw rows of the joint daily log-return matrix with
    replacement (an i.i.d. bootstrap of days) and re-cumulate from the first
    row. Each day's cross-section (the correlation between currencies, fat
    tails, the dollar factor) is kept; the order in time is destroyed, so
    every spread becomes a random walk and no pair is cointegrated.

    Drawing with replacement rather than permuting matters: a permutation
    keeps each series' end point, so the spread of a truly cointegrated pair
    is rebuilt as a bridge that starts and ends near zero and still looks
    stationary to the test, which makes the null too hard to beat."""
    d = np.diff(logp, axis=0)
    d = d[rng.integers(0, len(d), len(d))]
    return np.vstack([logp[:1], logp[:1] + np.cumsum(d, axis=0)])


def selection_count(logp: np.ndarray, lags: int) -> int:
    k = logp.shape[1]
    n = 0
    for i in range(k):
        for j in range(i + 1, k):
            if engle_granger(logp[:, i], logp[:, j]).cointegrated and \
                    johansen_trace(logp[:, [i, j]], lags).rank >= 1:
                n += 1
    return n


def selection_calibration(px: pd.DataFrame, p: dict = fs.PAIRS_V2, n_perms: int = N_PERMS,
                          seed: int = 0, q: float = FDR_Q, years: tuple[int, ...] | None = None) -> dict:
    """Per training fold: pairs selected, the null distribution of
    that count, and the pairs that also survive BH at ``q`` on the EG p-value."""
    px = training_only(px)
    logp = pd.DataFrame(np.log(px))
    idx = pd.DatetimeIndex(logp.index)
    years = years or tuple(range(p["first_test_year"], idx.max().year + 1))
    rng = np.random.default_rng(seed)
    folds, tot_actual, tot_null = [], 0, np.zeros(n_perms)
    for year in years:
        form, _ = pairs_fold_masks(idx, year, p)
        tests = pair_tests(logp, form, p["lags"])
        selected = [t for t in tests if t["eg"] and t["johansen_rank"] >= 1]
        bh = bh_reject(np.array([t["eg_pvalue"] for t in tests]), q)
        bh_sel = [t["pair"] for t, r in zip(tests, bh, strict=True) if r and t["johansen_rank"] >= 1]
        f = logp.to_numpy()[form]
        null = np.array([selection_count(null_levels(f, rng), p["lags"]) for _ in range(n_perms)])
        tot_actual += len(selected)
        tot_null += null
        folds.append({"year": year, "candidates": len(tests), "selected": len(selected),
                      "selected_pairs": [t["pair"] for t in selected],
                      "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                      "perm_pvalue": float((1 + (null >= len(selected)).sum()) / (1 + n_perms)),
                      "bh_selected": len(bh_sel), "bh_selected_pairs": bh_sel})
    return {"method": ("null: rows of the formation-window daily log-return matrix drawn with replacement, "
                       "levels re-cumulated, the same EG AND Johansen selection re-run; "
                       f"FDR: Benjamini-Hochberg at q = {q} on MacKinnon EG p-values, then Johansen"),
            "n_perms": n_perms, "seed": seed, "fdr_q": q, "folds": folds,
            "total_selected": tot_actual, "total_null_mean": float(tot_null.mean()),
            "total_null_p95": float(np.percentile(tot_null, 95)),
            "total_perm_pvalue": float((1 + (tot_null >= tot_actual).sum()) / (1 + n_perms)),
            "total_bh_selected": int(sum(f["bh_selected"] for f in folds))}


# ---- cost and event-gap stress ---------------------------------------------

def event_days(px: pd.DataFrame, sigma: float = EVENT_SIGMA, window: int = EVENT_WINDOW,
               min_rows: int = EVENT_MIN_ROWS, gap_days: int = GAP_DAYS) -> pd.Series:
    """Boolean per date: True if any currency's log return that day exceeds
    ``sigma`` times its standard deviation over the ``window`` rows before
    it, or the fix follows the previous one by more than ``gap_days``
    calendar days. Uses only data up to the day itself."""
    r = np.log(px).diff()
    sd = r.shift(1).rolling(window, min_periods=min_rows).std()
    jump = (r.abs() > sigma * sd).any(axis=1)
    gaps = pd.Series(pd.DatetimeIndex(px.index), index=px.index).diff().dt.days > gap_days
    return (jump | gaps).rename("event")


def _stats(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    if np.count_nonzero(r) < 3 or not np.std(r) > 0:
        return {"n_days": int(r.size), "sharpe": None, "sharpe_se_lo": None, "net_return_ann": None}
    return {"n_days": int(r.size), "sharpe": float(sharpe_ratio(r)), "sharpe_se_lo": float(lo_sharpe_se(r)),
            "net_return_ann": float(r.mean() * 252)}


def stress_table(r: pd.Series, events: pd.Series) -> dict:
    """``base``, ``exclude`` (event days dropped) and ``adverse`` (on event
    days the return is min(r, 0)) statistics of the daily returns ``r``."""
    ev = events.reindex(r.index, fill_value=False).to_numpy()
    a = r.to_numpy()
    return {"base": _stats(a), "exclude": _stats(a[~ev]), "adverse": _stats(np.where(ev, np.minimum(a, 0.0), a)),
            "event_days": int(ev.sum()), "event_days_exposed": int((ev & (a != 0.0)).sum()),
            "event_pnl": float(a[ev].sum())}


def pairs_v2_daily(px: pd.DataFrame, p: dict = fs.PAIRS_V2) -> pd.Series:
    """Dated daily net returns of the registered level-z pairs rule (training)."""
    px = training_only(px)
    logp = pd.DataFrame(np.log(px))
    idx = pd.DatetimeIndex(logp.index)
    parts = []
    for year in range(p["first_test_year"], idx.max().year + 1):
        form, test = pairs_fold_masks(idx, year, p)
        window = form | test
        rets = [fs.pair_returns_level(logp[a].to_numpy()[window], logp[b].to_numpy()[window], test[window], p)[0]
                for a, b in fs.select_pairs(logp, form, p)]
        port = np.mean(rets, axis=0) if rets else np.zeros(int(test.sum()))
        parts.append(pd.Series(port, index=idx[test]))
    return pd.concat(parts)


def cost_event_stress(px: pd.DataFrame, costs: tuple[float, ...] = COSTS_BP, p: dict = fs.PAIRS_V2) -> dict:
    """Stress tables for pairs v2 and momentum at each cost in ``costs`` bp."""
    px = training_only(px)
    events = event_days(px)
    out: dict = {"pairs_v2": {}, "momentum": {}}
    for c in costs:
        out["pairs_v2"][f"{c:g}bp"] = stress_table(pairs_v2_daily(px, {**p, "cost_bp": c}), events)
        m = fs.momentum_returns(px, {**fs.MOMENTUM, "cost_bp": c})
        m = m[pd.DatetimeIndex(m.index).year >= fs.MOMENTUM["first_year"]]
        out["momentum"][f"{c:g}bp"] = stress_table(m, events)
    ev = events[events]
    out["events"] = {"sigma": EVENT_SIGMA, "window": EVENT_WINDOW, "min_rows": EVENT_MIN_ROWS,
                     "gap_days": GAP_DAYS, "n": int(ev.size),
                     "dates": [str(d.date()) for d in pd.DatetimeIndex(ev.index)]}
    return out


def _headline(tables: dict) -> dict:
    """Net Sharpe per cost and variant, for the trials log."""
    return {f"sharpe_{c}_{v}": t[v]["sharpe"] for c, t in tables.items() for v in ("base", "exclude", "adverse")}


# ---- driver ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["calibration", "stress", "all"])
    ap.add_argument("--perms", type=int, default=N_PERMS)
    a = ap.parse_args(argv)
    git = trials.git_state(REPO)  # before the run writes anything
    from qp_fetch import RAW_DIR, fetch_ecb_fx, parse_ecb_csv

    cached = RAW_DIR / f"ecb_exr_1999-01-01_{fs.TRAIN_END}.csv"
    eur = parse_ecb_csv(cached.read_text()) if cached.exists() else fetch_ecb_fx(end=fs.TRAIN_END)
    px = training_only(usd_prices(eur))
    period = (str(px.index.min().date()), str(px.index.max().date()))
    if a.which in ("calibration", "all"):
        res = selection_calibration(px, n_perms=a.perms)
        print(fs._write("pairs_selection_calibration", "pairs-fx-g10-v2.md", period, res,
                        params={**fs.PAIRS_V2, "n_perms": a.perms, "fdr_q": FDR_Q}, kind="calibration",
                        metric={k: res[k] for k in ("total_selected", "total_null_mean", "total_perm_pvalue",
                                                    "total_bh_selected")}, git=git))
    if a.which in ("stress", "all"):
        res = cost_event_stress(px)
        params = {"costs_bp": list(COSTS_BP), "event_sigma": EVENT_SIGMA, "event_window": EVENT_WINDOW,
                  "gap_days": GAP_DAYS}
        print(fs._write("pairs_fx_g10_v2_stress", "pairs-fx-g10-v2.md", period,
                        {"stress": res["pairs_v2"], "events": res["events"]},
                        params={**fs.PAIRS_V2, **params}, kind="stress", metric=_headline(res["pairs_v2"]), git=git))
        print(fs._write("momentum_fx_g10_stress", "momentum-fx-g10.md", period,
                        {"stress": res["momentum"], "events": res["events"]},
                        params={**fs.MOMENTUM, **params}, kind="stress", metric=_headline(res["momentum"]), git=git))
    return 0


if __name__ == "__main__":
    sys.exit(main())
