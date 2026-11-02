"""Training-period runs for the two forward-hold-out registrations.

- ``pairs``: ``prereg/pairs-fx-g10-v2.md``. The signed pairs design with the
  Kalman observation variance chosen on training data (1999-2021) from a
  fixed menu of four rules, every rule counted as a trial.
- ``momentum``: ``prereg/momentum-fx-g10.md``. Cross-sectional spot momentum
  on the nine G10 currencies against USD, one variant fixed from the
  literature before any data was looked at.

Both registrations hold out FORWARD data: ECB fixes from the signing date
(2026-11-03) onward, read from the forward log (``research/forward/SPEC.md``).
This module only ever scores 1999-01-04 to 2021-12-31; every study calls
``models.training_only`` and refuses anything later. Nothing here reads or
scores 2022-2025, the single-use hold-out of the two signed registrations.

Run: ``python research/studies/forward_studies.py [pairs|momentum|all]``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "research" / "tools"), str(REPO / "data" / "fetch")]

import holdout as ho  # noqa: E402
import prereg  # noqa: E402
from fx_pairs.cointegration import engle_granger  # noqa: E402
from fx_studies import PAIRS, _sharpe_block, pairs_fold_masks, usd_prices  # noqa: E402
from models import johansen_trace, kalman_hedge, training_only  # noqa: E402

PREREG = REPO / "research" / "prereg"
RESULTS = REPO / "research" / "results"
TRAIN_END = "2021-12-31"

# The variance-rule menu, fixed before the first run (see the registration).
# Each maps the formation-window OLS residual u to the Kalman observation variance.
VAR_RULES = {
    "levels": lambda u: float(np.var(u)),             # the signed v1 rule
    "levels_x0.25": lambda u: 0.25 * float(np.var(u)),
    "levels_x0.1": lambda u: 0.1 * float(np.var(u)),
    "diff": lambda u: float(np.var(np.diff(u))),      # variance of the daily residual change
}
# Trial count: the 4 rules above plus the 2 earlier v1 drafting variants that
# were not the "levels" rule (pairs-fx-g10.md, primary metric section).
PAIRS_V2_TRIALS = 6
# Activity floor: a rule must trade in at least this share of training test
# years to be eligible. Added after the first run picked the dormant v1 rule
# (see the registration, "How the rule was chosen"); disclosed, not hidden.
MIN_ACTIVE_YEAR_FRAC = 0.75
PAIRS_V2 = dict(PAIRS)

MOMENTUM = dict(lookback_days=63, n_long=3, n_short=3, cost_bp=1.0, first_year=2000, trial_count=1)


# ---- pairs v2 ------------------------------------------------------------

def pair_returns_rule(ly: np.ndarray, lx: np.ndarray, test_mask: np.ndarray, rule: str,
                      p: dict = PAIRS_V2) -> tuple[np.ndarray, int]:
    """Net daily return of one pair over the test rows, with the Kalman
    observation variance set by ``VAR_RULES[rule]`` on the formation rows.

    Same trading rule as the signed design (``fx_studies.pair_returns``):
    z = one-step error / its forecast SD; enter at |z| > entry_z against z,
    exit at |z| < exit_z, flat outside the test rows; position at close t
    earns the return of t+1; cost on gross notional traded. Returns the
    returns and the number of entries."""
    form = ~test_mask
    a = np.column_stack([np.ones(int(form.sum())), lx[form]])
    coef, *_ = np.linalg.lstsq(a, ly[form], rcond=None)
    obs_var = VAR_RULES[rule](ly[form] - a @ coef)
    if not obs_var > 0:
        raise ValueError("observation variance must be positive")
    kf = kalman_hedge(ly, lx, delta=p["delta"], obs_var=obs_var)
    z = kf.error / np.sqrt(kf.error_var)
    n = len(z)
    pos = np.zeros(n)
    for t in range(1, n):
        prev = pos[t - 1]
        if not test_mask[t]:
            pos[t] = 0.0
        elif prev == 0.0 and abs(z[t]) > p["entry_z"]:
            pos[t] = -np.sign(z[t])
        elif prev != 0.0 and abs(z[t]) < p["exit_z"]:
            pos[t] = 0.0
        else:
            pos[t] = prev
    scale = 1.0 + np.abs(kf.beta)
    wy, wx = pos / scale, -pos * kf.beta / scale
    gross = np.zeros(n)
    gross[1:] = wy[:-1] * np.diff(ly) + wx[:-1] * np.diff(lx)
    cost = p["cost_bp"] * 1e-4 * (np.abs(np.diff(wy, prepend=0.0)) + np.abs(np.diff(wx, prepend=0.0)))
    held = pos[test_mask] != 0.0
    entries = int((held & ~np.concatenate([[False], held[:-1]])).sum())
    return (gross - cost)[test_mask], entries


def select_pairs(logp: pd.DataFrame, form: np.ndarray, p: dict = PAIRS_V2) -> list[tuple[str, str]]:
    """Engle-Granger at 5 % AND Johansen trace rank >= 1 on the formation rows."""
    out = []
    for a, b in itertools.combinations(list(logp.columns), 2):
        fy, fx = logp[a].to_numpy()[form], logp[b].to_numpy()[form]
        if engle_granger(fy, fx).cointegrated and johansen_trace(np.column_stack([fy, fx]), p["lags"]).rank >= 1:
            out.append((a, b))
    return out


def choose_rule(variants: dict, n_years: int, min_active_frac: float) -> str:
    """Highest training net Sharpe among rules that trade in at least
    ``min_active_frac`` of the test years; ties go to menu order."""
    ok = [k for k in VAR_RULES if variants[k]["sharpe"] is not None
          and variants[k]["years_with_trades"] >= min_active_frac * n_years]
    if not ok:
        raise ValueError("no variance rule meets the activity floor")
    return max(ok, key=lambda k: variants[k]["sharpe"])


def pairs_v2_study(px: pd.DataFrame, p: dict = PAIRS_V2) -> dict:
    """Walk-forward training run of every rule in ``VAR_RULES``; the rule
    with the highest training net Sharpe is the one registered."""
    px = training_only(px)
    logp = pd.DataFrame(np.log(px))
    idx = pd.DatetimeIndex(logp.index)
    years = range(p["first_test_year"], idx.max().year + 1)
    daily: dict[str, list[np.ndarray]] = {r: [] for r in VAR_RULES}
    entries = {r: 0 for r in VAR_RULES}
    active_years = {r: 0 for r in VAR_RULES}
    for year in years:
        form, test = pairs_fold_masks(idx, year, p)
        selected = select_pairs(logp, form, p)
        window = form | test
        for rule in VAR_RULES:
            rets = []
            for a, b in selected:
                r, e = pair_returns_rule(logp[a].to_numpy()[window], logp[b].to_numpy()[window],
                                         test[window], rule, p)
                rets.append(r)
                entries[rule] += e
            port = np.mean(rets, axis=0) if rets else np.zeros(int(test.sum()))
            active_years[rule] += int(np.any(port != 0.0))
            daily[rule].append(port)
    variants = {}
    for rule in VAR_RULES:
        r = np.concatenate(daily[rule])
        variants[rule] = {**_sharpe_block(r), "entries": entries[rule], "years_with_trades": active_years[rule],
                          "net_return_total": float(r.sum())}
    n_years = len(years)
    chosen_unconstrained = choose_rule(variants, n_years, 0.0)
    chosen = choose_rule(variants, n_years, MIN_ACTIVE_YEAR_FRAC)
    r_best = np.concatenate(daily[chosen])
    return {"primary_metric": "annualised net Sharpe, walk-forward test years (training)",
            "trial_count": PAIRS_V2_TRIALS, "variants": variants, "test_years": n_years,
            "min_active_year_frac": MIN_ACTIVE_YEAR_FRAC, "chosen_rule": chosen,
            "chosen_rule_without_activity_floor": chosen_unconstrained,
            "deflated_sharpe_chosen": ho.deflated_sharpe_of(r_best, n_trials=PAIRS_V2_TRIALS)}


# ---- cross-sectional momentum --------------------------------------------

def momentum_weights(px: pd.DataFrame, p: dict = MOMENTUM) -> pd.DataFrame:
    """Daily target weights. On the last business day of each month, rank the
    currencies by their log spot return over the previous ``lookback_days``
    rows; long the top ``n_long`` and short the bottom ``n_short`` at
    1 / (n_long + n_short) each (unit gross). Weights hold until the next
    month end. Rows before the first full lookback are zero. Columns come
    back in alphabetical order.

    Design after Menkhoff, Sarno, Schmeling and Schrimpf (2012), "Currency
    momentum strategies", J. Financial Economics 106(3), 660-684, section 2
    (rank on past returns, long winners, short losers, monthly rebalance),
    here on spot returns only, without the forward premium."""
    px = px[sorted(px.columns)]  # ties: earlier currency code ranks lower
    logp = np.log(px)
    idx = pd.DatetimeIndex(px.index)
    month_end = np.r_[idx.month[1:] != idx.month[:-1], True]
    w = np.zeros(px.shape)
    cur = np.zeros(px.shape[1])
    k = p["lookback_days"]
    gross = 1.0 / (p["n_long"] + p["n_short"])
    for t in range(len(idx)):
        if month_end[t] and t >= k:
            past = logp.iloc[t].to_numpy() - logp.iloc[t - k].to_numpy()
            order = np.argsort(past, kind="stable")  # ties: column order
            cur = np.zeros(px.shape[1])
            cur[order[-p["n_long"]:]] = gross
            cur[order[: p["n_short"]]] = -gross
        w[t] = cur
    return pd.DataFrame(w, index=px.index, columns=px.columns)


def momentum_returns(px: pd.DataFrame, p: dict = MOMENTUM) -> pd.Series:
    """Net daily return: weights at close t earn the log return of t+1;
    cost ``cost_bp`` per unit of gross weight traded."""
    w = momentum_weights(px, p)
    r = np.log(px[w.columns]).diff()
    gross = (w.shift(1) * r).sum(axis=1)
    traded = w.diff().abs().sum(axis=1)
    traded.iloc[0] = w.iloc[0].abs().sum()
    return (gross - p["cost_bp"] * 1e-4 * traded).iloc[1:]


def momentum_study(px: pd.DataFrame, p: dict = MOMENTUM) -> dict:
    px = training_only(px)
    r = momentum_returns(px, p)
    r = r[pd.DatetimeIndex(r.index).year >= p["first_year"]]
    yearly = r.groupby(pd.DatetimeIndex(r.index).year).sum()
    arr = r.to_numpy()
    return {"primary_metric": "annualised net Sharpe of the daily return (training)",
            "trial_count": p["trial_count"], **_sharpe_block(arr),
            "deflated_sharpe": ho.deflated_sharpe_of(arr, n_trials=p["trial_count"]),
            "per_year_net_return": {int(y): float(v) for y, v in yearly.items()}}


# ---- driver --------------------------------------------------------------

def _write(name: str, reg_file: str, period: tuple[str, str], payload: dict) -> Path:
    reg = prereg.load(PREREG / reg_file)
    RESULTS.mkdir(parents=True, exist_ok=True)
    doc = {"prereg": f"research/prereg/{reg_file}", "prereg_sha256": reg.sha256,
           "registration_signed": reg.signed, "mode": "train",
           "period": {"start": period[0], "end": period[1]},
           "dataset": "ECB euro foreign exchange reference rates (EXR), daily", **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["pairs", "momentum", "all"])
    a = ap.parse_args(argv)
    from qp_fetch import RAW_DIR, fetch_ecb_fx, parse_ecb_csv

    cached = RAW_DIR / f"ecb_exr_1999-01-01_{TRAIN_END}.csv"
    eur = parse_ecb_csv(cached.read_text()) if cached.exists() else fetch_ecb_fx(end=TRAIN_END)
    px = training_only(usd_prices(eur))
    period = (str(px.index.min().date()), str(px.index.max().date()))
    if a.which in ("pairs", "all"):
        print(_write("pairs_fx_g10_v2_train", "pairs-fx-g10-v2.md", period, pairs_v2_study(px)))
    if a.which in ("momentum", "all"):
        print(_write("momentum_fx_g10_train", "momentum-fx-g10.md", period, momentum_study(px)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
