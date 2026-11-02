"""Training-period studies on ECB reference rates (1999-01-04 to 2021-12-31).

Each study takes its parameters from the pre-registration it names, refuses
any data on or after 2022-01-01 (``models.training_only``), and writes one
JSON result carrying the registration hash so ``research/tools/prereg.py
check`` can verify it.

Run: ``python research/studies/fx_studies.py [pairs|regime|all]``. The ECB
file is downloaded once into ``data/raw/`` by ``data/fetch/qp_fetch.py``.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "research" / "tools"), str(REPO / "data" / "fetch")]

import prereg  # noqa: E402
from models import johansen_trace, kalman_hedge, training_only  # noqa: E402

from fx_pairs.cointegration import engle_granger  # noqa: E402
from fx_regime.hmm import filtered_probabilities, fit_hmm  # noqa: E402
from qp_core.stats import block_bootstrap_ci, lo_sharpe_se, sharpe_ratio  # noqa: E402

PREREG = REPO / "research" / "prereg"
RESULTS = REPO / "research" / "results"
TRAIN_START, TRAIN_END = "1999-01-04", "2021-12-31"

# Parameters copied from the pre-registrations. A test checks they match.
PAIRS = dict(formation_years=3, embargo_days=5, first_test_year=2003, eg_level="5%",
             delta=1e-6, entry_z=2.0, exit_z=0.5, cost_bp=1.0, lags=1)
REGIME = dict(first_test_year=2003, n_states=2, seed=0, n_init=5, threshold=0.5)


def usd_prices(eur_rates: pd.DataFrame) -> pd.DataFrame:
    """USD value of one unit of each non-USD currency, from EUR-based rates."""
    usd = eur_rates["USD"]
    out = pd.DataFrame({c: usd / eur_rates[c] for c in eur_rates.columns if c != "USD"})
    out["EUR"] = usd
    return out.dropna()


def _write(name: str, reg_file: str, period: tuple[str, str], payload: dict) -> Path:
    reg = prereg.load(PREREG / reg_file)
    RESULTS.mkdir(parents=True, exist_ok=True)
    doc = {"prereg": f"research/prereg/{reg_file}", "prereg_sha256": reg.sha256,
           "registration_signed": reg.signed, "period": {"start": period[0], "end": period[1]},
           "dataset": "ECB euro foreign exchange reference rates (EXR), daily", **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def _sharpe_block(r: np.ndarray) -> dict:
    active = int(np.count_nonzero(r))
    out = {"n_days": int(r.size), "active_days": active}
    if active < 3:
        return {**out, "sharpe": None, "sharpe_se_lo": None, "ci95": None, "note": "no trading"}
    try:
        point, lo, hi = block_bootstrap_ci(r, lambda a: sharpe_ratio(a), n_boot=2000, seed=0)
        ci = [lo, hi]
    except ValueError:  # mostly-flat series: too many zero-variance resamples
        point, ci = sharpe_ratio(r), None
    return {**out, "sharpe": point, "sharpe_se_lo": lo_sharpe_se(r), "ci95": ci}


def pair_returns(ly: pd.Series, lx: pd.Series, test_mask: np.ndarray, p: dict) -> np.ndarray:
    """Net daily return of one pair over the test rows (position decided at
    close t, earns the return of t+1; cost on gross notional traded)."""
    form = ~test_mask
    a = np.column_stack([np.ones(int(form.sum())), lx.to_numpy()[form]])
    coef, *_ = np.linalg.lstsq(a, ly.to_numpy()[form], rcond=None)
    obs_var = float(np.var(ly.to_numpy()[form] - a @ coef))
    kf = kalman_hedge(ly.to_numpy(), lx.to_numpy(), delta=p["delta"], obs_var=obs_var)
    z = kf.error / np.sqrt(kf.error_var)
    beta = kf.beta
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
    scale = 1.0 + np.abs(beta)
    wy, wx = pos / scale, -pos * beta / scale
    ry, rx = np.diff(ly.to_numpy(), prepend=np.nan), np.diff(lx.to_numpy(), prepend=np.nan)
    gross = np.zeros(n)
    gross[1:] = wy[:-1] * ry[1:] + wx[:-1] * rx[1:]
    cost = p["cost_bp"] * 1e-4 * (np.abs(np.diff(wy, prepend=0.0)) + np.abs(np.diff(wx, prepend=0.0)))
    return (gross - cost)[test_mask]


def pairs_study(px: pd.DataFrame, p: dict = PAIRS) -> dict:
    px = training_only(px)
    logp = np.log(px)
    years = range(p["first_test_year"], logp.index.max().year + 1)
    names = list(px.columns)
    folds, daily = [], []
    for year in years:
        test = logp.index.year == year
        form_end = np.flatnonzero(test)[0] - p["embargo_days"]
        form = (logp.index.year >= year - p["formation_years"]) & (np.arange(len(logp)) < form_end)
        selected = []
        for a, b in itertools.combinations(names, 2):
            fy, fx = logp[a][form], logp[b][form]
            eg = engle_granger(fy.to_numpy(), fx.to_numpy())
            jo = johansen_trace(np.column_stack([fy, fx]), k_ar_diff=p["lags"])
            if eg.cointegrated and jo.rank >= 1:
                selected.append((a, b))
        window = form | test
        rets = [pair_returns(logp[a][window], logp[b][window], test[window], p) for a, b in selected]
        port = np.mean(rets, axis=0) if rets else np.zeros(int(test.sum()))
        daily.append(port)
        folds.append({"year": year, "pairs": [f"{a}/{b}" for a, b in selected],
                      "net_return": float(port.sum())})
    r = np.concatenate(daily)
    return {"folds": folds, "candidate_pairs": math.comb(len(names), 2), "trial_count": 3,
            "primary_metric": "annualised net Sharpe, walk-forward test years", **_sharpe_block(r)}


def usd_basket_returns(px: pd.DataFrame) -> pd.Series:
    """Equal-weight daily log return of holding the G10 currencies against USD."""
    return np.log(px).diff().dropna().mean(axis=1)


def regime_study(px: pd.DataFrame, p: dict = REGIME) -> dict:
    r = usd_basket_returns(training_only(px))
    years = range(p["first_test_year"], r.index.max().year + 1)
    next_ret, in_high = [], []
    for year in years:
        fit = r[r.index.year < year].to_numpy()[:, None]
        model = fit_hmm(fit, p["n_states"], seed=p["seed"], n_init=p["n_init"])
        high = int(np.argmax(model.covs[:, 0, 0]))
        upto = r[r.index.year <= year]
        prob = filtered_probabilities(model, upto.to_numpy()[:, None])[:, high]
        mask = (upto.index.year == year)[:-1]  # state at close t, return of t+1
        flag = prob[:-1][mask] > p["threshold"]
        nxt = upto.to_numpy()[1:][mask]
        next_ret.append(nxt)
        in_high.append(flag)
    nr, hi = np.concatenate(next_ret), np.concatenate(in_high)
    h = nr[hi]
    lag = int(4 * (h.size / 100) ** (2 / 9))
    d = h - h.mean()
    lrv = d @ d / h.size + sum(2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / h.size for k in range(1, lag + 1))
    t_stat = h.mean() / math.sqrt(lrv / h.size)
    return {"primary_metric": "mean next-day basket return in the predicted high-vol state",
            "trial_count": 1, "high_vol_days": int(h.size), "all_days": int(nr.size),
            "mean_ann_high": float(h.mean() * 252), "mean_ann_low": float(nr[~hi].mean() * 252),
            "hac_t_high": float(t_stat), "hac_lag": lag,
            "negative_mean_supported": bool(t_stat < -1.645)}


def main(which: str) -> None:
    from qp_fetch import fetch_ecb_fx, parse_ecb_csv, RAW_DIR

    cached = RAW_DIR / f"ecb_exr_1999-01-01_{TRAIN_END}.csv"
    eur = parse_ecb_csv(cached.read_text()) if cached.exists() else fetch_ecb_fx(end=TRAIN_END)
    px = usd_prices(eur)
    period = (str(px.index.min().date()), str(px.index.max().date()))
    if which in ("pairs", "all"):
        print(_write("pairs_fx_g10_train", "pairs-fx-g10.md", period, pairs_study(px)))
    if which in ("regime", "all"):
        print(_write("regime_fx_basket_train", "regime-fx-basket.md", period, regime_study(px)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
