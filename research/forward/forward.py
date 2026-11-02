"""Live forward tracking of the two registered FX studies on ECB rates.

Why this exists. The 2022-2025 hold-out of ``pairs-fx-g10.md`` and
``regime-fx-basket.md`` can be scored once. Every day after that is new,
unseen data, and the only way to keep it unseen is to record it as it
arrives, before anyone can look at it with a choice in mind. This script
appends one line per ECB business day to a JSON Lines log on the
``forward-log`` branch, with the rates, the hash of the bytes they came
from and the commit of the code that wrote them.

What it refuses. It never asks the ECB for a date before
``FORWARD_START`` (2026-11-03) and raises ``ForwardGuardError`` if a
response contains one. That keeps the single-use hold-out (2022-01-01 to
2025-12-31) and the training years out of the forward log entirely: the
strategies below see only data the log itself holds.

The registered rules, applied to forward history only.

- regime (``regime-fx-basket.md``): basket = mean of the nine daily log
  returns of the G10 currencies in USD. For a day in year Y the two-state
  HMM (``fx_regime.hmm.fit_hmm``, seed 0, 5 restarts) is fitted on every
  basket return before 1 January of Y; the high-vol state is the one with
  the larger emission variance; the day is flagged high-vol if its filtered
  probability is above 0.5, and the flag classifies the return of the next
  day. The registration's training run started its first test year four
  calendar years after the data began (1999 to 2003); the forward run keeps
  that gap, counted in whole calendar years of forward data.
- pairs (``pairs-fx-g10.md``): for test year Y, formation is the three
  calendar years before Y ending five business days before Y's first day;
  a pair is selected if Engle-Granger at 5 % and Johansen trace rank >= 1
  at 95 % both hold; the Kalman hedge (delta 1e-6, observation variance =
  OLS residual variance over the formation window) runs from the start of
  the formation window; enter at |z| > 2, exit at |z| < 0.5; legs scaled
  by 1 / (1 + |beta|); 1 bp per leg traded; equal weight over the selected
  pairs. The formation window must lie wholly inside forward data.

Until those windows exist (regime from 2031, pairs from 2030, since
forward data starts in November 2026) each line carries
``status: "warming_up"`` and the year it becomes scorable. The rates are
logged from day one, so the scores can be recomputed from the log alone.

Run: ``python research/forward/forward.py update --log PATH`` (fetches
from the ECB), or ``--csv FILE`` to read an ECB ``csvdata`` file instead
(the tests use synthetic files that way).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import subprocess
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "research" / "studies"), str(REPO / "research" / "tools"),
                str(REPO / "data" / "fetch")]

import prereg  # noqa: E402
import qp_fetch  # noqa: E402
from fx_pairs.cointegration import engle_granger  # noqa: E402
from fx_regime.hmm import filtered_probabilities, fit_hmm  # noqa: E402
from models import johansen_trace, kalman_hedge  # noqa: E402

FORWARD_START = dt.date(2026, 11, 3)
HOLDOUT = (dt.date(2022, 1, 1), dt.date(2025, 12, 31))
SCHEMA = 1
CURRENCIES = qp_fetch.ECB_G10  # quoted per EUR
# Column order decides which leg of a pair is y and which is x. The study read
# ECB files through parse_ecb_csv, whose pivot sorts currencies alphabetically,
# so the forward run uses that order too, whatever order the input arrives in.
ORDER = tuple(sorted(CURRENCIES))
PREREG_FILES = ("pairs-fx-g10.md", "regime-fx-basket.md")

# Copied from the registrations (fx_studies.PAIRS / REGIME); a test checks they match.
PAIRS = dict(formation_years=3, embargo_days=5, eg_level="5%", delta=1e-6,
             entry_z=2.0, exit_z=0.5, cost_bp=1.0, lags=1)
REGIME = dict(n_states=2, seed=0, n_init=5, threshold=0.5)
REGIME_HISTORY_YEARS = 4  # 1999 -> 2003 in the registered training run


class ForwardGuardError(RuntimeError):
    """A request or a response touched a date before FORWARD_START."""


# ---------------------------------------------------------------- guards

def check_request(start: dt.date, end: dt.date) -> None:
    """Refuse any ECB request that starts before FORWARD_START."""
    if start < FORWARD_START:
        raise ForwardGuardError(f"request starts {start}; forward data starts {FORWARD_START}")
    if end < start:
        raise ForwardGuardError(f"request ends {end} before it starts {start}")


def check_rows(eur: pd.DataFrame) -> pd.DataFrame:
    """Refuse a parsed frame holding any row before FORWARD_START."""
    if len(eur) and pd.DatetimeIndex(eur.index).min().date() < FORWARD_START:
        raise ForwardGuardError(
            f"data holds {pd.DatetimeIndex(eur.index).min().date()}; forward data starts {FORWARD_START}")
    return eur


# ---------------------------------------------------------------- data

def usd_prices(eur: pd.DataFrame) -> pd.DataFrame:
    """USD value of one unit of each currency (same as fx_studies.usd_prices)."""
    usd = eur["USD"]
    out = pd.DataFrame({c: usd / eur[c] for c in eur.columns if c != "USD"})
    out["EUR"] = usd
    return out.dropna()


def fetch(start: dt.date, end: dt.date) -> tuple[pd.DataFrame, str, str]:
    """ECB rates for [start, end], the SHA-256 of the response and its URL."""
    check_request(start, end)
    url = qp_fetch.ECB_URL.format(ccys="+".join(CURRENCIES), start=start.isoformat(), end=end.isoformat())
    payload = qp_fetch._download(url)
    return check_rows(qp_fetch.parse_ecb_csv(payload.decode("utf-8"))), hashlib.sha256(payload).hexdigest(), url


def read_csv(path: Path) -> tuple[pd.DataFrame, str, str]:
    payload = path.read_bytes()
    eur = check_rows(qp_fetch.parse_ecb_csv(payload.decode("utf-8")))
    return eur, hashlib.sha256(payload).hexdigest(), f"file:{path.name}"


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log_frame(records: list[dict]) -> pd.DataFrame:
    """EUR-quoted rates held in the log, as a date x currency frame."""
    if not records:
        return pd.DataFrame(columns=list(ORDER), dtype=float)
    df = pd.DataFrame({r["date"]: r["rates_per_eur"] for r in records}).T
    df.index = pd.to_datetime(df.index)
    return check_rows(df[list(ORDER)].sort_index().astype(float))


# ---------------------------------------------------------------- regime

def regime_first_year(first_day: pd.Timestamp) -> int:
    """First year whose fit window holds REGIME_HISTORY_YEARS whole calendar years."""
    full = first_day.year if (first_day.month, first_day.day) <= (1, 5) else first_day.year + 1
    return full + REGIME_HISTORY_YEARS


def regime_signal(px: pd.DataFrame, day: pd.Timestamp, p: dict = REGIME,
                  cache: dict | None = None) -> dict:
    """High-vol flag at the close of ``day`` (classifies the return of the next day)."""
    year = day.year
    first = regime_first_year(pd.DatetimeIndex(px.index).min())
    if year < first:
        return {"status": "warming_up", "scorable_from_year": first}
    r = pd.DataFrame(np.log(px)).diff().dropna().mean(axis=1)
    ryear = pd.DatetimeIndex(r.index).year
    cache = {} if cache is None else cache
    if year not in cache:  # the fit for year Y only sees days before 1 January of Y
        cache[year] = fit_hmm(r[ryear < year].to_numpy()[:, None], p["n_states"],
                              seed=p["seed"], n_init=p["n_init"])
    model = cache[year]
    high = int(np.argmax(model.covs[:, 0, 0]))
    upto = r[(ryear <= year) & (r.index <= day)]
    prob = float(filtered_probabilities(model, upto.to_numpy()[:, None])[-1, high])
    return {"status": "scored", "fit_through": str(r[ryear < year].index.max().date()),
            "p_high": prob, "high_vol": bool(prob > p["threshold"]),
            "basket_return": float(upto.iloc[-1])}


# ---------------------------------------------------------------- pairs

def pairs_first_year(first_day: pd.Timestamp, p: dict = PAIRS) -> int:
    full = first_day.year if (first_day.month, first_day.day) <= (1, 5) else first_day.year + 1
    return full + p["formation_years"]


def pair_path(ly: np.ndarray, lx: np.ndarray, form: np.ndarray, test: np.ndarray,
              p: dict = PAIRS) -> dict[str, np.ndarray]:
    """Positions, leg weights and net returns of one pair over formation + test rows.

    Mirrors ``fx_studies.pair_returns`` line for line (a test checks the
    test-row net returns agree) but also returns the state the log records."""
    a = np.column_stack([np.ones(int(form.sum())), lx[form]])
    coef, *_ = np.linalg.lstsq(a, ly[form], rcond=None)
    obs_var = float(np.var(ly[form] - a @ coef))
    kf = kalman_hedge(ly, lx, delta=p["delta"], obs_var=obs_var)
    z = kf.error / np.sqrt(kf.error_var)
    beta = kf.beta
    n = len(z)
    pos = np.zeros(n)
    for t in range(1, n):
        prev = pos[t - 1]
        if not test[t]:
            pos[t] = 0.0
        elif prev == 0.0 and abs(z[t]) > p["entry_z"]:
            pos[t] = -np.sign(z[t])
        elif prev != 0.0 and abs(z[t]) < p["exit_z"]:
            pos[t] = 0.0
        else:
            pos[t] = prev
    scale = 1.0 + np.abs(beta)
    wy, wx = pos / scale, -pos * beta / scale
    ry, rx = np.diff(ly, prepend=np.nan), np.diff(lx, prepend=np.nan)
    gross = np.zeros(n)
    gross[1:] = wy[:-1] * ry[1:] + wx[:-1] * rx[1:]
    cost = p["cost_bp"] * 1e-4 * (np.abs(np.diff(wy, prepend=0.0)) + np.abs(np.diff(wx, prepend=0.0)))
    return {"z": z, "beta": beta, "pos": pos, "wy": wy, "wx": wx, "net": gross - cost}


def fold_masks(index: pd.DatetimeIndex, year: int, p: dict = PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """Same rule as fx_studies.pairs_fold_masks."""
    test = np.asarray(index.year == year)
    form_end = np.flatnonzero(test)[0] - p["embargo_days"]
    form = np.asarray(index.year >= year - p["formation_years"]) & (np.arange(len(index)) < form_end)
    return form, test


def select_pairs(logp: pd.DataFrame, form: np.ndarray, p: dict = PAIRS) -> list[tuple[str, str]]:
    out = []
    for a, b in itertools.combinations(list(logp.columns), 2):
        fy, fx = logp[a].to_numpy()[form], logp[b].to_numpy()[form]
        if engle_granger(fy, fx).cointegrated and johansen_trace(np.column_stack([fy, fx]),
                                                                 k_ar_diff=p["lags"]).rank >= 1:
            out.append((a, b))
    return out


def pairs_signal(px: pd.DataFrame, day: pd.Timestamp, p: dict = PAIRS,
                 cache: dict | None = None) -> dict:
    """Pairs held at the close of ``day`` and the portfolio net return earned on ``day``."""
    year = day.year
    first = pairs_first_year(pd.DatetimeIndex(px.index).min(), p)
    if year < first:
        return {"status": "warming_up", "scorable_from_year": first}
    px = px[px.index <= day]
    logp = pd.DataFrame(np.log(px))
    idx = pd.DatetimeIndex(logp.index)
    form, test = fold_masks(idx, year, p)
    cache = {} if cache is None else cache
    if year not in cache:  # selection only sees the formation window
        cache[year] = select_pairs(logp, form, p)
    selected = cache[year]
    window = form | test
    held, rets = {}, []
    for a, b in selected:
        path = pair_path(logp[a].to_numpy()[window], logp[b].to_numpy()[window], form[window], test[window], p)
        rets.append(path["net"][-1])
        if path["pos"][-1] != 0.0:
            held[f"{a}/{b}"] = {"z": float(path["z"][-1]), "beta": float(path["beta"][-1]),
                                "w_y": float(path["wy"][-1]), "w_x": float(path["wx"][-1])}
    return {"status": "scored", "formation": [str(idx[form].min().date()), str(idx[form].max().date())],
            "selected": [f"{a}/{b}" for a, b in selected], "held": held,
            "net_return": float(np.mean(rets)) if rets else 0.0}


# ---------------------------------------------------------------- update

def code_commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prereg_hashes() -> dict[str, str]:
    return {f: prereg.load(REPO / "research" / "prereg" / f).sha256 for f in PREREG_FILES}


@dataclass
class UpdateResult:
    appended: list[str]
    skipped: list[str]


def append(log: Path, eur: pd.DataFrame, data_sha256: str, source: str,
           now: dt.datetime | None = None) -> UpdateResult:
    """Append one record per new date in ``eur``. Dates already logged are
    skipped (first value wins, so an ECB revision never rewrites history)."""
    check_rows(eur)
    records = read_log(log)
    have = {r["date"] for r in records}
    hist = log_frame(records)
    now = now or dt.datetime.now(dt.UTC)
    hashes = prereg_hashes()
    commit = code_commit()
    eur = eur[list(ORDER)].dropna().sort_index()
    out, done, skipped = [], [], []
    regime_cache: dict = {}
    pairs_cache: dict = {}
    for day, row in eur.iterrows():
        key = str(day.date())
        if key in have:
            skipped.append(key)
            continue
        hist.loc[day] = row.to_numpy(float)
        px = usd_prices(hist.sort_index())
        rec = {"schema": SCHEMA, "date": key,
               "rates_per_eur": {c: float(row[c]) for c in CURRENCIES},
               "usd_per_unit": {c: float(px.loc[day, c]) for c in px.columns},
               "regime": regime_signal(px, day, cache=regime_cache),
               "pairs": pairs_signal(px, day, cache=pairs_cache),
               "provenance": {"code_commit": commit, "data_sha256": data_sha256, "source": source,
                              "written_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "prereg_sha256": hashes}}
        out.append(json.dumps(rec, sort_keys=True))
        have.add(key)
        done.append(key)
    if out:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(out) + "\n")
    return UpdateResult(done, skipped)


def next_start(log: Path) -> dt.date:
    records = read_log(log)
    if not records:
        return FORWARD_START
    return max(FORWARD_START, dt.date.fromisoformat(max(r["date"] for r in records)) + dt.timedelta(days=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("update", help="append new ECB days to the log")
    up.add_argument("--log", type=Path, required=True)
    up.add_argument("--csv", type=Path, help="read this ECB csvdata file instead of fetching")
    up.add_argument("--today", type=dt.date.fromisoformat, default=None)
    args = ap.parse_args(argv)
    today = args.today or dt.datetime.now(dt.UTC).date()
    if args.csv:
        eur, sha, src = read_csv(args.csv)
    else:
        start = next_start(args.log)
        if start > today:
            print(f"nothing to fetch: log is current through {start - dt.timedelta(days=1)}")
            return 0
        try:
            eur, sha, src = fetch(start, today)
        except urllib.error.HTTPError as exc:  # the ECB answers an empty window with 404
            if exc.code != 404:
                raise
            print(f"no ECB rows yet for {start}..{today}")
            return 0
        except pd.errors.EmptyDataError as exc:
            print(f"no ECB rows for {start}..{today}: {exc}")
            return 0
    res = append(args.log, eur, sha, src)
    print(f"appended {len(res.appended)} day(s) {res.appended}; skipped {len(res.skipped)} already logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
