"""Forward evaluation of the two forward-hold-out registrations.

Scores ``prereg/pairs-fx-g10-v2.md`` and ``prereg/momentum-fx-g10.md`` on
the forward log (schema 1, ``research/forward/README.md``) as their texts
and ``research/forward/SPEC.md`` say:

- only fixes dated 2026-11-03 or later are scored; the window is
  2026-11-03 to 2028-11-02 inclusive;
- formation windows (pairs) and lookbacks (momentum) read the history file
  (ECB fixes 1999-01-04 to 2026-11-02, SHA-256 from the signing commit)
  followed by the log, as one contiguous series;
- the decision rule (both): reject H0 if Sharpe - 1.645 x Lo SE > 0; for
  pairs, no trade in the window means H0 is not rejected.

A verdict is produced only when all of these hold: today is on or after
2028-11-13, the log reaches 2028-11-02 with no missing ECB business day
(TARGET closing days excepted), every ``usd_per_unit`` agrees with
``rates_per_eur``, the history file matches its recorded SHA-256, and the
registration is signed. Otherwise the script prints interim statistics,
labelled INTERIM, which decide nothing, and writes no result file.

Run: ``python research/forward/evaluate.py --log forward/log.jsonl
--history research/forward/history_1999-01-04_2026-11-02.csv
--history-sha256 <hash> --log-commit <sha> [--out DIR] [--today YYYY-MM-DD]``
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "research" / "studies"), str(REPO / "research" / "tools"), str(REPO / "data" / "fetch")]

import forward_studies as fs  # noqa: E402
import holdout as ho  # noqa: E402
import prereg  # noqa: E402
import qp_fetch  # noqa: E402
from fx_studies import _sharpe_block, usd_prices  # noqa: E402

FORWARD_START = dt.date(2026, 11, 3)
WINDOW_END = dt.date(2028, 11, 2)
EVALUATION_DATE = dt.date(2028, 11, 13)
FOLDS = ((dt.date(2026, 11, 3), dt.date(2027, 11, 2)), (dt.date(2027, 11, 3), dt.date(2028, 11, 2)))
FORMATION_DAYS = 756
Z_ONE_SIDED = 1.645
LOG_SCHEMA = 1
REL_TOL = 1e-9
REGS = {"pairs": ("pairs-fx-g10-v2.md", fs.PAIRS_V2_TRIALS), "momentum": ("momentum-fx-g10.md", 1)}


class EvaluationError(RuntimeError):
    """The inputs cannot be scored (bad log, bad history file)."""


# ---- calendar ------------------------------------------------------------

def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return dt.date(year, month, day + 1)


def target_closed(day: dt.date) -> bool:
    """TARGET closing days, on which the ECB publishes no reference rate."""
    e = _easter(day.year)
    return day.weekday() >= 5 or (day.month, day.day) in {(1, 1), (5, 1), (12, 25), (12, 26)} or day in {
        e - dt.timedelta(days=2), e + dt.timedelta(days=1)}


def expected_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    out, d = [], start
    while d <= end:
        if not target_closed(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# ---- inputs --------------------------------------------------------------

def read_log(path: Path) -> pd.DataFrame:
    """USD-per-unit prices from a schema-1 log, checked line by line."""
    rows: dict[pd.Timestamp, dict] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("schema") != LOG_SCHEMA:
            raise EvaluationError(f"line {n}: schema {rec.get('schema')!r}, expected {LOG_SCHEMA}")
        day = dt.date.fromisoformat(rec["date"])
        if day < FORWARD_START:
            raise EvaluationError(f"line {n}: {day} is before {FORWARD_START}")
        ts = pd.Timestamp(day)
        if ts in rows:
            raise EvaluationError(f"line {n}: duplicate date {day}")
        eur = pd.DataFrame([rec["rates_per_eur"]], index=[ts])
        derived = usd_prices(eur).iloc[0]
        logged = rec["usd_per_unit"]
        if set(logged) != set(derived.index):
            raise EvaluationError(f"line {n}: usd_per_unit currencies {sorted(logged)} != {sorted(derived.index)}")
        for c, v in logged.items():
            if not np.isclose(v, derived[c], rtol=REL_TOL, atol=0.0):
                raise EvaluationError(f"line {n}: usd_per_unit[{c}]={v} disagrees with rates_per_eur ({derived[c]})")
        rows[ts] = logged
    if not rows:
        raise EvaluationError("forward log is empty")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def read_history(path: Path, sha256: str) -> pd.DataFrame:
    """USD-per-unit prices from the ECB history file, after its hash check."""
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256.lower():
        raise EvaluationError(f"history file sha256 {got} != recorded {sha256}")
    px = usd_prices(qp_fetch.parse_ecb_csv(data.decode("utf-8")))
    if px.index.max() >= pd.Timestamp(FORWARD_START):
        raise EvaluationError(f"history file reaches {px.index.max().date()}; it must end before {FORWARD_START}")
    return px


def combine(history: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    log = log[sorted(history.columns)] if set(log.columns) == set(history.columns) else None
    if log is None:
        raise EvaluationError("history and log carry different currencies")
    return pd.concat([history[sorted(history.columns)], log]).astype(float)


def missing_dates(log: pd.DataFrame, end: dt.date) -> list[dt.date]:
    have = {d.date() for d in log.index}
    return [d for d in expected_dates(FORWARD_START, end) if d not in have]


# ---- the two registrations -----------------------------------------------

def pairs_returns(px: pd.DataFrame, through: dt.date) -> tuple[pd.Series, list[dict]]:
    """Daily portfolio return of pairs v2 over the scored days up to ``through``."""
    logp = pd.DataFrame(np.log(px))
    idx = pd.DatetimeIndex(logp.index)
    p = fs.PAIRS_V2
    out, folds = [], []
    for start, end in FOLDS:
        test = np.asarray((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(min(end, through))))
        if not test.any():
            continue
        first = int(np.flatnonzero(test)[0])
        form_end = first - p["embargo_days"]
        if form_end < FORMATION_DAYS:
            raise EvaluationError(f"fold {start}: fewer than {FORMATION_DAYS} formation days of history")
        form = np.zeros(len(idx), dtype=bool)
        form[form_end - FORMATION_DAYS:form_end] = True
        selected = fs.select_pairs(logp, form, p)
        window = form | test
        rets, acts = [], []
        for a, b in selected:
            r, act = fs.pair_returns_level(logp[a].to_numpy()[window], logp[b].to_numpy()[window], test[window], p)
            rets.append(r)
            acts.append(act)
        port = np.mean(rets, axis=0) if rets else np.zeros(int(test.sum()))
        out.append(pd.Series(port, index=idx[test]))
        folds.append({"start": start.isoformat(), "end": end.isoformat(), "pairs": [f"{a}/{b}" for a, b in selected],
                      "net_return": float(port.sum()),
                      **{k: sum(x[k] for x in acts) for k in ("entry_signal_days", "entries", "days_in_market")}})
    return (pd.concat(out) if out else pd.Series(dtype=float)), folds


def momentum_scored(px: pd.DataFrame, through: dt.date) -> pd.Series:
    """Daily net return of the momentum rule over the scored days; the build
    of the weights in force on 2026-11-03 is charged on 2026-11-03."""
    p = fs.MOMENTUM
    r = fs.momentum_returns(px, p)
    w = fs.momentum_weights(px, p)
    idx = pd.DatetimeIndex(r.index)
    keep = (idx >= pd.Timestamp(FORWARD_START)) & (idx <= pd.Timestamp(through))
    r = r[keep].copy()
    if len(r):
        prior = w.loc[w.index < pd.Timestamp(FORWARD_START)]
        if prior.empty:
            raise EvaluationError("no lookback history before 2026-11-03")
        r.iloc[0] -= p["cost_bp"] * 1e-4 * float(prior.iloc[-1].abs().sum())
    return r


def score(r: np.ndarray) -> dict:
    block = _sharpe_block(r)
    s, se = block["sharpe"], block["sharpe_se_lo"]
    reject = bool(s is not None and se is not None and s - Z_ONE_SIDED * se > 0)
    return {**block, "lower_bound_one_sided_95": None if s is None or se is None else s - Z_ONE_SIDED * se,
            "reject_h0": reject}


# ---- driver --------------------------------------------------------------

def evaluate(log_path: Path, history_path: Path, history_sha256: str, log_commit: str,
             today: dt.date, repo: Path = REPO) -> dict:
    """Return one report per registration; ``final`` says whether it is a verdict."""
    log = read_log(log_path)
    px = combine(read_history(history_path, history_sha256), log)
    last = min(log.index.max().date(), WINDOW_END)
    gaps = missing_dates(log, last)
    reasons = []
    if today < EVALUATION_DATE:
        reasons.append(f"today {today} is before the evaluation date {EVALUATION_DATE}")
    if log.index.max().date() < WINDOW_END:
        reasons.append(f"log ends {log.index.max().date()}, before the window end {WINDOW_END}")
    if gaps:
        reasons.append(f"{len(gaps)} ECB business day(s) missing from the log, first {gaps[0]}")
    pr, folds = pairs_returns(px, last)
    series = {"pairs": pr, "momentum": momentum_scored(px, last)}
    reports = {}
    for key, (name, trials) in REGS.items():
        reg = prereg.load(repo / "research" / "prereg" / name)
        why = list(reasons) + ([] if reg.signed else ["registration is not signed"])
        r = series[key].to_numpy()
        stats = score(r)
        if key == "pairs" and not any(f["entries"] for f in folds):
            stats["reject_h0"] = False
            stats["note"] = "no pair traded in the window: H0 not rejected"
        rep = {"prereg": f"research/prereg/{name}", "prereg_sha256": reg.sha256, "mode": "forward",
               "period": {"start": FORWARD_START.isoformat(), "end": last.isoformat()},
               "forward_log_commit": log_commit, "history_sha256": history_sha256.lower(),
               "final": not why, "not_final_because": why, **stats,
               "deflated_sharpe": ho.deflated_sharpe_of(r, n_trials=trials) if stats["sharpe"] is not None else None,
               "deflated_sharpe_programme": ho.deflated_sharpe_of(r) if stats["sharpe"] is not None else None,
               "role": "monitoring (W6: single-series forward tests confirm nothing)"}
        if key == "pairs":
            rep["folds"] = folds
        if why:  # interim: no verdict is reported, only descriptive numbers
            rep["reject_h0"] = None
            rep["label"] = "INTERIM - descriptive only, decides nothing"
        reports[key] = rep
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--history-sha256", required=True)
    ap.add_argument("--log-commit", required=True)
    ap.add_argument("--out", type=Path, help="directory for the result JSON (final verdicts only)")
    ap.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today())
    a = ap.parse_args(argv)
    try:
        reports = evaluate(a.log, a.history, a.history_sha256, a.log_commit, a.today)
    except EvaluationError as e:
        print(f"REFUSED: {e}")
        return 1
    for key, rep in reports.items():
        if not rep["final"]:
            print(f"INTERIM {key} (no verdict): " + "; ".join(rep["not_final_because"]))
            print(f"  descriptive only: n_days={rep['n_days']} sharpe={rep['sharpe']} lo_se={rep['sharpe_se_lo']}")
            continue
        print(f"FINAL {key}: sharpe={rep['sharpe']} lo_se={rep['sharpe_se_lo']} reject_h0={rep['reject_h0']}")
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            path = a.out / f"{Path(rep['prereg']).stem.replace('-', '_')}_forward.json"
            path.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
