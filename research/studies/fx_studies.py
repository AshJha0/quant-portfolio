"""Training-period studies on ECB reference rates (1999-01-04 to 2021-12-31).

Each study takes its parameters from the pre-registration it names, refuses
any data on or after 2022-01-01 (``models.training_only``), and writes one
JSON result carrying the registration hash so ``research/tools/prereg.py
check`` can verify it.

Run: ``python research/studies/fx_studies.py [pairs|regime|all]``. The ECB
file is downloaded once into ``data/raw/`` by ``data/fetch/qp_fetch.py``.

Hold-out (single use, 2022-01-01 to 2025-12-31):
``python research/studies/fx_studies.py regime --mode holdout``. The pairs
registration was withdrawn before the hold-out (DR-R4-5), so ``pairs`` is
refused and ``all`` means ``regime``. It refuses
unless the registration is signed with the hashes in ``SIGNED_SHA256``,
the git tree is clean and no ``*_holdout.json`` exists yet. ``--dry-run``
runs the same path on a seeded synthetic file with fake 2012-2025 dates and
writes to a temporary directory; it never reads real data or the network.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "research" / "tools"), str(REPO / "data" / "fetch")]

import holdout as ho  # noqa: E402
import prereg  # noqa: E402
import trials  # noqa: E402
from fx_pairs.cointegration import engle_granger  # noqa: E402
from fx_regime.hmm import filtered_probabilities, fit_hmm  # noqa: E402
from models import johansen_trace, kalman_hedge, training_only  # noqa: E402
from qp_core.stats import block_bootstrap_ci, lo_sharpe_se, sharpe_ratio  # noqa: E402

PREREG = REPO / "research" / "prereg"
RESULTS = REPO / "research" / "results"
TRAIN_START, TRAIN_END = "1999-01-04", "2021-12-31"

# Parameters copied from the pre-registrations. A test checks they match.
PAIRS = dict(formation_years=3, embargo_days=5, first_test_year=2003, eg_level="5%",
             delta=1e-6, entry_z=2.0, exit_z=0.5, cost_bp=1.0, lags=1)
REGIME = dict(first_test_year=2003, n_states=2, seed=0, n_init=5, threshold=0.5)

# The registration hashes this code was written against (signed in 35d351a).
# A hold-out run refuses unless prereg.py reports each file signed with this hash.
SIGNED_SHA256 = {
    "pairs-fx-g10.md": "dfca5c7ae8ea15c6483ed8c3a9b6946f23bcfa0aea71f45b8e499908ae953dcc",
    "regime-fx-basket.md": "bd64743240ec273a2e7e92a69c6ab6bb3d58054f5f99d9b184ff0e65fa11e687",
}
STUDIES = {"pairs": ("pairs_fx_g10", "pairs-fx-g10.md"),
           "regime": ("regime_fx_basket", "regime-fx-basket.md")}
# Registrations withdrawn before any hold-out data was seen (research/README.md,
# DR-R4-5). Their hold-out is never run; ``all`` in hold-out mode skips them.
WITHDRAWN = {"pairs": "DR-R4-5 (2026-11-02): withdrawn before the hold-out; the signal has "
                      "near-zero power as registered"}


def holdout_studies(which: str) -> list[str]:
    """Studies a hold-out run of ``which`` scores. Refuses a withdrawn study."""
    if which in WITHDRAWN:
        raise ho.HoldoutRefused(f"{which}: registration withdrawn, {WITHDRAWN[which]}")
    names = [n for n in STUDIES if n not in WITHDRAWN] if which == "all" else [which]
    if which not in STUDIES and which != "all":
        raise ho.HoldoutRefused(f"unknown study {which!r}")
    return names


def usd_prices(eur_rates: pd.DataFrame) -> pd.DataFrame:
    """USD value of one unit of each non-USD currency, from EUR-based rates."""
    usd = eur_rates["USD"]
    out = pd.DataFrame({c: usd / eur_rates[c] for c in eur_rates.columns if c != "USD"})
    out["EUR"] = usd
    return out.dropna()


def _write(name: str, reg_file: str, period: tuple[str, str], payload: dict,
           params: dict | None = None, kind: str = "train", git: tuple | None = None) -> Path:
    """Write ``results/<name>.json`` and append the run to ``trials.jsonl``."""
    reg = prereg.load(PREREG / reg_file)
    # Callers pass the git state read at the start of the run, before any
    # output exists; reading it here is the fallback for single writes.
    git = git if git is not None else trials.git_state(REPO)
    RESULTS.mkdir(parents=True, exist_ok=True)
    doc = {"prereg": f"research/prereg/{reg_file}", "prereg_sha256": reg.sha256,
           "registration_signed": reg.signed, "period": {"start": period[0], "end": period[1]},
           "dataset": "ECB euro foreign exchange reference rates (EXR), daily", **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2, default=float) + "\n", encoding="utf-8")
    trials.append(name, kind, f"research/prereg/{reg_file}", reg.sha256, f"research/results/{name}.json",
                  params or {}, trials.metric_of(payload), git=git)
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


def pair_returns(ly: pd.Series, lx: pd.Series, test_mask: np.ndarray, p: dict,
                 activity: dict | None = None) -> np.ndarray:
    """Net daily return of one pair over the test rows (position decided at
    close t, earns the return of t+1; cost on gross notional traded).

    If ``activity`` is given it is filled with the test-year counts: days
    with |z| above the entry level, entries, days in the market and the
    largest |z|."""
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
    if activity is not None:
        zt = np.abs(z[test_mask])
        held = pos[test_mask] != 0.0
        activity.update(entry_signal_days=int((zt > p["entry_z"]).sum()),
                        entries=int((held & ~np.concatenate([[False], held[:-1]])).sum()),
                        days_in_market=int(held.sum()), max_abs_z=float(zt.max()) if zt.size else 0.0)
    return (gross - cost)[test_mask]


def pairs_fold_masks(index: pd.DatetimeIndex, year: int, p: dict = PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """(formation, test) row masks for test ``year``: formation is the
    ``formation_years`` calendar years before it, ending ``embargo_days``
    business days before its first day."""
    test = np.asarray(index.year == year)
    form_end = np.flatnonzero(test)[0] - p["embargo_days"]
    form = np.asarray(index.year >= year - p["formation_years"]) & (np.arange(len(index)) < form_end)
    return form, test


def pairs_study(px: pd.DataFrame, p: dict = PAIRS, years: tuple[int, ...] | None = None) -> dict:
    """Walk-forward pairs study. ``years=None`` is the training run (guarded
    by ``training_only``); hold-out callers pass the years after the
    signature check in ``run_holdout``."""
    if years is None:
        px = training_only(px)
        years = tuple(range(p["first_test_year"], px.index.max().year + 1))
    logp = pd.DataFrame(np.log(px))
    names = list(px.columns)
    folds, daily = [], []
    for year in years:
        form, test = pairs_fold_masks(pd.DatetimeIndex(logp.index), year, p)
        selected = []
        for a, b in itertools.combinations(names, 2):
            fy, fx = logp[a][form], logp[b][form]
            eg = engle_granger(fy.to_numpy(), fx.to_numpy())
            jo = johansen_trace(np.column_stack([fy, fx]), k_ar_diff=p["lags"])
            if eg.cointegrated and jo.rank >= 1:
                selected.append((a, b))
        window = form | test
        acts: list[dict] = [{} for _ in selected]
        rets = [pair_returns(logp[a][window], logp[b][window], test[window], p, act)
                for (a, b), act in zip(selected, acts, strict=True)]
        port = np.mean(rets, axis=0) if rets else np.zeros(int(test.sum()))
        daily.append(port)
        folds.append({"year": year, "pairs": [f"{a}/{b}" for a, b in selected],
                      "net_return": float(port.sum()),
                      "entry_signal_days": sum(x["entry_signal_days"] for x in acts),
                      "entries": sum(x["entries"] for x in acts),
                      "pair_days_in_market": sum(x["days_in_market"] for x in acts),
                      "max_abs_z": max((x["max_abs_z"] for x in acts), default=None)})
    r = np.concatenate(daily)
    return {"folds": folds, "candidate_pairs": math.comb(len(names), 2), "trial_count": 3,
            "primary_metric": "annualised net Sharpe, walk-forward test years", **_sharpe_block(r),
            "deflated_sharpe": ho.deflated_sharpe_of(r, n_trials=3)}


def usd_basket_returns(px: pd.DataFrame) -> pd.Series:
    """Equal-weight daily log return of holding the G10 currencies against USD."""
    return pd.DataFrame(np.log(px)).diff().dropna().mean(axis=1)


def regime_study(px: pd.DataFrame, p: dict = REGIME, years: tuple[int, ...] | None = None) -> dict:
    """Yearly expanding-window HMM refit. ``years=None`` is the training run."""
    if years is None:
        px = training_only(px)
        years = tuple(range(p["first_test_year"], px.index.max().year + 1))
    r = usd_basket_returns(px)
    ryear = pd.DatetimeIndex(r.index).year
    next_ret, in_high = [], []
    for year in years:
        fit = r[ryear < year].to_numpy()[:, None]
        model = fit_hmm(fit, p["n_states"], seed=p["seed"], n_init=p["n_init"])
        high = int(np.argmax(model.covs[:, 0, 0]))
        upto = r[ryear <= year]
        prob = filtered_probabilities(model, upto.to_numpy()[:, None])[:, high]
        mask = np.asarray(pd.DatetimeIndex(upto.index).year == year)[:-1]  # state at close t, return of t+1
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


def pairs_decision(res: dict) -> dict:
    """Pairs decision rule: reject H0 if the one-sided 95 % lower bound of the
    hold-out net Sharpe is above zero. Primary bound: Sharpe - 1.645 x the Lo
    (2002) Newey-West SE. The bootstrap interval's lower end is reported beside it."""
    sr, se, ci = res.get("sharpe"), res.get("sharpe_se_lo"), res.get("ci95")
    if sr is None or se is None:
        return {"rule": "Sharpe - 1.645 * Lo NW SE > 0", "h0_rejected": False,
                "note": "no trading in the hold-out; H0 not rejected"}
    lower = sr - 1.645 * se
    return {"rule": "Sharpe - 1.645 * Lo NW SE > 0", "lower_bound_95_one_sided": lower,
            "h0_rejected": bool(lower > 0),
            "secondary_bootstrap": {
                "label": ("lower end of the two-sided 95 % circular block-bootstrap interval "
                          "(a 97.5 % one-sided bound); reported only, not the decision (DR-R4-1)"),
                "lower_end": None if ci is None else ci[0],
                "above_zero": None if ci is None else bool(ci[0] > 0)}}


def regime_decision(res: dict) -> dict:
    return {"rule": "HAC t-statistic of the high-vol mean < -1.645",
            "h0_rejected": bool(res["hac_t_high"] < -1.645)}


def load_ecb(path: Path) -> tuple[pd.DataFrame, str]:
    """Read an ECB csvdata file and hash exactly the bytes that were parsed."""
    from qp_fetch import parse_ecb_csv

    payload = path.read_bytes()
    return parse_ecb_csv(payload.decode("utf-8")), ho.sha256_bytes(payload)


def holdout_prices(eur: pd.DataFrame) -> pd.DataFrame:
    """USD prices cut at the hold-out end. Refuses data that does not reach
    December 2025 or starts too late for the 2022 formation window."""
    px = usd_prices(eur)
    px = px[px.index <= ho.HOLDOUT_END]
    if px.empty or px.index.max() < pd.Timestamp("2025-12-01"):
        raise ho.HoldoutRefused("input data does not reach December 2025")
    if px.index.min() > pd.Timestamp("2018-12-31"):
        raise ho.HoldoutRefused("input data starts after 2018; the 2022 formation window needs 2019-2021")
    return px


def check_holdout_ready(which: str, out_dir: Path, dry_run: bool, repo: Path = REPO,
                        prereg_dir: Path = PREREG) -> tuple[dict, dict, dict]:
    """Every refusal, made before any data is read or downloaded."""
    names = holdout_studies(which)
    regs = {n: ho.signed_registration(prereg_dir / STUDIES[n][1], SIGNED_SHA256[STUDIES[n][1]])
            for n in names}
    git = ho.git_state(repo)
    if git["dirty"] and not dry_run:
        raise ho.HoldoutRefused(f"git tree is dirty ({', '.join(git['dirty_paths'][:5])}); commit first")
    suffix = "_holdout_dryrun.json" if dry_run else "_holdout.json"
    targets = {n: out_dir / f"{STUDIES[n][0]}{suffix}" for n in names}
    for t in targets.values():
        if t.exists():
            raise ho.HoldoutRefused(f"{t} exists; the hold-out is single-use")
    return regs, git, targets


def run_holdout(which: str, data_path: Path, out_dir: Path, dry_run: bool,
                repo: Path = REPO, prereg_dir: Path = PREREG) -> list[Path]:
    """The single hold-out run: years 2022-2025, each fold formed and fitted
    only on data before it, scored by the registration's metric and rule."""
    regs, git, targets = check_holdout_ready(which, out_dir, dry_run, repo, prereg_dir)
    eur, sha = load_ecb(data_path)
    px = holdout_prices(eur)
    test_px = px[px.index >= ho.HOLDOUT_START]
    period = (str(test_px.index.min().date()), str(test_px.index.max().date()))
    try:
        data_name = data_path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        data_name = data_path.name
    reg_hashes = {f"research/prereg/{STUDIES[n][1]}": regs[n].sha256 for n in targets}
    written = []
    for n, target in targets.items():
        if n == "pairs":
            res = pairs_study(px, years=ho.HOLDOUT_YEARS)
            res["decision"] = pairs_decision(res)
        else:
            res = regime_study(px, years=ho.HOLDOUT_YEARS)
            res["decision"] = regime_decision(res)
        reg_file = STUDIES[n][1]
        doc = {"prereg": f"research/prereg/{reg_file}", "prereg_sha256": regs[n].sha256,
               "registration_signed": regs[n].signed, "mode": "holdout",
               "period": {"start": period[0], "end": period[1]},
               "data_start": str(px.index.min().date()),
               "dataset": ("SYNTHETIC (dry run, fake dates)" if dry_run
                           else "ECB euro foreign exchange reference rates (EXR), daily"),
               "provenance": ho.provenance(git, {data_name: sha}, reg_hashes, dry_run), **res}
        written.append(ho.write_once(target, doc))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=["pairs", "regime", "all"])
    ap.add_argument("--mode", default="train", choices=["train", "holdout"])
    ap.add_argument("--dry-run", action="store_true", help="hold-out path on synthetic data, no network")
    ap.add_argument("--out", type=Path, default=None, help="output directory (dry run: a temp dir)")
    a = ap.parse_args(argv)
    git = trials.git_state(REPO)  # before the run writes anything
    if a.dry_run:
        out = a.out or Path(tempfile.mkdtemp(prefix="qp_holdout_dryrun_"))
        if out.resolve() == RESULTS.resolve():
            raise ho.HoldoutRefused("a dry run never writes into research/results")
        out.mkdir(parents=True, exist_ok=True)
        data = out / "synthetic_ecb_2012-2025.csv"
        if not data.exists():
            data.write_text(ho.synthetic_ecb_csv(), encoding="utf-8", newline="\n")
        for path in run_holdout(a.which, data, out, dry_run=True):
            print(path)
        return 0
    if a.mode == "holdout":
        from qp_fetch import RAW_DIR, fetch_ecb_fx

        out = a.out or RESULTS
        check_holdout_ready(a.which, out, dry_run=False)  # refuse before any download
        data = RAW_DIR / "ecb_exr_1999-01-01_2025-12-31.csv"
        if not data.exists():
            fetch_ecb_fx(end="2025-12-31")
        for path in run_holdout(a.which, data, out, dry_run=False):
            print(path)
        return 0
    train(a.which, git=git)
    return 0


def train(which: str, git: tuple | None = None) -> None:
    from qp_fetch import RAW_DIR, fetch_ecb_fx, parse_ecb_csv

    cached = RAW_DIR / f"ecb_exr_1999-01-01_{TRAIN_END}.csv"
    eur = parse_ecb_csv(cached.read_text()) if cached.exists() else fetch_ecb_fx(end=TRAIN_END)
    px = usd_prices(eur)
    period = (str(px.index.min().date()), str(px.index.max().date()))
    if which in ("pairs", "all"):
        print(_write("pairs_fx_g10_train", "pairs-fx-g10.md", period, pairs_study(px),
                     params=PAIRS, git=git))
    if which in ("regime", "all"):
        print(_write("regime_fx_basket_train", "regime-fx-basket.md", period, regime_study(px),
                     params=REGIME, git=git))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ho.HoldoutRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(3)
