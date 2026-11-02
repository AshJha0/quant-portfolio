"""Machinery for the single-use 2022-2025 hold-out run.

The hold-out can be scored once. Everything here exists to make that one run
trustworthy afterwards and to make a second, accidental run impossible:

- ``signed_registration``: the only way past ``models.training_only``. It
  returns the registration only when ``prereg.py`` says it is signed and its
  hash equals the one the study code was written against.
- ``git_state`` and ``provenance``: the commit, whether the tree was dirty,
  the SHA-256 of every input file as it was read, the registration hashes, a
  UTC timestamp and the package versions, written into every result.
- ``write_once``: opens the result file in exclusive-create mode, so an
  existing hold-out result is never overwritten.
- ``deflated_sharpe``: the Bailey and Lopez de Prado (2014) deflated Sharpe
  ratio from summary moments, checked against the paper's worked example.
- ``synthetic_ecb_csv``: a seeded fake ECB file dated 2012-2025, used by
  ``--dry-run`` so the whole hold-out path runs without real hold-out data.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import prereg
from qp_core.stats import expected_max_sharpe
from scipy import stats as st

HOLDOUT_START = pd.Timestamp("2022-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")
HOLDOUT_YEARS = (2022, 2023, 2024, 2025)
PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "qp-core", "fx-pairs", "fx-regime")


class HoldoutRefused(RuntimeError):
    """A hold-out run was refused before it read any hold-out data."""


# ---- registration ---------------------------------------------------------

def signed_registration(path: Path, expected_sha256: str) -> prereg.Registration:
    """Return the registration at ``path`` only if it is signed with the hash
    the study code expects; otherwise raise ``HoldoutRefused``."""
    reg = prereg.load(path)
    if reg.sha256 != expected_sha256:
        raise HoldoutRefused(f"{path.name}: registration hash {reg.sha256[:12]} is not the "
                             f"expected {expected_sha256[:12]}; the registration was edited")
    if not reg.signed:
        raise HoldoutRefused(f"{path.name}: registration is not signed with hash {reg.sha256[:12]}")
    return reg


# ---- provenance -----------------------------------------------------------

def git_state(repo: Path) -> dict[str, Any]:
    """Commit and dirty flag of ``repo``. Untracked files count as dirty."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                              text=True).stdout
    try:
        commit = run("rev-parse", "HEAD").strip()
        status = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HoldoutRefused(f"cannot read git state of {repo}: {exc}") from exc
    return {"commit": commit, "dirty": bool(status.strip()),
            "dirty_paths": [line[3:] for line in status.splitlines()][:50]}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def provenance(git: dict[str, Any], inputs: dict[str, str], regs: dict[str, str],
               dry_run: bool) -> dict[str, Any]:
    return {"git_commit": git["commit"], "git_dirty": git["dirty"],
            "input_sha256": dict(inputs), "prereg_sha256": dict(regs),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "packages": package_versions(), "dry_run": dry_run}


def write_once(path: Path, doc: dict[str, Any]) -> Path:
    """Write ``doc`` to ``path``; refuse if the file already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(doc, indent=2, default=float) + "\n")
    except FileExistsError as exc:
        raise HoldoutRefused(f"{path} exists; a hold-out result is written once and never "
                             "overwritten") from exc
    return path


# ---- deflated Sharpe ------------------------------------------------------

def deflated_sharpe(sr: float, n_obs: int, skew: float, kurtosis: float, n_trials: int,
                    trial_sr_variance: float | None = None) -> dict[str, float]:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014, eq. 2 with the
    expected maximum of eq. 3), from per-period moments.

    ``sr`` is the per-period (not annualised) Sharpe ratio over ``n_obs``
    returns, ``kurtosis`` is the raw (non-excess) kurtosis. The benchmark
    ``sr0`` is the expected maximum Sharpe of ``n_trials`` skill-less trials
    whose Sharpe estimates have variance ``trial_sr_variance``; when that is
    not given it is the sampling variance of a zero-Sharpe estimate,
    ``1 / (n_obs - 1)``. Returns ``sr0`` and the deflated Sharpe, the
    probability that the true Sharpe exceeds ``sr0``.
    """
    if n_obs < 4:
        raise ValueError(f"need at least 4 observations, got {n_obs}")
    v = 1.0 / (n_obs - 1) if trial_sr_variance is None else float(trial_sr_variance)
    sr0 = expected_max_sharpe(int(n_trials), v)
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if not denom > 0:
        raise ValueError("non-positive denominator; the moments are inconsistent")
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return {"sr0_per_period": float(sr0), "dsr": float(st.norm.cdf(z)), "z": float(z)}


def deflated_sharpe_of(returns: np.ndarray, n_trials: int) -> dict[str, Any]:
    """Deflated Sharpe of a daily return series, with the moments it used."""
    r = np.asarray(returns, dtype=float)
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    if r.size < 4 or not sd > 0:
        return {"n_trials": n_trials, "dsr": None, "note": "flat or too-short series"}
    sr = float(r.mean() / sd)
    g3, g4 = float(st.skew(r)), float(st.kurtosis(r, fisher=False))
    out = deflated_sharpe(sr, r.size, g3, g4, n_trials)
    return {"n_trials": n_trials, "trial_sr_variance": "1/(n-1), null sampling variance",
            "sr_per_period": sr, "skew": g3, "kurtosis": g4, "n_obs": int(r.size), **out}


# ---- synthetic data for --dry-run ----------------------------------------

def synthetic_ecb_csv(seed: int = 0, start: str = "2012-01-02",
                      end: str = "2025-12-31") -> str:
    """A fake ECB ``csvdata`` body: units of each G10 currency per EUR on
    business days from ``start`` to ``end``. Seeded random walks with two
    cointegrated pairs and a volatility regime, so every branch of the
    hold-out path has something to do. None of it is market data."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    vol = np.where((np.arange(n) // 250) % 3 == 2, 0.012, 0.005)
    common = np.cumsum(vol * rng.standard_normal(n))
    ccys = ["USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK"]
    level = {"USD": 1.2, "GBP": 0.85, "JPY": 130.0, "CHF": 1.1, "AUD": 1.6, "CAD": 1.5,
             "NZD": 1.7, "SEK": 10.0, "NOK": 9.5}
    walks = {c: np.cumsum(vol * rng.standard_normal(n)) for c in ccys}
    walks["NZD"] = walks["AUD"] + 0.002 * rng.standard_normal(n)  # cointegrated with AUD
    walks["NOK"] = walks["SEK"] + 0.002 * rng.standard_normal(n)  # cointegrated with SEK
    rows = ["KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE"]
    for c in ccys:
        vals = level[c] * np.exp(walks[c] + common)
        rows += [f"EXR.D.{c}.EUR.SP00.A,D,{c},EUR,SP00,A,{d.date()},{v:.6f}"
                 for d, v in zip(days, vals, strict=False)]
    return "\n".join(rows) + "\n"
