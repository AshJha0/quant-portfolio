"""US Treasury zero curve from the Gurkaynak-Sack-Wright parameters (W23 D).

Plain idea: the Federal Reserve publishes, for every business day since
1961, six numbers that describe the whole Treasury zero-coupon curve. From
them we can read the yield of any maturity from about 1 to 30 years.

Formal statement (Gurkaynak, Sack and Wright 2007, "The U.S. Treasury yield
curve: 1961 to the present", J. Monetary Economics 54(8), eq. 22). The
continuously compounded zero yield at maturity ``n`` years is

    y(n) = b0 + b1 (1 - e^{-n/t1}) / (n/t1)
              + b2 [(1 - e^{-n/t1}) / (n/t1) - e^{-n/t1}]
              + b3 [(1 - e^{-n/t2}) / (n/t2) - e^{-n/t2}]

(Svensson 1994). On days before the Fed fitted the fourth term, b3 and t2
are missing and the Nelson-Siegel form (b3 = 0) applies. Parameters are in
percent; this module returns decimals.

Data: ``feds200628.csv`` from federalreserve.gov/data/yield-curve-tables,
free, no sign-up. The file starts with nine lines of notes, then a header
row with ``Date``, ``BETA0``..``BETA3``, ``TAU1``, ``TAU2`` and the fitted
yields. ``load`` reads those six columns and the date, nothing else.

Worked example (``tests/test_ust.py``): b0 = 5, b1 = -2, b2 = b3 = 0,
t1 = 1. At n = 1, ``(1 - e^-1)/1 = 0.632121``, so ``y = 5 - 1.264241 =
3.735759`` percent; as n grows y tends to b0 = 5 percent.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")
URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"


def _load(n: np.ndarray, tau: float) -> np.ndarray:
    x = n / tau
    return (1.0 - np.exp(-x)) / x


def zero_yield(params: pd.Series | dict, n: float | np.ndarray) -> np.ndarray:
    """Zero yield(s), decimal, continuously compounded, for maturities ``n`` > 0."""
    m = np.atleast_1d(np.asarray(n, dtype=float))
    if np.any(m <= 0):
        raise ValueError("maturity must be positive")
    p = {k: float(params[k]) if k in params else float("nan") for k in PARAMS}
    b0, b1, b2, b3, t1, t2 = (p[k] for k in PARAMS)
    l1 = _load(m, t1)
    y = b0 + b1 * l1 + b2 * (l1 - np.exp(-m / t1))
    if np.isfinite(b3) and np.isfinite(t2) and b3 != 0.0:
        l2 = _load(m, t2)
        y = y + b3 * (l2 - np.exp(-m / t2))
    return y / 100.0


def load(path: Path) -> tuple[pd.DataFrame, str]:
    """(parameters by date, SHA-256 of the file bytes). Rows with a missing
    BETA0 or TAU1 are dropped; that only happens on the file's blank lines."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), usecols=["Date", *PARAMS])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["BETA0", "TAU1"])
    return df, sha
