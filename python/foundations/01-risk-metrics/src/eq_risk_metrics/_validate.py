"""Shared input checks for the public functions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def finite_series(x, func: str, name: str = "returns", min_len: int = 1) -> pd.Series:
    """Return ``x`` as a float Series; raise ValueError if too short or non-finite."""
    s = pd.Series(x, dtype=float) if not isinstance(x, pd.Series) else x.astype(float)
    if len(s) < min_len:
        raise ValueError(f"{func}: {name} needs at least {min_len} observation(s), got {len(s)}")
    if not np.isfinite(s.to_numpy()).all():
        raise ValueError(f"{func}: {name} contains a non-finite value (NaN or inf)")
    return s


def finite_scalar(v, func: str, name: str) -> float:
    if not np.isfinite(v):
        raise ValueError(f"{func}: {name} must be finite, got {v!r}")
    return float(v)
