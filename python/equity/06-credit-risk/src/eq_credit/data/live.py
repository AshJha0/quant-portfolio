"""Loader stub for freely available credit datasets (German credit /
Lending Club style CSVs).

This project's tests and examples run entirely on the seeded synthetic
generator in :mod:`eq_credit.data.synthetic` (per portfolio conventions: no
network access in test code paths).  For experimentation with real data,
:func:`load_credit_csv` maps two standard public schemas onto the column
names the pipeline expects:

* ``schema="german"``: the UCI Statlog German Credit data (as CSV with
  headers); target column ``kredit``/``class`` (1 = good) is mapped to
  ``default`` (1 = bad).
* ``schema="lendingclub"``: Lending Club loan exports; ``loan_status`` in
  {"Charged Off", "Default"} maps to ``default = 1``, "Fully Paid" to 0
  (other statuses dropped as unresolved).

Both datasets are consumer-credit, not corporate: the WOE/logit pipeline is
identical, but the Basel IRB functions in :mod:`eq_credit.portfolio_risk`
use the *corporate* correlation formula and would need the retail variants.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_credit_csv"]

_GERMAN_TARGETS = ("kredit", "class", "target")
_LC_STATUS_BAD = ("Charged Off", "Default")
_LC_STATUS_GOOD = ("Fully Paid",)


def load_credit_csv(path: str | Path, schema: str = "german") -> pd.DataFrame:
    """Load a public credit CSV and standardise the target column.

    Parameters
    ----------
    path : str or Path
        Local CSV file path (this function never touches the network).
    schema : {"german", "lendingclub"}
        Column-mapping convention, see module docstring.

    Returns
    -------
    pd.DataFrame with a binary ``default`` column (1 = bad).

    Raises
    ------
    FileNotFoundError, ValueError
        With informative messages; no silent fallbacks.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"credit CSV not found at {p}. Download e.g. the UCI Statlog "
            "German credit dataset or a Lending Club export manually; this "
            "loader never downloads data itself."
        )
    df = pd.read_csv(p)
    if schema == "german":
        target = next((c for c in _GERMAN_TARGETS if c in df.columns), None)
        if target is None:
            raise ValueError(
                f"no recognised target column among {_GERMAN_TARGETS} in {p.name}"
            )
        # Explicit codings, never guessed from class frequencies:
        # UCI Statlog "class"/"target": 1 = good, 2 = bad;
        # Fahrmeir-Hamerle "kredit": 1 = good, 0 = bad.
        vals = pd.to_numeric(df[target], errors="coerce")
        bad = 0 if target == "kredit" else 2
        if not vals.isin([1, bad]).all():
            raise ValueError(
                f"target column {target!r} must use the coding 1 = good, "
                f"{bad} = bad (UCI coding for class/target); found values "
                f"{sorted(df[target].dropna().unique())[:5]}"
            )
        df["default"] = (vals == bad).astype(int)
        return df.drop(columns=[target])
    if schema == "lendingclub":
        if "loan_status" not in df.columns:
            raise ValueError(f"'loan_status' column required for lendingclub schema in {p.name}")
        keep = df["loan_status"].isin(_LC_STATUS_BAD + _LC_STATUS_GOOD)
        out = df[keep].copy()
        out["default"] = out["loan_status"].isin(_LC_STATUS_BAD).astype(int)
        return out.drop(columns=["loan_status"])
    raise ValueError(f"unknown schema {schema!r}; use 'german' or 'lendingclub'")
