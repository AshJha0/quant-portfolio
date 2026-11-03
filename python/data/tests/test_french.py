"""Offline tests for the Kenneth French library adapter. The two zips in
``fixtures/`` were downloaded 2026-11-03 from
mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/ (files built from the
202609 CRSP database)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qp_data import french
from qp_data.errors import SchemaError, SourceNotPermitted

FIX = Path(__file__).parent / "fixtures"


def test_worked_example_first_row_in_decimals():
    f = french.parse_french((FIX / "F-F_Research_Data_Factors.zip").read_bytes())
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    row = f.loc[pd.Timestamp("1926-07-31")]
    assert row.to_list() == pytest.approx([0.0289, -0.0242, -0.0275, 0.0022], abs=1e-12)
    assert f.index.is_monotonic_increasing and not f.index.has_duplicates
    assert f.index[-1] == pd.Timestamp("2026-09-30")
    assert (f.index == f.index + pd.offsets.MonthEnd(0)).all()


def test_annual_table_is_separate_and_year_end_stamped():
    raw = (FIX / "F-F_Research_Data_Factors.zip").read_bytes()
    a = french.parse_french(raw, table="annual")
    assert a.index[0] == pd.Timestamp("1927-12-31")
    assert a.loc["2025-12-31", "RF"].item() == pytest.approx(0.0425)
    m = french.parse_french(raw)
    assert len(m) > 12 * len(a) - 24  # annual rows did not leak into the monthly table


def test_momentum_file_has_one_column():
    f = french.parse_french((FIX / "F-F_Momentum_Factor.zip").read_bytes())
    assert list(f.columns) == ["Mom"]
    assert f.iloc[0, 0] == pytest.approx(0.0057)
    assert f.index[0] == pd.Timestamp("1927-01-31")


def _zip(text: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.CSV", text)
    return b.getvalue()


def test_daily_codes_and_missing_values():
    text = "preamble\r\n\r\n,A,B\r\n20240102, 1.00, -99.99\r\n20240103, -999, 0.50\r\n\r\n"
    f = french.parse_french(_zip(text))
    assert list(f.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert f.loc["2024-01-02", "A"].item() == pytest.approx(0.01)
    assert np.isnan(f.loc["2024-01-02", "B"].item()) and np.isnan(f.loc["2024-01-03", "A"].item())


@pytest.mark.parametrize("text", [
    "no table here\n",
    ",A,B\n202401, 1.0\n",           # ragged row
    ",A\n202401, x\n",              # non-numeric
    ",A\n202402, 1\n202401, 1\n",   # out of order
])
def test_schema_errors(text):
    with pytest.raises(SchemaError):
        french.parse_french(text.encode())


def test_url_allow_list():
    assert french.french_url("french:F-F_Research_Data_Factors") == (
        french.BASE_URL + "F-F_Research_Data_Factors_CSV.zip")
    with pytest.raises(SourceNotPermitted):
        french.french_url("french:Some_Other_File")
    assert french.SOURCE.name == "french"
