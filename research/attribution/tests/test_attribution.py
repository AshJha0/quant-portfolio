"""Tests for research/attribution/attribution.py (W23a). Oracle: statsmodels
OLS with cov_type="HAC" (Bartlett kernel)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "attribution"), str(REPO / "research" / "lib")]

import attribution as at  # noqa: E402
import panel  # noqa: E402

IDX = pd.date_range("2000-01-31", periods=120, freq="ME")


def _data(seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.normal(0, 0.02, size=(120, 2)), index=IDX, columns=["DOL", "HML_FX"])
    e = np.convolve(rng.normal(0, 0.01, 121), [1, 0.5], "valid")  # MA(1) errors
    y = 0.001 + 0.8 * f["DOL"] - 0.3 * f["HML_FX"] + e
    return y, f


def test_worked_example_exact_fit():
    _, f = _data()
    y = 0.002 + 0.5 * f["DOL"]
    r = at.ols_hac(y, f[["DOL"]])
    assert r["coef"]["alpha"] == pytest.approx(0.002, abs=1e-14)
    assert r["coef"]["DOL"] == pytest.approx(0.5, abs=1e-12)
    assert r["r2"] == pytest.approx(1.0)


@pytest.mark.parametrize("lag,small", [(0, False), (3, False), (6, True)])
def test_matches_statsmodels_hac(lag, small):
    sm = pytest.importorskip("statsmodels.api")
    y, f = _data(1)
    r = at.ols_hac(y, f, lag=lag, small_sample=small)
    ref = sm.OLS(y.to_numpy(), sm.add_constant(f.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": small})
    assert np.allclose(r["coef"].to_numpy(), ref.params, rtol=0, atol=1e-12)
    assert np.allclose(r["se"].to_numpy(), ref.bse, rtol=1e-10)
    assert r["r2"] == pytest.approx(ref.rsquared, rel=1e-12)


def test_default_lag_and_information_ratio():
    y, f = _data(2)
    r = at.ols_hac(y, f)
    assert r["lag"] == 4  # floor(4 * 1.2^(2/9)) = floor(4.165)
    e = r["resid"]
    ir = r["coef"]["alpha"] * 12 / (e.std(ddof=3) * np.sqrt(12))
    assert r["information_ratio"] == pytest.approx(ir)


def test_alignment_and_minimum_length():
    y, f = _data(3)
    r = at.ols_hac(y.iloc[10:], f.iloc[:-5])
    assert r["nobs"] == 105
    with pytest.raises(ValueError):
        at.ols_hac(y.iloc[:20], f)


def test_rolling_betas_recover_a_break():
    _, f = _data(4)
    y2 = 0.5 * f["DOL"].where(f.index < IDX[60], 1.5 * f["DOL"] / 0.5)
    rb = at.rolling_betas(y2, f[["DOL"]], window=36)
    assert rb["DOL"].iloc[0] == pytest.approx(0.5)
    assert rb["DOL"].iloc[-1] == pytest.approx(1.5)
    assert rb.index[0] == IDX[35]


def test_fx_factors_from_lrv_sort_and_named_sets():
    sig = pd.DataFrame(np.tile(np.arange(6.0), (30, 1)), index=IDX[:30], columns=list("ABCDEF"))
    nxt = sig / 100.0
    ports = panel.lrv_sort(sig, nxt, n_port=2)
    fx = at.fx_factors(ports)
    assert fx["HML_FX"].iloc[0] == pytest.approx(0.03)
    assert fx["DOL"].iloc[0] == pytest.approx(0.025)
    with pytest.raises(KeyError):
        at.select(fx, "ff3")
    y, f = _data(5)
    assert at.attribute(y, f, "fx")["factor_set"] == "fx"
