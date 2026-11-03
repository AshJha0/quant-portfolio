"""Hosmer-Lemeshow size by simulation, and per-grade PD back-tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from eq_credit import fit_logistic, hosmer_lemeshow, pd_backtest_by_grade
from eq_credit.validation import roc_auc


def test_hl_out_of_sample_size_is_nominal():
    # Known PDs, no fitting: H0 holds exactly, so a 5% test should reject
    # 5% of the time. 1000 trials: MC SE 0.0069, band [0.03, 0.07] is ~3 SE.
    # With the old g - 2 = 8 df the rejection rate was about 11%.
    rng = np.random.default_rng(101)
    rej = 0
    for _ in range(1000):
        p = rng.uniform(0.01, 0.3, 2000)
        y = (rng.uniform(size=2000) < p).astype(float)
        rej += hosmer_lemeshow(y, p, out_of_sample=True)[1] < 0.05
    assert 0.03 <= rej / 1000 <= 0.07


def test_hl_in_sample_size_with_fitted_model():
    # PDs from a logistic model fitted on the same rows: reference g - 2.
    # 300 trials: MC SE 0.0126; [0.015, 0.09] allows ~3 SE plus the known
    # small-sample approximation error of the g - 2 rule.
    rng = np.random.default_rng(202)
    rej = 0
    for _ in range(300):
        x = rng.normal(size=(2000, 2))
        eta = -2.0 + x @ np.array([0.8, -0.5])
        y = (rng.uniform(size=2000) < 1 / (1 + np.exp(-eta))).astype(float)
        fit = fit_logistic(x, y)
        p = fit.predict_proba(x)
        rej += hosmer_lemeshow(y, p, out_of_sample=False)[1] < 0.05
    assert 0.015 <= rej / 300 <= 0.09


def test_hl_mode_is_required_since_3_0():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.2, 1000)
    y = (rng.uniform(size=1000) < p).astype(float)
    with pytest.raises(TypeError, match="out_of_sample"):
        hosmer_lemeshow(y, p)  # type: ignore[call-arg]
    pv = hosmer_lemeshow(y, p, out_of_sample=False)[1]
    assert hosmer_lemeshow(y, p, out_of_sample=True)[1] > pv  # more df, larger p


def test_by_grade_binomial_and_jeffreys_hand_values():
    # grade A: N = 100, D = 6, PD 2%; grade B: N = 200, D = 0, PD 1%
    y = np.r_[np.ones(6), np.zeros(94), np.zeros(200)]
    pd_hat = np.r_[np.full(100, 0.02), np.full(200, 0.01)]
    grades = np.r_[["A"] * 100, ["B"] * 200]
    t = pd_backtest_by_grade(y, pd_hat, grades).set_index("grade")
    # binomial upper tail by direct summation
    p_bin = sum(math.comb(100, k) * 0.02**k * 0.98 ** (100 - k) for k in range(6, 101))
    assert t.loc["A", "binomial_p"] == pytest.approx(p_bin, rel=1e-10)
    assert t.loc["B", "binomial_p"] == 1.0
    # Jeffreys: Beta(6.5, 94.5) cdf at 0.02 by trapezoid quadrature of the pdf
    a, b = 6.5, 94.5
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    xs = np.linspace(0.0, 0.02, 200_001)[1:]
    pdf = np.exp((a - 1) * np.log(xs) + (b - 1) * np.log1p(-xs) - lb)
    q = float(np.sum((pdf[1:] + pdf[:-1]) / 2) * (xs[1] - xs[0]))
    assert t.loc["A", "jeffreys_p"] == pytest.approx(q, rel=1e-6)
    assert t.loc["A", "light"] == "amber"
    assert 0.001 <= p_bin < 0.05
    assert t.loc["B", "light"] == "green"


def test_by_grade_red_light_and_validation():
    y = np.r_[np.ones(10), np.zeros(90)]
    t = pd_backtest_by_grade(y, np.full(100, 0.01), np.zeros(100))
    assert t.loc[0, "light"] == "red"
    with pytest.raises(ValueError):
        pd_backtest_by_grade(y, np.full(99, 0.01), np.zeros(100))
    with pytest.raises(ValueError):
        pd_backtest_by_grade(y, np.full(100, np.nan), np.zeros(100))


def test_auc_does_not_need_numpy2(monkeypatch):
    # numpy 1.x has no np.trapezoid; the module falls back to np.trapz
    import eq_credit.validation as v

    monkeypatch.setattr(v, "_trapezoid", lambda yv, xv: float(np.sum((yv[1:] + yv[:-1]) / 2 * np.diff(xv))))
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    assert roc_auc(y, s) == pytest.approx(8 / 9, abs=1e-15)
