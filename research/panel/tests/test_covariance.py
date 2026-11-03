"""Tests for research/lib/covariance.py (W23). Oracles: sklearn's Ledoit-Wolf
(2004), the Marchenko-Pastur spread under a spherical truth, and the
Frobenius-loss ordering reported by Ledoit and Wolf (2020, tables 1-3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "lib")]

import covariance as cv  # noqa: E402


def _draw(true: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(len(true)), true, size=n)


def test_linear_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    x = _draw(np.diag(np.linspace(1, 5, 20)), 60, 1)
    ours, d = cv.linear_shrinkage(x)
    ref, dref = sk.ledoit_wolf(x)
    assert d == pytest.approx(dref, rel=1e-10)
    assert np.allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_worked_example_spherical_truth_is_pulled_back_to_one():
    x = _draw(np.eye(100), 201, 2)  # n = 200 after demeaning
    lam = np.linalg.eigvalsh(cv.sample_cov(x))
    assert lam.min() < 0.2 and lam.max() > 2.5  # Marchenko-Pastur: 0.09 .. 2.91
    d = np.linalg.eigvalsh(cv.nonlinear_shrinkage(x))
    assert np.abs(d - 1).max() < 0.35  # the sample's worst error is above 1.5
    assert np.abs(d - 1).max() < 0.25 * np.abs(lam - 1).max()


@pytest.mark.parametrize("spec", [np.linspace(1, 10, 30), np.r_[np.ones(25), np.full(25, 10.0)],
                                  np.linspace(1, 10, 200)])
def test_tracks_the_finite_sample_oracle(spec):
    """Ledoit-Wolf (2020) target: d*_i = u_i' Sigma u_i, the best eigenvalue
    for each sample eigenvector. Nonlinear must be far nearer than the sample
    eigenvalues and nearer than linear shrinkage."""
    true = np.diag(spec)
    x = _draw(true, 4 * len(spec) + 1, 7)
    lam, u = np.linalg.eigh(cv.sample_cov(x))
    star = np.einsum("ij,ik,kj->j", u, true, u)
    def rms(est):
        return float(np.sqrt(np.mean((np.diag(u.T @ est @ u) - star) ** 2)))
    nl, lin = rms(cv.nonlinear_shrinkage(x)), rms(cv.linear_shrinkage(x)[0])
    assert nl < 0.35 * float(np.sqrt(np.mean((lam - star) ** 2)))
    assert nl < lin


def test_keeps_sample_eigenvectors_and_order():
    x = _draw(np.diag(np.linspace(1, 10, 30)), 120, 3)
    s = cv.sample_cov(x)
    out = cv.nonlinear_shrinkage(x)
    lam, u = np.linalg.eigh(s)
    d = np.diag(u.T @ out @ u)
    off = u.T @ out @ u - np.diag(d)
    assert np.abs(off).max() < 1e-10
    # not forced monotone: the kernel estimate can reorder close neighbours
    assert d[0] > lam[0] and d[-1] < lam[-1]  # extremes pulled inwards
    assert d.mean() == pytest.approx(lam.mean(), rel=0.05)


def test_converges_to_sample_when_n_is_large():
    x = _draw(np.diag(np.linspace(1, 4, 5)), 20001, 4)
    s, out = cv.sample_cov(x), cv.nonlinear_shrinkage(x)
    assert np.abs(out - s).max() / np.abs(s).max() < 0.01


def test_loss_ordering_on_dispersed_spectrum():
    true = np.diag(np.r_[np.ones(25), np.full(25, 10.0)])
    loss = {"sample": 0.0, "linear": 0.0, "nonlinear": 0.0}
    for seed in range(10):
        x = _draw(true, 101, 100 + seed)
        loss["sample"] += cv.frobenius_loss(cv.sample_cov(x), true)
        loss["linear"] += cv.frobenius_loss(cv.linear_shrinkage(x)[0], true)
        loss["nonlinear"] += cv.frobenius_loss(cv.nonlinear_shrinkage(x), true)
    assert loss["nonlinear"] < loss["linear"] < loss["sample"]


def test_more_variables_than_observations_is_positive_definite():
    x = _draw(np.eye(80), 41, 5)  # p = 80 > n = 40
    out = cv.nonlinear_shrinkage(x)
    d = np.linalg.eigvalsh(out)
    assert d.min() > 0
    assert np.median(np.abs(d - 1)) < 0.35


def test_rejects_nan_and_tiny_samples():
    with pytest.raises(ValueError):
        cv.nonlinear_shrinkage(np.full((20, 3), np.nan))
    with pytest.raises(ValueError):
        cv.nonlinear_shrinkage(np.ones((5, 3)))


def test_pca_diagonal_and_rank():
    rng = np.random.default_rng(6)
    beta = rng.normal(size=(40, 1))
    x = rng.normal(size=(30, 1)) @ beta.T + 0.5 * rng.normal(size=(30, 40))
    s = cv.sample_cov(x)
    p1 = cv.pca_covariance(x, 1)
    assert np.allclose(np.diag(p1), np.diag(s))
    assert np.allclose(cv.pca_covariance(x, 40), s)
    lam, v = np.linalg.eigh(s)
    top = lam[-1] * np.outer(v[:, -1], v[:, -1])
    off = ~np.eye(40, dtype=bool)
    assert np.allclose(p1[off], top[off])  # off-diagonal is exactly rank one
    true = beta @ beta.T + 0.25 * np.eye(40)
    # off-diagonal noise dominates at n = 30 < p = 40: the factor model wins
    assert cv.frobenius_loss(p1, true) < cv.frobenius_loss(s, true)
    with pytest.raises(ValueError):
        cv.pca_covariance(x, 41)
