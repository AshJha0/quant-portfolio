"""PCA tests: identities, sklearn cross-check, RORO loading structure."""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SkPCA

from fx_regime import (
    EM,
    G10_CARRY,
    HAVENS,
    build_features,
    fit_pca,
    generate_roro_panel,
    roro_axis_check,
)


def _data(n=200, p=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    return rng.standard_normal((n, p)) @ A


def test_components_orthonormal():
    res = fit_pca(_data(), 6)
    G = res.components @ res.components.T
    assert np.allclose(G, np.eye(6), atol=1e-10)


def test_full_reconstruction_identity():
    X = _data()
    res = fit_pca(X)
    Xr = res.inverse_transform(res.scores)
    assert np.allclose(Xr, X, atol=1e-8)


def test_explained_variance_sums_to_total():
    X = _data()
    res = fit_pca(X)
    total = X.var(axis=0, ddof=1).sum()
    assert np.isclose(res.explained_variance.sum(), total, atol=1e-8)
    assert np.isclose(res.explained_variance_ratio.sum(), 1.0, atol=1e-10)


def test_scores_uncorrelated_with_correct_variance():
    X = _data()
    res = fit_pca(X, 4)
    cov = np.cov(res.scores.T, ddof=1)
    assert np.allclose(cov, np.diag(res.explained_variance), atol=1e-8)


def test_evr_sorted_descending():
    res = fit_pca(_data(), 6)
    assert (np.diff(res.explained_variance) <= 1e-12).all()


def test_sklearn_cross_check():
    X = _data(300, 8, seed=1)
    ours = fit_pca(X, 5)
    sk = SkPCA(n_components=5).fit(X)
    assert np.allclose(
        ours.explained_variance, sk.explained_variance_, atol=1e-8
    )
    assert np.allclose(
        ours.explained_variance_ratio, sk.explained_variance_ratio_, atol=1e-8
    )
    for i in range(5):  # match up to sign
        dot = abs(ours.components[i] @ sk.components_[i])
        assert np.isclose(dot, 1.0, atol=1e-8)


def test_sign_convention_deterministic():
    X = _data(150, 5, seed=2)
    a, b = fit_pca(X, 3), fit_pca(X.copy(), 3)
    assert np.allclose(a.components, b.components)
    for i in range(3):
        j = np.argmax(np.abs(a.components[i]))
        assert a.components[i, j] > 0


def test_transform_matches_scores():
    X = _data()
    res = fit_pca(X, 3)
    assert np.allclose(res.transform(X), res.scores, atol=1e-10)


def test_roro_loading_sign_structure(panel2):
    """PC1 of the RORO panel IS the risk-on/risk-off axis."""
    X = panel2.returns.to_numpy()
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    res = fit_pca(Xs, 3)
    cols = list(panel2.returns.columns)
    assert roro_axis_check(
        res.components, cols, list(G10_CARRY) + list(EM), list(HAVENS)
    )
    assert res.explained_variance_ratio[0] > 0.3


def test_roro_axis_on_feature_block(panel2, feats2):
    """In feature space, PC1 is the risk DIRECTION axis (carry-return
    features load opposite to haven strength) and one of the leading PCs
    (the vol/correlation factor) cleanly separates the true states."""
    import pandas as pd

    sub = feats2[["avg_vol", "carry_ret", "haven_rs", "usd_corr", "em_g10"]]
    res = fit_pca(sub.to_numpy(), 3)
    pc1 = dict(zip(sub.columns, res.components[0]))
    s = np.sign(pc1["carry_ret"])
    assert np.sign(pc1["em_g10"]) == s
    assert np.sign(pc1["haven_rs"]) == -s
    # among the top-3 PCs there is a regime axis with a large gap
    # between true risk_on and risk_off scores
    states = pd.Series(panel2.states, index=panel2.returns.index).reindex(
        sub.index
    )
    gaps = []
    for i in range(3):
        z = res.scores[:, i]
        gaps.append(
            abs(z[states == 1].mean() - z[states == 0].mean()) / z.std(ddof=0)
        )
    assert max(gaps) > 1.0


def test_rank_deficient_pegged_column():
    X = _data(100, 4, seed=3)
    X = np.hstack([X, np.zeros((100, 1))])  # pegged: zero variance
    res = fit_pca(X)
    assert np.isfinite(res.components).all()
    assert res.explained_variance[-1] < 1e-20


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        fit_pca(np.ones((1, 3)))
    with pytest.raises(ValueError):
        fit_pca(_data(), n_components=0)
    with pytest.raises(ValueError):
        fit_pca(_data(), n_components=99)
    with pytest.raises(ValueError):
        fit_pca(np.ones(5))
