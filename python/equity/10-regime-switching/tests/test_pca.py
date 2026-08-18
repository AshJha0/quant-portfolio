"""PCA-from-scratch tests, including the sklearn cross-check to 1e-8."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA as SkPCA

from eq_regime.pca import fit_pca, project, rolling_pca, scree_table


@pytest.fixture(scope="module")
def x_random():
    rng = np.random.default_rng(0)
    return rng.standard_normal((300, 5)) @ rng.standard_normal((5, 5))


def test_components_orthonormal(x_random):
    m = fit_pca(x_random)
    v = m.components
    np.testing.assert_allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-12)


def test_explained_variance_ratio_sums_to_one(x_random):
    m = fit_pca(x_random)  # all components
    assert m.explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-12)
    # eigenvalues of a correlation matrix sum to n_features
    assert m.explained_variance.sum() == pytest.approx(x_random.shape[1], abs=1e-10)


def test_matches_sklearn_to_1e8_up_to_sign(x_random):
    """Correlation-matrix PCA == sklearn covariance PCA on standardized data."""
    m = fit_pca(x_random)
    z = (x_random - x_random.mean(0)) / x_random.std(0, ddof=1)
    sk = SkPCA(n_components=x_random.shape[1]).fit(z)
    np.testing.assert_allclose(m.explained_variance, sk.explained_variance_, atol=1e-8)
    for j in range(x_random.shape[1]):
        ours, theirs = m.components[:, j], sk.components_[j]
        err = min(np.abs(ours - theirs).max(), np.abs(ours + theirs).max())
        assert err < 1e-8


def test_projection_matches_sklearn_scores(x_random):
    m = fit_pca(x_random, 3)
    z = (x_random - x_random.mean(0)) / x_random.std(0, ddof=1)
    sk = SkPCA(n_components=3).fit(z)
    ours = project(x_random, m)
    theirs = sk.transform(z)
    for j in range(3):
        err = min(np.abs(ours[:, j] - theirs[:, j]).max(), np.abs(ours[:, j] + theirs[:, j]).max())
        assert err < 1e-8


def test_first_pc_captures_correlated_block():
    """3 strongly correlated columns + 2 independent: PC1 loads on the block
    with a COMMON sign, and the largest loading is positive (sign convention)."""
    rng = np.random.default_rng(1)
    f = rng.standard_normal(500)
    block = np.column_stack([f + 0.1 * rng.standard_normal(500) for _ in range(3)])
    noise = rng.standard_normal((500, 2))
    x = np.hstack([block, noise])
    m = fit_pca(x, 2)
    pc1 = m.components[:, 0]
    assert np.all(pc1[:3] > 0.4)          # block loadings large, same sign
    assert np.abs(pc1[3:]).max() < 0.25   # noise loadings small
    assert m.explained_variance_ratio[0] > 0.5


def test_sign_convention_largest_loading_positive(x_random):
    m = fit_pca(x_random)
    for j in range(m.n_components):
        i = np.argmax(np.abs(m.components[:, j]))
        assert m.components[i, j] > 0


def test_scree_table(x_random):
    m = fit_pca(x_random)
    t = scree_table(m)
    assert t["cum_ratio"].iloc[-1] == pytest.approx(1.0, abs=1e-12)
    assert (t["var_ratio"].diff().dropna() <= 1e-12).all()  # descending


def test_dataframe_roundtrip(x_random):
    df = pd.DataFrame(x_random, columns=list("abcde"))
    m = fit_pca(df, 2)
    s = project(df, m)
    assert list(s.columns) == ["PC1", "PC2"]
    assert m.feature_names == ("a", "b", "c", "d", "e")


def test_rolling_pca_sign_continuity():
    """Rolling PC1 scores stay positively aligned with the full-sample PC1."""
    rng = np.random.default_rng(4)
    f = rng.standard_normal(600)
    x = np.column_stack([f + 0.3 * rng.standard_normal(600) for _ in range(4)])
    df = pd.DataFrame(x, index=pd.bdate_range("2020-01-01", periods=600))
    scores = rolling_pca(df, window=120, n_components=1, step=20)
    full = project(df.loc[scores.index], fit_pca(df, 1))
    corr = np.corrcoef(scores["PC1"], full["PC1"])[0, 1]
    assert corr > 0.95  # no sign flips across refits

    # block-to-block continuity: consecutive blocks never anti-correlate
    diffs = scores["PC1"].diff().abs()
    assert diffs.max() < 10 * scores["PC1"].std()


def test_validation_errors(x_random):
    with pytest.raises(ValueError, match="more observations"):
        fit_pca(x_random[:4])
    bad = x_random.copy()
    bad[:, 2] = 3.14
    with pytest.raises(ValueError, match="constant"):
        fit_pca(bad)
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(x_random, 99)
    nanx = x_random.copy()
    nanx[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_pca(nanx)
    m = fit_pca(x_random, 2)
    with pytest.raises(ValueError, match="mismatch"):
        project(x_random[:, :3], m)
    with pytest.raises(ValueError, match="window"):
        rolling_pca(pd.DataFrame(x_random), window=3, n_components=1)
