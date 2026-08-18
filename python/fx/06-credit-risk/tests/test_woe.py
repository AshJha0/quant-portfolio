"""WOE/IV: exact hand-computed tables, missing bin, monotone merge, leakage flag."""

import numpy as np
import pandas as pd
import pytest

from fx_credit.data.synthetic import generate_sovereign_panel
from fx_credit.woe import (
    flag_leaky_iv,
    iv_report,
    monotone_merge,
    woe_table,
    woe_transform,
)

LN2 = np.log(2.0)


def _hand_case():
    """Two bins, no smoothing. bad_total = good_total = 3.

    bin1 (x=1): 1 bad, 2 good -> WOE = ln((1/3)/(2/3)) = -ln2,
                IV = (1/3 - 2/3)(-ln2) = ln2/3
    bin2 (x=2): 2 bad, 1 good -> WOE = +ln2, IV = ln2/3
    Total IV = 2 ln2 / 3.
    """
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    edges = np.array([0.5, 1.5, 2.5])
    return x, y, edges


def test_woe_hand_computed_exact():
    x, y, edges = _hand_case()
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    woes = [b.woe for b in t.bins]
    assert woes == pytest.approx([-LN2, LN2], abs=1e-14)


def test_iv_hand_computed_exact():
    x, y, edges = _hand_case()
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    assert t.iv == pytest.approx(2.0 * LN2 / 3.0, abs=1e-14)


def test_bin_counts_hand_case():
    x, y, edges = _hand_case()
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    assert [(b.n, b.n_bad, b.n_good) for b in t.bins] == [(3, 1, 2), (3, 2, 1)]
    assert t.bad_total == 3 and t.good_total == 3


def test_missing_bin_created_and_exact():
    x, y, edges = _hand_case()
    x = np.r_[x, np.nan, np.nan]
    y = np.r_[y, 1.0, 0.0]  # missing bin: 1 bad, 1 good of totals 4 bad, 4 good
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    mb = t.missing_bin
    assert mb is not None and mb.is_missing
    assert (mb.n, mb.n_bad, mb.n_good) == (2, 1, 1)
    assert mb.woe == pytest.approx(np.log((1 / 4) / (1 / 4)), abs=1e-14)  # 0


def test_zero_cell_raises_without_smoothing():
    x = np.array([1.0, 1.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])  # bin1 has zero bads
    with pytest.raises(ValueError, match="smoothing"):
        woe_table(x, y, edges=np.array([0.5, 1.5, 2.5]), smoothing=0.0)


def test_smoothing_keeps_zero_cell_finite():
    x = np.array([1.0, 1.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])
    t = woe_table(x, y, edges=np.array([0.5, 1.5, 2.5]), smoothing=0.5)
    assert all(np.isfinite(b.woe) for b in t.bins)


def test_constant_feature_zero_iv():
    x = np.full(100, 7.0)
    y = np.r_[np.ones(10), np.zeros(90)]
    t = woe_table(x, y, n_bins=5, smoothing=0.0)
    assert len(t.numeric_bins) == 1
    assert t.iv == pytest.approx(0.0, abs=1e-12)


def test_nonbinary_y_raises():
    with pytest.raises(ValueError, match="binary"):
        woe_table(np.array([1.0, 2.0]), np.array([0.0, 2.0]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        woe_table(np.array([1.0, 2.0]), np.array([0.0]))


def test_all_missing_raises():
    with pytest.raises(ValueError, match="missing"):
        woe_table(np.array([np.nan, np.nan]), np.array([0.0, 1.0]))


def test_quantile_bins_respect_n_bins():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = (rng.random(2000) < 0.2).astype(float)
    t = woe_table(x, y, n_bins=5)
    assert len(t.numeric_bins) <= 5


def _nonmonotone_table():
    """Bad rates by bin: 10%, 30%, 20%, 50% -> WOE non-monotone at bin 3."""
    xs, ys = [], []
    for i, (n, nb) in enumerate([(100, 10), (100, 30), (100, 20), (100, 50)]):
        xs += [float(i)] * n
        ys += [1.0] * nb + [0.0] * (n - nb)
    edges = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    return woe_table(np.array(xs), np.array(ys), edges=edges, smoothing=0.0)


def test_monotone_merge_produces_monotone():
    t = monotone_merge(_nonmonotone_table())
    woes = [b.woe for b in t.numeric_bins]
    diffs = np.diff(woes)
    assert np.all(diffs >= 0) or np.all(diffs <= 0)
    assert len(t.numeric_bins) < 4  # something was merged


def test_monotone_merge_preserves_totals():
    t0 = _nonmonotone_table()
    t1 = monotone_merge(t0)
    assert sum(b.n for b in t1.bins) == sum(b.n for b in t0.bins)
    assert sum(b.n_bad for b in t1.bins) == sum(b.n_bad for b in t0.bins)


def test_monotone_merge_merged_counts_exact():
    t = monotone_merge(_nonmonotone_table())
    # bins 2 (30/100) and 3 (20/100) pool into 50/200
    pooled = [b for b in t.numeric_bins if b.n == 200]
    assert len(pooled) == 1 and pooled[0].n_bad == 50


def test_monotone_merge_keeps_missing_bin():
    x = np.r_[np.repeat([0.0, 1.0, 2.0, 3.0], 50), [np.nan] * 10]
    rng = np.random.default_rng(1)
    y = (rng.random(210) < np.r_[np.repeat([0.05, 0.3, 0.2, 0.5], 50), [0.3] * 10]).astype(float)
    t = monotone_merge(woe_table(x, y, edges=np.array([-0.5, 0.5, 1.5, 2.5, 3.5])))
    assert t.missing_bin is not None


def test_monotone_merge_noop_when_monotone():
    xs, ys = [], []
    for i, (n, nb) in enumerate([(100, 5), (100, 20), (100, 40)]):
        xs += [float(i)] * n
        ys += [1.0] * nb + [0.0] * (n - nb)
    t0 = woe_table(np.array(xs), np.array(ys), edges=np.array([-0.5, 0.5, 1.5, 2.5]), smoothing=0.0)
    t1 = monotone_merge(t0)
    assert len(t1.bins) == len(t0.bins)


def test_woe_transform_hand_case():
    x, y, edges = _hand_case()
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    out = woe_transform(np.array([1.0, 2.0]), t)
    assert out == pytest.approx([-LN2, LN2], abs=1e-14)


def test_woe_transform_missing_to_missing_bin():
    x, y, edges = _hand_case()
    x = np.r_[x, np.nan, np.nan]
    y = np.r_[y, 1.0, 0.0]
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    out = woe_transform(np.array([np.nan]), t)
    assert out[0] == pytest.approx(t.missing_bin.woe, abs=1e-14)


def test_woe_transform_missing_without_missing_bin_is_neutral():
    x, y, edges = _hand_case()
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    assert woe_transform(np.array([np.nan]), t)[0] == 0.0


def test_woe_transform_clamps_out_of_range():
    x, y, edges = _hand_case()
    t = woe_table(x, y, edges=edges, smoothing=0.0)
    out = woe_transform(np.array([-100.0, 100.0]), t)
    assert out == pytest.approx([-LN2, LN2], abs=1e-14)


def test_leaky_feature_iv_flagged_on_panel():
    df = generate_sovereign_panel(seed=42)
    rep = iv_report(
        df,
        ["reserves_import_cover", "st_debt_reserves", "devaluation_next_year_pct"],
    )
    flagged = set(rep.loc[rep["leaky_flag"], "feature"])
    assert flagged == {"devaluation_next_year_pct"}
    ivs = dict(zip(rep["feature"], rep["iv"]))
    assert ivs["devaluation_next_year_pct"] > 2.0  # absurd IV = leakage signature


def test_iv_report_sorted_descending():
    df = generate_sovereign_panel(seed=42)
    rep = iv_report(df, ["reserves_import_cover", "ca_gdp", "st_debt_reserves"])
    assert (rep["iv"].diff().dropna() <= 1e-12).all()


def test_flag_leaky_iv_threshold():
    ivs = {"good": 0.25, "strong": 0.8, "leaky": 1.7}
    assert flag_leaky_iv(ivs) == ["leaky"]
    assert flag_leaky_iv(ivs, threshold=0.5) == ["leaky", "strong"]
