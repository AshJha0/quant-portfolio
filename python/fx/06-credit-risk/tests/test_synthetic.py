"""Sovereign panel generator properties, trade book and counterparty set."""

import numpy as np
import pandas as pd
import pytest

from fx_credit.data.synthetic import (
    CRISIS_YEARS,
    FEATURES,
    LEAKY_FIELDS,
    RESERVE_COVER_THRESHOLD,
    generate_counterparty_set,
    generate_fx_trade_book,
    generate_logistic_data,
    generate_sovereign_panel,
)
from fx_credit.settlement import FXTrade


@pytest.fixture(scope="module")
def panel():
    return generate_sovereign_panel(seed=42)


def test_panel_has_all_columns(panel):
    for col in ("country", "region", "year", "default", "true_pd", *FEATURES, *LEAKY_FIELDS):
        assert col in panel.columns, col


def test_country_year_unique(panel):
    assert not panel.duplicated(subset=["country", "year"]).any()


def test_reproducible_same_seed(panel):
    other = generate_sovereign_panel(seed=42)
    pd.testing.assert_frame_equal(panel, other)


def test_different_seed_differs(panel):
    other = generate_sovereign_panel(seed=43)
    assert not panel["default"].equals(other["default"])


def test_base_rate_in_target_range(panel):
    rate = panel["default"].mean()
    assert 0.03 <= rate <= 0.06


def test_default_rate_matches_true_pd(panel):
    # outcome is Bernoulli(true_pd): realised rate close to mean true PD
    assert abs(panel["default"].mean() - panel["true_pd"].mean()) < 0.01


def test_true_pd_valid_probabilities(panel):
    assert ((panel["true_pd"] > 0) & (panel["true_pd"] < 1)).all()


def test_contagion_years_raise_default_rate(panel):
    by = panel.groupby("contagion")["default"].mean()
    assert by[1.0] > 2.0 * by[0.0]


def test_contagion_flag_matches_crisis_years(panel):
    for region, years in CRISIS_YEARS.items():
        sub = panel[panel["region"] == region]
        assert (sub.loc[sub["year"].isin(years), "contagion"] == 1.0).all()
        assert (sub.loc[~sub["year"].isin(years), "contagion"] == 0.0).all()


def test_reserve_cover_threshold_effect(panel):
    low = panel[panel["reserves_import_cover"] < RESERVE_COVER_THRESHOLD]["default"].mean()
    high = panel[panel["reserves_import_cover"] >= RESERVE_COVER_THRESHOLD]["default"].mean()
    assert low > 1.5 * high


def test_leaky_fields_track_outcome(panel):
    d1 = panel[panel["default"] == 1]
    d0 = panel[panel["default"] == 0]
    assert d1["imf_program_next_year"].mean() > 0.5 > d0["imf_program_next_year"].mean()
    assert d1["devaluation_next_year_pct"].mean() > 5 * d0["devaluation_next_year_pct"].mean()


def test_political_stability_has_missing(panel):
    frac = panel["political_stability"].isna().mean()
    assert 0.03 < frac < 0.12


def test_within_country_persistence(panel):
    acs = []
    for _, g in panel.groupby("country"):
        v = g.sort_values("year")["reserves_import_cover"].to_numpy()
        acs.append(np.corrcoef(v[:-1], v[1:])[0, 1])
    assert np.mean(acs) > 0.3  # panel rows are serially dependent


def test_invalid_year_range_raises():
    with pytest.raises(ValueError, match="end_year"):
        generate_sovereign_panel(start_year=2020, end_year=2020)


def test_logistic_data_shapes_and_rate():
    X, y = generate_logistic_data(np.array([0.5, -0.5]), -2.0, n=5000, seed=1)
    assert X.shape == (5000, 2) and y.shape == (5000,)
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert 0.05 < y.mean() < 0.30


def test_trade_book_composition():
    book = generate_fx_trade_book()
    assert len(book) == 6
    assert all(isinstance(t, FXTrade) for t in book)
    assert any(t.cls_settled for t in book) and any(not t.cls_settled for t in book)
    ccys = {t.base for t in book} | {t.quote for t in book}
    assert ccys == {"USD", "JPY", "EUR", "GBP"}


def test_counterparty_set_valid():
    cps = generate_counterparty_set()
    assert set(["counterparty", "rating", "pd_1y", "lgd", "cls_member"]) <= set(cps.columns)
    assert ((cps["pd_1y"] > 0) & (cps["pd_1y"] < 1)).all()
    assert ((cps["lgd"] >= 0) & (cps["lgd"] <= 1)).all()
    assert cps["counterparty"].is_unique
