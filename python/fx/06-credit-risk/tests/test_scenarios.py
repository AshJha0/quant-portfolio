"""End-to-end scenarios and documented edge cases (contract item 6).

Covers: full scorecard pipeline out-of-time, planted 2020 contagion year
breaking calibration, country with no crisis history, country-holdout
generalisation, low-default CI width, rating distribution, and the
settlement/exposure edge cases wired through realistic flows.
"""

import numpy as np
import pytest

from fx_credit.cleaning import (
    assert_no_leaky_fields,
    clean_panel,
    country_holdout_split,
    drop_leaky_fields,
    time_split,
)
from fx_credit.data.synthetic import (
    SCORECARD_FEATURES,
    generate_fx_trade_book,
    generate_sovereign_panel,
)
from fx_credit.exposure import FXForward, cva_for_forward
from fx_credit.model import (
    RATING_ORDER,
    assign_rating,
    fit_logistic_irls,
    predict_pd,
)
from fx_credit.settlement import FXTrade, gross_settlement_exposure
from fx_credit.validation import auc, bootstrap_auc_ci, psi
from fx_credit.woe import monotone_merge, woe_table, woe_transform


def _fit_scorecard(train):
    tables = {
        f: monotone_merge(woe_table(train[f], train["default"], n_bins=5, feature=f))
        for f in SCORECARD_FEATURES
    }
    Xtr = np.column_stack([woe_transform(train[f], tables[f]) for f in SCORECARD_FEATURES])
    fit = fit_logistic_irls(Xtr, train["default"].to_numpy(dtype=float))
    return tables, fit


def _transform(df, tables):
    return np.column_stack(
        [woe_transform(df[f], tables[f]) for f in SCORECARD_FEATURES]
    )


@pytest.fixture(scope="module")
def pipeline():
    panel = clean_panel(generate_sovereign_panel(seed=42))
    safe = drop_leaky_fields(panel, extra=("true_pd",))
    assert_no_leaky_fields(safe)
    train, test = time_split(safe, 2014)
    tables, fit = _fit_scorecard(train)
    return panel, train, test, tables, fit


def test_pipeline_discriminates_in_and_out_of_time(pipeline):
    _, train, test, tables, fit = pipeline
    pd_tr = predict_pd(fit, _transform(train, tables))
    pd_te = predict_pd(fit, _transform(test, tables))
    assert auc(train["default"].to_numpy(float), pd_tr) > 0.75
    assert auc(test["default"].to_numpy(float), pd_te) > 0.65


def test_reserve_cover_woe_shows_threshold(pipeline):
    """The binned WOE for reserve cover must be riskier in the lowest bin than
    the highest — recovering the planted below-3-months threshold effect."""
    _, _, _, tables, _ = pipeline
    t = tables["reserves_import_cover"]
    nums = t.numeric_bins
    assert nums[0].woe > nums[-1].woe  # low cover = high risk
    assert nums[0].woe > 0 > nums[-1].woe


def test_planted_contagion_year_breaks_calibration(pipeline):
    """2020 (global contagion, planted): observed default rate far exceeds the
    scorecard's predicted mean PD — contagion breaks calibration out-of-time."""
    _, _, test, tables, fit = pipeline
    y2020 = test[test["year"] == 2020]
    pd_hat = predict_pd(fit, _transform(y2020, tables))
    observed = y2020["default"].mean()
    assert observed > 1.5 * pd_hat.mean()


def test_non_contagion_oot_years_reasonably_calibrated(pipeline):
    _, _, test, tables, fit = pipeline
    rest = test[test["year"] != 2020]
    pd_hat = predict_pd(fit, _transform(rest, tables))
    ratio = rest["default"].mean() / pd_hat.mean()
    assert 0.5 < ratio < 1.6


def test_score_psi_stable_outside_contagion(pipeline):
    _, train, test, tables, fit = pipeline
    pd_tr = predict_pd(fit, _transform(train, tables))
    rest = test[test["year"] != 2020]
    pd_rest = predict_pd(fit, _transform(rest, tables))
    assert psi(pd_tr, pd_rest) < 0.25


def test_country_with_no_crisis_history_scored(pipeline):
    """A country that never defaulted still gets a finite PD and a rating."""
    panel, train, test, tables, fit = pipeline
    totals = panel.groupby("country")["default"].sum()
    quiet = totals[totals == 0].index
    assert len(quiet) > 0  # such countries exist in the panel
    rows = test[test["country"].isin(quiet)]
    pd_hat = predict_pd(fit, _transform(rows, tables))
    assert np.all((pd_hat > 0) & (pd_hat < 1))
    for p in pd_hat[:10]:
        assert assign_rating(float(p)) in RATING_ORDER


def test_rating_distribution_non_degenerate(pipeline):
    _, train, _, tables, fit = pipeline
    pd_tr = predict_pd(fit, _transform(train, tables))
    letters = {assign_rating(float(p)) for p in pd_tr}
    assert len(letters) >= 4


def test_band_average_pd_monotone_on_panel(pipeline):
    _, train, _, tables, fit = pipeline
    pd_tr = predict_pd(fit, _transform(train, tables))
    bands = {}
    for p in pd_tr:
        bands.setdefault(assign_rating(float(p)), []).append(p)
    means = [np.mean(bands[r]) for r in RATING_ORDER if r in bands]
    assert np.all(np.diff(means) > 0)


def test_country_holdout_generalisation():
    panel = drop_leaky_fields(clean_panel(generate_sovereign_panel(seed=42)),
                              extra=("true_pd",))
    train, held = country_holdout_split(panel, holdout_frac=0.25, seed=0)
    tables, fit = _fit_scorecard(train)
    pd_h = predict_pd(fit, _transform(held, tables))
    # weaker bar than in-time: no shared country effects, few events
    assert auc(held["default"].to_numpy(float), pd_h) > 0.62


def test_low_default_bootstrap_ci_is_wide(pipeline):
    """Out-of-time window has few events: honest AUC CI is wide (>~5 Gini pts)."""
    _, _, test, tables, fit = pipeline
    pd_te = predict_pd(fit, _transform(test, tables))
    point, lo, hi = bootstrap_auc_ci(test["default"].to_numpy(float), pd_te,
                                     n_boot=300, seed=0)
    assert lo <= point <= hi
    assert hi - lo > 0.05


def test_zero_notional_trade_in_book_is_harmless():
    book = generate_fx_trade_book()
    rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 1 / 148.0}
    base = gross_settlement_exposure(book, rates)
    with_zero = book + [FXTrade("Z", "TokyoBank", "USDJPY", 0.0, 148.0, True)]
    assert gross_settlement_exposure(with_zero, rates) == pytest.approx(base)


def test_cva_edge_cases_matured_and_riskless():
    matured = FXForward("EURUSD", 10e6, 1.08, 0.0, True)
    v1, _ = cva_for_forward(matured, 1.08, 0.12, 0.03, 0.02, pd_1y=0.02, lgd=0.6,
                            n_steps=4, n_paths=500, seed=0)
    live = FXForward("EURUSD", 10e6, 1.08, 1.0, True)
    v2, _ = cva_for_forward(live, 1.08, 0.12, 0.03, 0.02, pd_1y=0.0, lgd=0.6,
                            n_steps=4, n_paths=500, seed=0)
    assert v1 == 0.0 and v2 == 0.0


def test_live_loader_is_network_guarded(monkeypatch):
    from fx_credit.data.live import fetch_worldbank_indicators

    monkeypatch.delenv("FX_CREDIT_ALLOW_NETWORK", raising=False)
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        fetch_worldbank_indicators(["FI.RES.TOTL.MO"], ["ARG"], 2000, 2020)


def test_live_csv_schema_validation(tmp_path):
    from fx_credit.data.live import load_worldbank_csv

    good = tmp_path / "wb.csv"
    good.write_text("country,year,indicator,value\nARG,2001,FI.RES.TOTL.MO,5.2\n")
    df = load_worldbank_csv(str(good))
    assert list(df.columns) == ["country", "year", "indicator", "value"]

    bad = tmp_path / "bad.csv"
    bad.write_text("nation,yr\nARG,2001\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_worldbank_csv(str(bad))
