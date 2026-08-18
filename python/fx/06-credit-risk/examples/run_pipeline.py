"""End-to-end FX / cross-border credit risk pipeline.

Reproduces every number quoted in README.md and docs/:

  1. Sovereign PD scorecard: panel -> cleaning & leakage guards -> WOE/IV ->
     IRLS logistic (sklearn cross-checked) -> scorecard & ratings ->
     validation train vs out-of-time incl. the planted 2020 contagion year.
  2. FX settlement (Herstatt) risk: 6-trade book, gross vs CLS exposure.
  3. Pre-settlement risk: single-forward EE/PFE profile, netting-set
     comparison, CVA for a BB-rated counterparty.
  4. Capital table by rating band (EL, Basel standardized, Vasicek 99.9%).

Runs offline in well under 120 s.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # cross-check only

from fx_credit.capital import capital_table
from fx_credit.cleaning import (
    assert_no_leaky_fields,
    clean_panel,
    drop_leaky_fields,
    time_split,
)
from fx_credit.data.synthetic import (
    SCORECARD_FEATURES,
    generate_counterparty_set,
    generate_fx_trade_book,
    generate_sovereign_panel,
)
from fx_credit.exposure import FXForward, cva_for_forward, netting_set_profile
from fx_credit.model import (
    RATING_BANDS,
    RATING_ORDER,
    assign_rating,
    fit_logistic_irls,
    predict_pd,
    rating_midpoint_pd,
    score_from_pd,
)
from fx_credit.settlement import (
    book_settlement_report,
    gross_settlement_exposure,
    net_settlement_exposure,
    time_zone_gap_matrix,
)
from fx_credit.validation import (
    auc,
    bootstrap_auc_ci,
    gini,
    hosmer_lemeshow,
    ks_statistic,
    psi,
    within_country_autocorrelation,
)
from fx_credit.woe import iv_report, monotone_merge, woe_table, woe_transform

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Setting penalty=None")

    # ------------------------------------------------------------------ 1
    hr("1. SOVEREIGN PD SCORECARD")
    panel = clean_panel(generate_sovereign_panel(seed=42))
    print(f"Panel: {panel['country'].nunique()} countries x "
          f"{panel['year'].nunique()} years = {len(panel)} rows, "
          f"base default rate {panel['default'].mean():.2%}")

    print("\n-- Leakage screen (IV on raw frame; >1.0 flags post-outcome fields) --")
    screen = iv_report(panel, [*SCORECARD_FEATURES,
                               "imf_program_next_year", "devaluation_next_year_pct"])
    print(screen.to_string(index=False))

    safe = drop_leaky_fields(panel, extra=("true_pd", "contagion"))
    assert_no_leaky_fields(safe)
    train, test = time_split(safe, train_end_year=2014)
    print(f"\nTime split: train <=2014 ({len(train)} rows, "
          f"{int(train['default'].sum())} events), OOT 2015-2023 "
          f"({len(test)} rows, {int(test['default'].sum())} events)")

    tables = {
        f: monotone_merge(woe_table(train[f], train["default"], n_bins=5, feature=f))
        for f in SCORECARD_FEATURES
    }
    print("\n-- Reserve-cover binning (note the below-3-months threshold effect) --")
    print(tables["reserves_import_cover"].to_frame().to_string(index=False))

    def transform(df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [woe_transform(df[f], tables[f]) for f in SCORECARD_FEATURES]
        )

    Xtr, ytr = transform(train), train["default"].to_numpy(float)
    Xte, yte = transform(test), test["default"].to_numpy(float)
    fit = fit_logistic_irls(Xtr, ytr)
    sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(Xtr, ytr)
    xchk = max(np.max(np.abs(fit.coef - sk.coef_.ravel())),
               abs(fit.intercept - sk.intercept_[0]))

    print("\n-- IRLS coefficients (WOE space) --")
    coef_tab = pd.DataFrame({
        "feature": ["intercept", *SCORECARD_FEATURES],
        "coef": [fit.intercept, *fit.coef],
        "se": fit.se,
        "z": np.r_[fit.intercept, fit.coef] / fit.se,
    })
    print(coef_tab.to_string(index=False))
    print(f"\nsklearn cross-check: max |coef diff| = {xchk:.2e} "
          f"(converged in {fit.n_iter} IRLS iterations)")

    pd_tr, pd_te = predict_pd(fit, Xtr), predict_pd(fit, Xte)
    scores = score_from_pd(pd_te)
    ratings = pd.Series([assign_rating(float(p)) for p in pd_te], index=test.index)
    print("\n-- Ratings distribution (out-of-time), band mean PD monotone --")
    dist = (pd.DataFrame({"rating": ratings, "pd": pd_te, "score": scores,
                          "default": yte})
            .groupby("rating")
            .agg(n=("pd", "size"), mean_pd=("pd", "mean"),
                 mean_score=("score", "mean"), obs_rate=("default", "mean"))
            .reindex([r for r in RATING_ORDER if r in set(ratings)]))
    print(dist.to_string())

    hr("1b. VALIDATION REPORT (train vs out-of-time)")
    point, lo, hi = bootstrap_auc_ci(yte, pd_te, n_boot=500, seed=0)
    hl_te = hosmer_lemeshow(yte, pd_te)
    rows = [
        ("AUC", auc(ytr, pd_tr), auc(yte, pd_te)),
        ("Gini", gini(ytr, pd_tr), gini(yte, pd_te)),
        ("KS", ks_statistic(ytr, pd_tr), ks_statistic(yte, pd_te)),
        ("HL chi2 (dof=8)", hosmer_lemeshow(ytr, pd_tr).chi2, hl_te.chi2),
        ("HL p-value", hosmer_lemeshow(ytr, pd_tr).p_value, hl_te.p_value),
    ]
    print(pd.DataFrame(rows, columns=["metric", "train", "out_of_time"]).to_string(index=False))
    print(f"\nOOT AUC 95% bootstrap CI: {point:.3f} [{lo:.3f}, {hi:.3f}] "
          f"(width {hi - lo:.3f} — low-default portfolios have wide CIs)")
    print(f"Score PSI train -> OOT (all years):        {psi(pd_tr, pd_te):.4f}")
    rest = test["year"] != 2020
    print(f"Score PSI train -> OOT excl. 2020:         {psi(pd_tr, pd_te[rest.to_numpy()]):.4f}")

    y2020 = test[test["year"] == 2020]
    pd_2020 = predict_pd(fit, transform(y2020))
    print("\n-- Planted 2020 global contagion year (calibration stress) --")
    print(f"2020: predicted mean PD {pd_2020.mean():.2%} vs observed default "
          f"rate {y2020['default'].mean():.2%} "
          f"(x{y2020['default'].mean() / pd_2020.mean():.1f} understatement)")
    other = test[test["year"] != 2020]
    pd_other = predict_pd(fit, transform(other))
    print(f"Other OOT years: predicted {pd_other.mean():.2%} vs observed "
          f"{other['default'].mean():.2%}")

    resid = train.assign(resid=ytr - pd_tr)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ac = within_country_autocorrelation(resid, "resid")
    print(f"\nWithin-country residual lag-1 autocorrelation: mean "
          f"{ac.mean(skipna=True):+.3f} (panel dependence caveat)")
    for w in caught:
        print(f"WARNING raised by utility: {w.message}")

    # ------------------------------------------------------------------ 2
    hr("2. FX SETTLEMENT (HERSTATT) RISK — gross vs CLS")
    print("-- At-risk window matrix (hours; pay row currency, receive column) --")
    print(time_zone_gap_matrix().to_string())
    usd_rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 1.0 / 148.0}
    book = generate_fx_trade_book()
    rep = book_settlement_report(book, usd_rates)
    print("\n-- 6-trade book --")
    print(rep.to_string(index=False))
    gross = gross_settlement_exposure(book, usd_rates)
    net = net_settlement_exposure(book, usd_rates)
    all_gross = rep["exposure_if_gross_usd"].sum()
    print(f"\nIf every trade settled gross:      USD {all_gross:>15,.0f}")
    print(f"Actual (CLS trades at zero):       USD {gross:>15,.0f}")
    print(f"With bilateral payment netting:    USD {net:>15,.0f}")
    print(f"CLS/PvP eliminates USD {all_gross - gross:,.0f} of principal risk "
          f"({(all_gross - gross) / all_gross:.0%} of the gross book)")

    # ------------------------------------------------------------------ 3
    hr("3. PRE-SETTLEMENT RISK — EE/PFE profile, netting, CVA")
    spot, vol, rd, rf = 1.08, 0.12, 0.03, 0.02
    fwd = FXForward("EURUSD", 10_000_000, 1.08, 1.0, buy_base=True)
    cparty = generate_counterparty_set().set_index("counterparty").loc["EMSovereignX"]
    cva_val, prof = cva_for_forward(fwd, spot, vol, rd, rf,
                                    pd_1y=float(cparty["pd_1y"]),
                                    lgd=float(cparty["lgd"]),
                                    n_steps=24, n_paths=100_000, seed=1)
    show = [5, 11, 17, 23]
    prof_tab = pd.DataFrame({
        "t (y)": prof.times[show],
        "EE": prof.ee[show],
        "PFE 95%": prof.pfe[0.95][show],
        "PFE 99%": prof.pfe[0.99][show],
    })
    print("-- EUR 10m 1y EURUSD forward (ATM-forward strike) --")
    print(prof_tab.to_string(index=False))
    print("\nShape check: exposure GROWS to maturity ~ sqrt(t) for an outright "
          "forward\n(no interim cashflows — the mid-life hump belongs to swaps).")

    sell = FXForward("EURUSD", 10_000_000, 1.08, 1.0, buy_base=False)
    kw = dict(n_steps=24, n_paths=50_000, seed=2)
    net_p = netting_set_profile([fwd, sell], spot, vol, rd, rf, netting=True, **kw)
    gross_p = netting_set_profile([fwd, sell], spot, vol, rd, rf, netting=False, **kw)
    print(f"\n-- Netting set: +10m and -10m EURUSD forwards --")
    print(f"Peak PFE99 without netting agreement: USD {gross_p.peak_pfe(0.99):>13,.0f}")
    print(f"Peak PFE99 with netting agreement:    USD {net_p.peak_pfe(0.99):>13,.0f}")

    print(f"\n-- CVA vs {cparty.name} (rating {cparty['rating']}, "
          f"PD_1y {cparty['pd_1y']:.2%} from scorecard band midpoint, "
          f"LGD {cparty['lgd']:.0%}) --")
    usd_notional = fwd.notional_base * fwd.strike
    print(f"Flat-hazard PD curve, discounted at r_d = {rd:.1%}: "
          f"CVA = USD {cva_val:,.0f} "
          f"({cva_val / usd_notional * 1e4:.1f} bp of USD notional)")

    # ------------------------------------------------------------------ 4
    hr("4. CAPITAL BY RATING BAND (per USD 100 EAD, LGD 45%)")
    ratings4 = list(RATING_ORDER)
    tab = capital_table(ratings4, [rating_midpoint_pd(r) for r in ratings4])
    print(tab.to_string(index=False))
    print("\nNote: standardized RW is 0% for AAA/AA sovereigns (regulatory "
          "convention),\nwhile internal Vasicek capital with sovereign asset "
          "correlation 0.30 is strictly\npositive and exceeds the corporate-"
          "correlation figure at every band.")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
