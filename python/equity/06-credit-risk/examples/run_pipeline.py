"""End-to-end corporate credit risk pipeline.

Generate loan book (train + OOT with drift) -> clean -> WOE/IV binning ->
IRLS logistic scorecard -> PD -> validation (AUC/KS, Hosmer-Lemeshow, PSI) ->
expected loss by rating bucket -> Basel IRB K/RWA -> Vasicek analytic vs
Monte Carlo economic capital.

Run from the project root:  python examples/run_pipeline.py   (< 120 s)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eq_credit.cleaning import (
    apply_winsor,
    check_leakage,
    drop_duplicate_loans,
    fit_winsor_bounds,
    LeakageError,
)
from eq_credit.data.synthetic import generate_loan_book, generate_oot_sample
from eq_credit.model import (
    ScorecardScaling,
    crosscheck_sklearn,
    fit_logistic,
    scorecard_points_table,
    stepwise_select,
)
from eq_credit.portfolio_risk import (
    basel_report,
    economic_capital,
    el_by_bucket,
    expected_loss,
    simulate_portfolio_losses,
    vasicek_quantile,
)
from eq_credit.validation import (
    bootstrap_auc_ci,
    brier_score,
    gini,
    hosmer_lemeshow,
    ks_statistic,
    psi,
    psi_report,
    psi_status,
    decile_table,
    roc_auc,
)
from eq_credit.woe import WOETransformer, fit_numeric_binning

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

NUMERIC = [
    "leverage", "interest_coverage", "current_ratio", "roa",
    "log_assets", "behavioral_score", "noise_1", "noise_2",
]
CATEGORICAL = ["sector"]
TARGET = "default"


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------ data
    hr("1. DATA: synthetic loan book (train) + out-of-time sample with drift")
    train = generate_loan_book(30_000, seed=42, n_duplicates=40)
    oot = generate_oot_sample(12_000, seed=123, drift=0.5, calibration_shift=0.5)
    print(f"train: {len(train):,} rows, default rate {train[TARGET].mean():.3%}")
    print(f"OOT:   {len(oot):,} rows, default rate {oot[TARGET].mean():.3%} (drifted)")

    # -------------------------------------------------------------- cleaning
    hr("2. CLEANING: duplicates, leakage guard, winsorization")
    n0 = len(train)
    train = drop_duplicate_loans(train)
    print(f"dropped {n0 - len(train)} duplicate loan_ids")
    try:
        check_leakage(NUMERIC + CATEGORICAL + ["writeoff_flag"])
    except LeakageError as e:
        print(f"leakage guard DEMO (correctly rejected): {e}")
    check_leakage(NUMERIC + CATEGORICAL)  # the real feature set passes
    bounds = fit_winsor_bounds(train, ["leverage", "interest_coverage", "current_ratio", "roa"])
    train_w = apply_winsor(train, bounds)
    oot_w = apply_winsor(oot, bounds)  # OOT uses TRAIN bounds (no peeking)
    print(f"winsor bounds (1%/99%): leverage [{bounds.lower['leverage']:.3f}, "
          f"{bounds.upper['leverage']:.3f}], current_ratio "
          f"[{bounds.lower['current_ratio']:.3f}, {bounds.upper['current_ratio']:.3f}]")

    # ------------------------------------------------------------------- WOE
    hr("3. WOE / IV: supervised binning (missing = own bin)")
    wt = WOETransformer(
        NUMERIC, CATEGORICAL, non_monotone_features=["current_ratio"]
    ).fit(train_w, TARGET)
    print(wt.iv_table().to_string(index=False))

    print("\n-- Binning table: leverage (monotone, capped true effect) --")
    print(wt.binnings_["leverage"].table.to_string(index=False))
    print("\n-- Binning table: current_ratio (U-shaped — WOE captures the")
    print("   nonlinearity; a linear logit on the raw ratio misses it) --")
    print(wt.binnings_["current_ratio"].table.to_string(index=False))

    # Demonstrate: linear logit on RAW current_ratio finds nothing, WOE does.
    cr_med = train_w["current_ratio"].median()
    x_raw = train_w[["current_ratio"]].fillna(cr_med)
    fit_raw = fit_logistic(x_raw, train_w[TARGET], feature_names=["current_ratio_raw"])
    x_woe = pd.DataFrame(
        {"woe_cr": wt.binnings_["current_ratio"].transform(train_w["current_ratio"])}
    )
    fit_woe_cr = fit_logistic(x_woe, train_w[TARGET], feature_names=["woe_cr"])
    auc_raw = roc_auc(train_w[TARGET].to_numpy(), fit_raw.predict_proba(x_raw))
    auc_woe = roc_auc(train_w[TARGET].to_numpy(), fit_woe_cr.predict_proba(x_woe))
    print(f"\ncurrent_ratio alone — AUC raw linear logit: {auc_raw:.4f}  "
          f"vs WOE logit: {auc_woe:.4f}")

    woe_train = wt.transform(train_w)
    woe_oot = wt.transform(oot_w)
    y_train = train_w[TARGET].to_numpy()
    y_oot = oot_w[TARGET].to_numpy()

    # ------------------------------------------------------------- scorecard
    hr("4. SCORECARD: IRLS logistic regression (from scratch)")
    ivs = {f"woe_{f}": fb.iv for f, fb in wt.binnings_.items()}
    selected = stepwise_select(woe_train, y_train, ivs)
    dropped = sorted(set(woe_train.columns) - set(selected))
    print(f"stepwise-by-IV selected: {selected}")
    print(f"dropped (IV < 0.02 or insignificant): {dropped}")

    fit = fit_logistic(woe_train[selected], y_train, feature_names=selected)
    print(f"\nIRLS converged in {fit.n_iter} iterations, log-lik {fit.loglik:,.1f}")
    print(fit.summary().to_string(index=False))
    diff = crosscheck_sklearn(woe_train[selected], y_train, fit)
    print(f"\nmax |coef - sklearn LogisticRegression(no penalty)| = {diff:.2e}")

    # Noise features must be absent / insignificant.
    assert "woe_noise_1" not in selected and "woe_noise_2" not in selected

    scaling = ScorecardScaling(base_score=600, base_odds=50, pdo=20)
    print(f"\nscaling: 600 points at odds 50:1, PDO 20 -> factor={scaling.factor:.4f}, "
          f"offset={scaling.offset:.4f}")
    pts = scorecard_points_table(fit, wt.binnings_, scaling)
    show = pts[pts["feature"].isin(["leverage", "behavioral_score"])]
    print("\n-- Scorecard points (leverage & behavioral_score bins) --")
    print(show.to_string(index=False))

    pd_train = fit.predict_proba(woe_train[selected])
    pd_oot = fit.predict_proba(woe_oot[selected])
    score_train = scaling.score_from_pd(pd_train)
    score_oot = scaling.score_from_pd(pd_oot)

    # ------------------------------------------------------------ validation
    hr("5. VALIDATION: discrimination, calibration, stability")
    auc_tr, lo, hi = bootstrap_auc_ci(y_train, pd_train, n_boot=200, seed=7)
    auc_oo = roc_auc(y_oot, pd_oot)
    print(f"AUC   train: {auc_tr:.4f}  [95% CI {lo:.4f}, {hi:.4f}]   OOT: {auc_oo:.4f}")
    print(f"Gini  train: {gini(y_train, pd_train):.4f}                        "
          f"OOT: {gini(y_oot, pd_oot):.4f}")
    print(f"KS    train: {ks_statistic(y_train, pd_train):.4f}                        "
          f"OOT: {ks_statistic(y_oot, pd_oot):.4f}")
    auc_true = roc_auc(y_train, train_w["true_pd"].to_numpy())
    print(f"(theoretical max AUC on train, using true PDs: {auc_true:.4f})")

    chi2_tr, p_tr, _ = hosmer_lemeshow(y_train, pd_train)
    chi2_oo, p_oo, _ = hosmer_lemeshow(y_oot, pd_oot)
    print(f"\nHosmer-Lemeshow train: chi2={chi2_tr:.2f} (p={p_tr:.3f})   "
          f"OOT: chi2={chi2_oo:.2f} (p={p_oo:.2e})")
    if p_oo < 0.01:
        print("=> OOT calibration REJECTED: the shifted regime under-predicts PDs")
    print(f"Brier train: {brier_score(y_train, pd_train):.5f}   "
          f"OOT: {brier_score(y_oot, pd_oot):.5f}")
    print(f"realised OOT default rate {y_oot.mean():.3%} vs mean predicted {pd_oot.mean():.3%}")

    print("\n-- Rank ordering: observed default rate by predicted-PD decile (train) --")
    print(decile_table(y_train, pd_train).to_string(index=False))

    print("\n-- PSI per feature, train vs OOT (0.10 / 0.25 thresholds) --")
    print(psi_report(train_w, oot_w, NUMERIC).to_string(index=False))
    score_psi = psi(score_train, score_oot)
    print(f"\nscore PSI train->OOT: {score_psi:.4f}  ({psi_status(score_psi)})")

    # --------------------------------------------------------- expected loss
    hr("6. EXPECTED LOSS by rating bucket (PD x LGD x EAD)")
    port = train_w.copy()
    port["pd_hat"] = pd_train
    port["score"] = score_train
    band_edges = [-np.inf, 0.005, 0.01, 0.02, 0.05, 0.10, np.inf]
    band_labels = ["AAA-A (<0.5%)", "BBB (0.5-1%)", "BB (1-2%)",
                   "B (2-5%)", "CCC (5-10%)", "C (>10%)"]
    port["rating"] = pd.cut(port["pd_hat"], band_edges, labels=band_labels)
    el_tab = el_by_bucket(port, "rating", pd_col="pd_hat")
    print(el_tab.to_string(index=False))
    _, el_total = expected_loss(port["pd_hat"], port["lgd"], port["ead"])
    _, el_downturn = expected_loss(port["pd_hat"], port["lgd"], port["ead"],
                                   downturn_lgd_haircut=0.25)
    print(f"\nportfolio EL: {el_total:,.0f} "
          f"({el_total / port['ead'].sum():.3%} of EAD {port['ead'].sum():,.0f})")
    print(f"with 25% downturn-LGD haircut: {el_downturn:,.0f}")

    # ----------------------------------------------------------------- Basel
    hr("7. BASEL IRB: K / risk weight / RWA by PD band (LGD 45%, M 2.5)")
    rep = basel_report(np.array([0.0003, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]))
    print(rep.to_string(index=False))
    print("\nhand-check: K(PD=1%, LGD=45%, M=2.5) = "
          f"{rep.loc[rep['pd'] == 0.01, 'K'].iloc[0]:.6f} -> RW = "
          f"{rep.loc[rep['pd'] == 0.01, 'risk_weight'].iloc[0]:.2%} "
          "(published Basel II corporate curve: 92.32%)")

    from eq_credit.portfolio_risk import basel_k, risk_weighted_assets
    k_loans = basel_k(port["pd_hat"].to_numpy(), port["lgd"].to_numpy())
    rwa = risk_weighted_assets(k_loans, port["ead"].to_numpy())
    print(f"\nportfolio RWA: {rwa.sum():,.0f}  "
          f"(avg risk weight {rwa.sum() / port['ead'].sum():.1%}; "
          f"8% capital charge: {0.08 * rwa.sum():,.0f})")

    # --------------------------------------------------------------- Vasicek
    hr("8. VASICEK one-factor: analytic vs Monte Carlo, economic capital")
    pd_avg = float(port["pd_hat"].mean())
    lgd_avg = float(port["lgd"].mean())
    rho = 0.15
    el_rate = float((port["pd_hat"] * port["lgd"] * port["ead"]).sum()
                    / port["ead"].sum())

    q999_inf = float(vasicek_quantile(0.999, pd_avg, rho)) * lgd_avg
    print(f"homogeneous reference: PD={pd_avg:.3%}, LGD={lgd_avg:.1%}, rho={rho}")
    print(f"99.9% loss rate — analytic, infinitely granular:      {q999_inf:.3%}")
    # Granularity adjustment: same homogeneous portfolio, finite loan counts.
    for n_loans, seed in [(100, 12), (1_000, 13), (10_000, 14)]:
        ls = simulate_portfolio_losses(
            pd_avg, lgd_avg, 1.0, rho, n_sims=30_000, seed=seed, n_loans=n_loans
        )
        print(f"99.9% loss rate — MC, {n_loans:>6,} equal loans:           "
              f"{float(np.quantile(ls, 0.999)):.3%}")
    print("=> finite-portfolio tail >= infinitely granular tail, converging "
          "from above as N grows (granularity adjustment)")

    # The actual heterogeneous book (dispersed PDs and LGDs).
    losses_fin = simulate_portfolio_losses(
        port["pd_hat"].to_numpy(), port["lgd"].to_numpy(), port["ead"].to_numpy(),
        rho, n_sims=30_000, seed=11,
    )
    q999_fin = float(np.quantile(losses_fin, 0.999))
    print(f"\n99.9% loss rate — MC, actual book ({len(port):,} heterogeneous "
          f"loans): {q999_fin:.3%}")
    print("(below the homogeneous reference: PD dispersion around the mean "
          "thins the tail relative to a single-PD book at the mean PD)")

    ec = economic_capital(losses_fin, el_rate, q=0.999)
    k_port_rate = float((k_loans * port["ead"]).sum() / port["ead"].sum())
    print(f"\nexpected loss rate: {el_rate:.3%}")
    print(f"economic capital (99.9% MC quantile - EL): {ec:.3%} of EAD")
    print(f"Basel K (EAD-weighted, actual PDs/LGDs):   {k_port_rate:.3%} of EAD")
    print("(Basel K uses PD-dependent R(PD) ~ 0.19-0.24 vs the flat rho=0.15 "
          "here, and a maturity add-on — see docs/VALIDATION.md)")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
