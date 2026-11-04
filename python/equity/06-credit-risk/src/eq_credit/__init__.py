"""eq_credit: corporate credit risk / PD modelling (bank-style scorecard).

Pipeline: data cleaning -> WOE/IV binning -> IRLS logistic scorecard -> PD ->
validation (AUC/KS, Hosmer-Lemeshow, PSI) -> expected loss -> Basel IRB and
Vasicek economic capital.
"""

from eq_credit.cleaning import (
    FORBIDDEN_POST_OUTCOME_FIELDS,
    LeakageError,
    MedianImputer,
    WinsorBounds,
    apply_winsor,
    check_leakage,
    drop_duplicate_loans,
    find_duplicates,
    fit_winsor_bounds,
    train_oot_split,
)
from eq_credit.model import (
    LogisticFit,
    ScorecardScaling,
    SeparationWarning,
    crosscheck_sklearn,
    fit_logistic,
    scorecard_points_table,
    stepwise_select,
)
from eq_credit.portfolio_risk import (
    asset_correlation,
    basel_k,
    basel_report,
    economic_capital,
    el_by_bucket,
    expected_loss,
    maturity_adjustment_b,
    risk_weighted_assets,
    simulate_portfolio_losses,
    vasicek_cdf,
    vasicek_quantile,
)
from eq_credit.validation import (
    bootstrap_auc_ci,
    brier_score,
    calibration_table,
    decile_table,
    gini,
    hosmer_lemeshow,
    is_monotone,
    ks_statistic,
    ks_table,
    pd_backtest_by_grade,
    psi,
    psi_from_proportions,
    psi_report,
    psi_status,
    roc_auc,
    roc_curve_points,
)
from eq_credit.woe import (
    FeatureBinning,
    SuspiciousIVWarning,
    WOETransformer,
    fit_categorical_binning,
    fit_numeric_binning,
    iv_strength,
    woe_iv_from_counts,
)

__all__ = [
    # cleaning
    "FORBIDDEN_POST_OUTCOME_FIELDS",
    "FeatureBinning",
    "LeakageError",
    "LogisticFit",
    "MedianImputer",
    "ScorecardScaling",
    # model
    "SeparationWarning",
    # woe
    "SuspiciousIVWarning",
    "WOETransformer",
    "WinsorBounds",
    "apply_winsor",
    "asset_correlation",
    "basel_k",
    "basel_report",
    "bootstrap_auc_ci",
    "brier_score",
    "calibration_table",
    "check_leakage",
    "crosscheck_sklearn",
    "decile_table",
    "drop_duplicate_loans",
    "economic_capital",
    "el_by_bucket",
    # portfolio risk
    "expected_loss",
    "find_duplicates",
    "fit_categorical_binning",
    "fit_logistic",
    "fit_numeric_binning",
    "fit_winsor_bounds",
    "gini",
    "hosmer_lemeshow",
    "is_monotone",
    "iv_strength",
    "ks_statistic",
    "ks_table",
    "maturity_adjustment_b",
    "pd_backtest_by_grade",
    "psi",
    "psi_from_proportions",
    "psi_report",
    "psi_status",
    "risk_weighted_assets",
    "roc_auc",
    # validation
    "roc_curve_points",
    "scorecard_points_table",
    "simulate_portfolio_losses",
    "stepwise_select",
    "train_oot_split",
    "vasicek_cdf",
    "vasicek_quantile",
    "woe_iv_from_counts",
]

__version__ = "3.0.0"
