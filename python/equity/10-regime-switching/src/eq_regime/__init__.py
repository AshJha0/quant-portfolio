"""eq_regime: equity regime-switching quant strategy.

Pipeline: feature engineering -> PCA -> HMM / GMM -> regime detection
(ONLINE filtered probabilities) -> regime-conditional signals -> portfolio
construction -> walk-forward backtest -> regime-conditional risk analysis.

All models (PCA, GMM, HMM) are implemented from scratch and cross-checked
against scikit-learn / hmmlearn in the test suite.
"""

from .backtest import (
    BacktestResult,
    ExecutionConfig,
    backtest_from_detection,
    ma_timing_weights,
    random_timing_weights,
    run_ledger,
    summary_stats,
    vol_target_weights,
    walk_forward_backtest,
)
from .detection import (
    RegimeLabels,
    expanding_fit_detect,
    filtered_probabilities,
    flip_flop_rate,
    label_states,
    labels_from_vol_means,
    smoothed_probabilities,
)
from .features import (
    average_pairwise_correlation,
    build_features,
    credit_proxy_spread,
    drawdown_depth,
    expanding_zscore,
    realized_vol,
    return_dispersion,
    term_proxy,
    trend_strength,
)
from .gmm import GMMFit, aic, bic, fit_gmm, gmm_log_likelihood, match_permutation, select_k_bic
from .hmm import (
    HMMFit,
    expected_durations,
    fit_hmm,
    forward_backward,
    forward_filter,
    stationary_distribution,
    viterbi,
)
from .pca import PCAModel, fit_pca, project, rolling_pca, scree_table
from .risk import (
    flip_aftermath,
    per_regime_stats,
    regime_runs,
    regimes_known_at_decision,
    transition_attribution,
)
from .strategy import (
    build_weights,
    causal_min_duration,
    hysteresis_regime,
    naive_threshold_regime,
    regime_target_weight,
    turnover,
    vol_target_scale,
)

__version__ = "3.0.0"

__all__ = [
    # backtest
    "BacktestResult",
    "ExecutionConfig",
    # gmm
    "GMMFit",
    # hmm
    "HMMFit",
    # pca
    "PCAModel",
    # detection
    "RegimeLabels",
    "aic",
    "average_pairwise_correlation",
    "backtest_from_detection",
    "bic",
    "build_features",
    "build_weights",
    "causal_min_duration",
    "credit_proxy_spread",
    "drawdown_depth",
    "expanding_fit_detect",
    # features
    "expanding_zscore",
    "expected_durations",
    "filtered_probabilities",
    "fit_gmm",
    "fit_hmm",
    "fit_pca",
    "flip_aftermath",
    "flip_flop_rate",
    "forward_backward",
    "forward_filter",
    "gmm_log_likelihood",
    # strategy
    "hysteresis_regime",
    "label_states",
    "labels_from_vol_means",
    "ma_timing_weights",
    "match_permutation",
    "naive_threshold_regime",
    "per_regime_stats",
    "project",
    "random_timing_weights",
    "realized_vol",
    "regime_runs",
    "regime_target_weight",
    # risk
    "regimes_known_at_decision",
    "return_dispersion",
    "rolling_pca",
    "run_ledger",
    "scree_table",
    "select_k_bic",
    "smoothed_probabilities",
    "stationary_distribution",
    "summary_stats",
    "term_proxy",
    "transition_attribution",
    "trend_strength",
    "turnover",
    "viterbi",
    "vol_target_scale",
    "vol_target_weights",
    "walk_forward_backtest",
]
