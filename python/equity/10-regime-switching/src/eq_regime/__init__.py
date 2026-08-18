"""eq_regime — equity regime-switching quant strategy.

Pipeline: feature engineering -> PCA -> HMM / GMM -> regime detection
(ONLINE filtered probabilities) -> regime-conditional signals -> portfolio
construction -> walk-forward backtest -> regime-conditional risk analysis.

All models (PCA, GMM, HMM) are implemented from scratch and cross-checked
against scikit-learn / hmmlearn in the test suite.
"""

from .backtest import (
    BacktestResult,
    ma_timing_weights,
    run_ledger,
    summary_stats,
    walk_forward_backtest,
)
from .detection import (
    RegimeLabels,
    expanding_fit_detect,
    filtered_probabilities,
    flip_flop_rate,
    label_states,
    labels_from_vol_means,
    min_duration_filter,
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
from .risk import flip_aftermath, per_regime_stats, regime_runs, transition_attribution
from .strategy import (
    build_weights,
    hysteresis_regime,
    naive_threshold_regime,
    regime_target_weight,
    turnover,
    vol_target_scale,
)

__version__ = "1.0.0"

__all__ = [
    # features
    "expanding_zscore", "realized_vol", "return_dispersion",
    "average_pairwise_correlation", "drawdown_depth", "trend_strength",
    "credit_proxy_spread", "term_proxy", "build_features",
    # pca
    "PCAModel", "fit_pca", "project", "scree_table", "rolling_pca",
    # gmm
    "GMMFit", "fit_gmm", "gmm_log_likelihood", "bic", "aic",
    "select_k_bic", "match_permutation",
    # hmm
    "HMMFit", "fit_hmm", "forward_filter", "forward_backward", "viterbi",
    "stationary_distribution", "expected_durations",
    # detection
    "RegimeLabels", "label_states", "labels_from_vol_means", "filtered_probabilities",
    "smoothed_probabilities", "flip_flop_rate", "min_duration_filter",
    "expanding_fit_detect",
    # strategy
    "hysteresis_regime", "naive_threshold_regime", "regime_target_weight",
    "vol_target_scale", "build_weights", "turnover",
    # backtest
    "BacktestResult", "run_ledger", "ma_timing_weights",
    "walk_forward_backtest", "summary_stats",
    # risk
    "per_regime_stats", "transition_attribution", "flip_aftermath", "regime_runs",
]
