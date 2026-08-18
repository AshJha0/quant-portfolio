"""fx_port — FX portfolio optimization & currency risk allocation.

Pipeline: total-return estimation (spot + carry) → style signals
(carry/momentum/value) → covariance modelling (LW shrinkage, EWMA,
risk-on/off factor) → mean-variance / risk parity / CVaR-constrained
allocation → optimal currency hedging → walk-forward backtesting with pip
costs, carry accrual and base-currency (GBP) reporting.
"""

from .backtest import (
    BacktestResult,
    base_conversion_returns,
    convert_base,
    pips_to_bps,
    run_backtest,
)
from .covariance import (
    OneFactorModel,
    ewma_cov,
    is_psd,
    lw_shrinkage,
    one_factor_cov,
    psd_repair,
    sample_cov,
)
from .cvar_opt import (
    CVaRResult,
    carry_sizing,
    empirical_cvar,
    empirical_var,
    max_return_cvar_constrained,
    min_cvar,
)
from .hedging import (
    HedgeReport,
    hedged_returns,
    optimal_hedge_ratios,
    variance_decomposition,
)
from .metrics import (
    annualized_return,
    annualized_vol,
    excess_kurtosis,
    max_drawdown,
    sharpe_ratio,
    sharpe_se_lo,
    skewness,
    sortino_ratio,
    style_attribution,
    summary,
)
from .mvo import (
    MVOResult,
    dollar_neutral_weights,
    efficient_frontier,
    frontier_weights,
    max_utility,
    min_variance_slsqp,
    min_variance_weights,
    tangency_weights,
)
from .returns_est import (
    ReturnDecomposition,
    carry_log_returns,
    carry_signal,
    momentum_signal,
    rank_weights,
    shrunk_means,
    signal_weights,
    spot_log_returns,
    style_returns,
    total_log_returns,
    value_signal,
)
from .risk_parity import (
    erc_weights,
    portfolio_vol,
    risk_contributions,
    vol_target,
)

__all__ = [
    "BacktestResult",
    "CVaRResult",
    "HedgeReport",
    "MVOResult",
    "OneFactorModel",
    "ReturnDecomposition",
    "annualized_return",
    "annualized_vol",
    "base_conversion_returns",
    "carry_log_returns",
    "carry_signal",
    "carry_sizing",
    "convert_base",
    "dollar_neutral_weights",
    "efficient_frontier",
    "empirical_cvar",
    "empirical_var",
    "erc_weights",
    "ewma_cov",
    "excess_kurtosis",
    "frontier_weights",
    "hedged_returns",
    "is_psd",
    "lw_shrinkage",
    "max_drawdown",
    "max_return_cvar_constrained",
    "max_utility",
    "min_cvar",
    "min_variance_slsqp",
    "min_variance_weights",
    "momentum_signal",
    "one_factor_cov",
    "optimal_hedge_ratios",
    "pips_to_bps",
    "portfolio_vol",
    "psd_repair",
    "rank_weights",
    "risk_contributions",
    "run_backtest",
    "sample_cov",
    "sharpe_ratio",
    "sharpe_se_lo",
    "shrunk_means",
    "signal_weights",
    "skewness",
    "sortino_ratio",
    "spot_log_returns",
    "style_attribution",
    "style_returns",
    "summary",
    "tangency_weights",
    "total_log_returns",
    "value_signal",
    "variance_decomposition",
    "vol_target",
]

__version__ = "1.0.0"
