"""Research portfolio layer for ``qp_bt`` (W24c): covariance estimators,
constrained optimisers and a walk-forward portfolio P&L so studies report
book-level results. Pure NumPy/SciPy; no dependency on the event core."""

from .covariance import ledoit_wolf, pca_cov, sample_cov
from .optimise import mean_variance, risk_contributions, risk_parity, vol_target
from .research import PortfolioResult, portfolio_backtest

__all__ = [
    "PortfolioResult",
    "ledoit_wolf",
    "mean_variance",
    "pca_cov",
    "portfolio_backtest",
    "risk_contributions",
    "risk_parity",
    "sample_cov",
    "vol_target",
]
