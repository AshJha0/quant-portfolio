"""Research statistics. The code lives in :mod:`qp_core.stats`.

This module re-exports it so ``fx_pairs.research_stats`` keeps working.
"""

from __future__ import annotations

from qp_core.stats import (
    SeedSummary,
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    lo_sharpe_se,
    newey_west_lag,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)

__all__ = [
    "newey_west_lag",
    "sharpe_ratio",
    "lo_sharpe_se",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "pbo_cscv",
    "block_bootstrap_ci",
    "seed_distribution",
    "SeedSummary",
    "TrialLog",
]
