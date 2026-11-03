"""RFQ (W26): hit-ratio model, optimal and inventory-skewed quotes, win-rate,
markout and last-look reports (FX Global Code 2021, principle 17)."""

from __future__ import annotations

from qp_exec.rfq.analytics import (
    LastLookRecord,
    LastLookReport,
    MarkoutStat,
    RateStat,
    RfqRecord,
    last_look_report,
    markouts,
    win_rates,
)
from qp_exec.rfq.quote import (
    HitRatioModel,
    expected_profit,
    fit_hit_ratio,
    lambert_w0,
    optimal_markup,
    skewed_markup,
)

__all__ = [
    "HitRatioModel",
    "LastLookRecord",
    "LastLookReport",
    "MarkoutStat",
    "RateStat",
    "RfqRecord",
    "expected_profit",
    "fit_hit_ratio",
    "lambert_w0",
    "last_look_report",
    "markouts",
    "optimal_markup",
    "skewed_markup",
    "win_rates",
]
