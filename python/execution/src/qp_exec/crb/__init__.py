"""Central risk book (W26): netting, internalisation, residual hedging.

book   Desk positions, netting, factor exposure, stand-alone versus CRB cost.
hedge  Synthetic tiered flow, hedge-delay policy, exact P&L decomposition,
       saved cost with a bootstrap CI, and hedges as qp_oms submit arguments.
"""

from __future__ import annotations

from qp_exec.crb.book import CentralRiskBook, CostModel, Fill, NettingReport, internalised_volume
from qp_exec.crb.hedge import (
    CrbRun,
    FlowConfig,
    HedgeOrder,
    SavedCost,
    saved_cost,
    simulate_crb,
    to_oms_orders,
)

__all__ = [
    "CentralRiskBook",
    "CostModel",
    "CrbRun",
    "Fill",
    "FlowConfig",
    "HedgeOrder",
    "NettingReport",
    "SavedCost",
    "internalised_volume",
    "saved_cost",
    "simulate_crb",
    "to_oms_orders",
]
