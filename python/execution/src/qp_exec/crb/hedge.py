"""Synthetic tiered client flow, residual hedging with alpha decay, and the saved cost.

Plain idea
----------
Client flow is informative: after a tier-A (informed) client buys, the
price tends to drift up for a while. A CRB that holds the residual pays that
drift (adverse selection) but may wait for offsetting flow from other desks;
a CRB that hedges at once pays spread and impact. This module simulates
flow, runs a hedge-delay policy, and decomposes the CRB P&L so the pieces
add to the total exactly.

Formal statement
----------------
Mid follows ``m_{t+1} = m_t + alpha_t + sigma_m eps_t`` with alpha from
client flow, ``alpha_t = sum_j kappa_{tier_j} x_j exp(-(t - t_j) / tau)``
(Cartea, Jaimungal and Penalva 2015, ch. 10). The CRB internalises each
client trade, earns ``client_spread`` per unit, and once every ``delay``
steps hedges a fraction ``ratio`` of its inventory at cost ``c(v)``
(``CostModel``). P&L decomposition, exact by construction:

    total = spread_capture + inventory_pnl - hedge_cost,

with ``inventory_pnl = sum_t q_t (m_{t+1} - m_t)``; ``test_crb.py`` checks
the sum to 1e-9. The saved cost of the CRB versus hedging every client trade
at once is reported per seed with a bootstrap percentile CI.

Worked example
--------------
>>> from qp_exec.crb.hedge import FlowConfig, simulate_crb
>>> r = simulate_crb(FlowConfig(steps=200), delay=5, seed=1)
>>> abs(r.total - (r.spread_capture + r.inventory_pnl - r.hedge_cost)) < 1e-9
True
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = [
    "CrbRun",
    "FlowConfig",
    "HedgeOrder",
    "SavedCost",
    "saved_cost",
    "simulate_crb",
    "to_oms_orders",
]


@dataclass(frozen=True)
class FlowConfig:
    """Synthetic flow: tier mix, per-tier size, alpha loading and spread charged."""

    steps: int = 500
    tiers: Sequence[str] = ("A", "B", "C")
    tier_prob: Sequence[float] = (0.2, 0.4, 0.4)
    arrival_prob: float = 0.6
    mean_size: Mapping[str, float] = field(default_factory=lambda: {"A": 5.0, "B": 2.0, "C": 1.0})
    alpha_kappa: Mapping[str, float] = field(default_factory=lambda: {"A": 0.004, "B": 0.001, "C": 0.0})
    client_spread: Mapping[str, float] = field(default_factory=lambda: {"A": 0.005, "B": 0.01, "C": 0.02})
    alpha_tau: float = 10.0
    mid0: float = 100.0
    sigma_mid: float = 0.02
    half_spread: float = 0.01  # market half-spread per unit when hedging
    impact: float = 0.002  # temporary impact per unit^1.5
    n_desks: int = 3


@dataclass(frozen=True)
class HedgeOrder:
    """A hedge the CRB sends; maps one to one onto ``qp_oms`` ``OrderBook.submit``."""

    step: int
    side: str  # "1" buy, "2" sell (FIX 54)
    qty: int
    symbol: str
    price: float | None


@dataclass(frozen=True)
class CrbRun:
    """One simulated path; ``total == spread_capture + inventory_pnl - hedge_cost``."""

    total: float
    spread_capture: float
    inventory_pnl: float
    hedge_cost: float
    internalised: float
    client_volume: float
    hedge_volume: float
    max_abs_inventory: float
    orders: list[HedgeOrder]


def _cost(v: float, cfg: FlowConfig) -> float:
    a = abs(v)
    return float(a * cfg.half_spread + cfg.impact * a**1.5)


def simulate_crb(
    cfg: FlowConfig, *, delay: int = 1, ratio: float = 1.0, seed: int = 0, symbol: str = "SYN"
) -> CrbRun:
    """Simulate flow and a hedge-every-``delay``-steps policy hedging ``ratio`` of inventory.

    ``delay=1, ratio=1`` hedges every step (no internalisation across time);
    larger ``delay`` lets offsetting flow from other desks net first.
    """
    if delay < 1 or not (0.0 <= ratio <= 1.0) or cfg.steps < 1:
        raise ValueError("delay >= 1, ratio in [0, 1], steps >= 1")
    rng = np.random.default_rng(seed)
    probs = np.asarray(cfg.tier_prob, dtype=np.float64)
    probs = probs / probs.sum()
    m = cfg.mid0
    q = 0.0
    alpha_state = 0.0
    decay = math.exp(-1.0 / cfg.alpha_tau)
    spread = inv = hcost = 0.0
    client_vol = hedge_vol = 0.0
    pending: list[float] = []
    internal = 0.0
    max_q = 0.0
    orders: list[HedgeOrder] = []
    for t in range(cfg.steps):
        if rng.random() < cfg.arrival_prob:
            tier = str(cfg.tiers[int(rng.choice(len(probs), p=probs))])
            size = float(rng.exponential(cfg.mean_size[tier]))
            client_side = 1.0 if rng.random() < 0.5 else -1.0  # client buys -> firm sells
            x = -client_side * size  # firm's signed fill
            q += x
            pending.append(x)
            spread += size * cfg.client_spread[tier]
            client_vol += size
            alpha_state += cfg.alpha_kappa[tier] * client_side * size
        if (t + 1) % delay == 0:
            buys = sum(v for v in pending if v > 0)
            sells = -sum(v for v in pending if v < 0)
            internal += min(buys, sells)
            pending = []
            h = -ratio * q
            if h != 0.0:
                c = _cost(h, cfg)
                hcost += c
                hedge_vol += abs(h)
                q += h
                qty = round(abs(h))
                if qty > 0:
                    orders.append(HedgeOrder(t, "1" if h > 0 else "2", qty, symbol, None))
        max_q = max(max_q, abs(q))
        dm = alpha_state + cfg.sigma_mid * float(rng.standard_normal())
        inv += q * dm
        m += dm
        alpha_state *= decay
    total = spread + inv - hcost
    return CrbRun(
        total=total,
        spread_capture=spread,
        inventory_pnl=inv,
        hedge_cost=hcost,
        internalised=internal,
        client_volume=client_vol,
        hedge_volume=hedge_vol,
        max_abs_inventory=max_q,
        orders=orders,
    )


@dataclass(frozen=True)
class SavedCost:
    """Mean per-seed P&L gain of a policy over immediate hedging, with a percentile CI."""

    mean: float
    lower: float
    upper: float
    n_seeds: int
    per_seed: npt.NDArray[np.float64]


def saved_cost(
    cfg: FlowConfig,
    *,
    delay: int,
    ratio: float = 1.0,
    n_seeds: int = 200,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> SavedCost:
    """Paired comparison: same seed (same flow and noise) under both policies."""
    if n_seeds < 10:
        raise ValueError("n_seeds must be >= 10")
    diffs = np.empty(n_seeds)
    for s in range(n_seeds):
        base = simulate_crb(cfg, delay=1, ratio=1.0, seed=s)
        pol = simulate_crb(cfg, delay=delay, ratio=ratio, seed=s)
        diffs[s] = pol.total - base.total
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return SavedCost(float(diffs.mean()), float(lo), float(hi), n_seeds, diffs)


def to_oms_orders(orders: Sequence[HedgeOrder], prefix: str = "crb") -> list[dict[str, object]]:
    """Keyword arguments for ``qp_oms.core.book.OrderBook.submit``, one per hedge.

    Kept as plain dicts so this package does not import ``qp_oms``; the
    integration seam is ``book.submit(**kw)``.
    """
    return [
        {
            "clordid": f"{prefix}-{i:06d}",
            "side": o.side,
            "symbol": o.symbol,
            "order_qty": o.qty,
            "price": o.price,
        }
        for i, o in enumerate(orders)
    ]
