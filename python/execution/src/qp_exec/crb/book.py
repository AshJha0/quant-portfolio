"""Central risk book: net desk flow, internalise, hedge the residual.

Plain idea
----------
Several desks trade the same instruments with clients. Run separately, each
desk hedges its own flow in the market and pays the spread every time. A
central risk book (CRB) pools the flow first: a client buy on one desk and
a client sell on another cancel inside the firm (internalisation), and only
the net residual goes to the market. The saving is the spread and impact
the firm no longer pays.

Formal statement
----------------
Let desk ``d`` hold position vector ``q_d`` over instruments. The CRB
position is ``Q = sum_d q_d``; netting conserves risk exactly, so for any
factor loadings ``B`` the factor exposure ``B Q`` equals ``sum_d B q_d``.
Internalised volume in instrument ``i`` over a batch of client trades with
signed sizes ``x`` is ``min(sum x^+, sum x^-)``.

Hedging cost of trading ``|v|`` with half-spread ``h`` and square-root
impact coefficient ``eta`` (daily volume ``V``, daily vol ``sigma``):

    c(v) = |v| p (h + eta sigma sqrt(|v| / V))   (Almgren et al. 2005 form)

Stand-alone desks that flatten at the same time all hit the same market,
so their joint cost in instrument ``i`` is ``c(G_i)`` with gross volume
``G_i = sum_d |q_{d,i}|``. The CRB pays ``c(Q_i)``. Since ``|Q_i| <= G_i``
and ``c`` increases in ``|v|``, the CRB never costs more. (Summing
``c(q_{d,i})`` desk by desk would understate the stand-alone cost: ``c``
grows like ``|v|^1.5``, so splitting a same-sign order across desks looks
cheaper only because it ignores that the desks impact each other.)

Worked example
--------------
>>> from qp_exec.crb.book import CentralRiskBook, Fill
>>> crb = CentralRiskBook(["ES", "NQ"])
>>> crb.book(Fill("eq_cash", "ES", 10.0, 100.0, "A"))
>>> crb.book(Fill("eq_deriv", "ES", -7.0, 100.0, "B"))
>>> crb.net()["ES"], crb.gross()["ES"]
(3.0, 17.0)
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["CentralRiskBook", "CostModel", "Fill", "NettingReport", "internalised_volume"]


@dataclass(frozen=True)
class Fill:
    """One client trade booked by a desk. ``qty`` is signed from the firm's side."""

    desk: str
    instrument: str
    qty: float
    price: float
    tier: str = "B"
    ts_ns: int = 0
    client_spread: float = 0.0  # half-spread earned against mid, price units per unit

    def __post_init__(self) -> None:
        if not (math.isfinite(self.qty) and math.isfinite(self.price) and self.price > 0.0):
            raise ValueError("qty must be finite and price finite and > 0")


@dataclass(frozen=True)
class CostModel:
    """Linear spread plus square-root impact, per instrument."""

    half_spread_bps: Mapping[str, float]
    daily_volume: Mapping[str, float]
    daily_vol: Mapping[str, float]
    eta: float = 0.5

    def cost(self, instrument: str, qty: float, price: float) -> float:
        v = abs(qty)
        if v == 0.0:
            return 0.0
        h = self.half_spread_bps[instrument] * 1e-4
        impact = self.eta * self.daily_vol[instrument] * math.sqrt(v / self.daily_volume[instrument])
        return v * price * (h + impact)


@dataclass(frozen=True)
class NettingReport:
    """Stand-alone versus CRB hedge cost and the volume kept inside the firm."""

    standalone_cost: float
    crb_cost: float
    saved: float
    internalised: dict[str, float]
    net: dict[str, float]
    gross: dict[str, float]


def internalised_volume(signed: Iterable[float]) -> float:
    """``min(buys, sells)`` of a batch of signed sizes: volume that crossed internally."""
    buys = sells = 0.0
    for x in signed:
        if x > 0:
            buys += x
        else:
            sells -= x
    return min(buys, sells)


@dataclass
class CentralRiskBook:
    """Positions by desk and instrument, plus the fills that built them."""

    instruments: Sequence[str]
    positions: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    fills: list[Fill] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)

    def book(self, f: Fill) -> None:
        if f.instrument not in self.instruments:
            raise KeyError(f"unknown instrument {f.instrument!r}")
        d = self.positions[f.desk]
        d[f.instrument] = d.get(f.instrument, 0.0) + f.qty
        self.fills.append(f)
        self.marks[f.instrument] = f.price

    def desk_vector(self, desk: str) -> npt.NDArray[np.float64]:
        d = self.positions.get(desk, {})
        return np.array([d.get(i, 0.0) for i in self.instruments])

    def net(self) -> dict[str, float]:
        out = dict.fromkeys(self.instruments, 0.0)
        for d in self.positions.values():
            for i, q in d.items():
                out[i] += q
        return out

    def gross(self) -> dict[str, float]:
        out = dict.fromkeys(self.instruments, 0.0)
        for d in self.positions.values():
            for i, q in d.items():
                out[i] += abs(q)
        return out

    def factor_exposure(self, loadings: npt.ArrayLike, desk: str | None = None) -> npt.NDArray[np.float64]:
        """``B @ Q`` (or ``B @ q_desk``); ``loadings`` has shape (factors, instruments).

        Positions are in units; multiply by price inside ``loadings`` for value exposure.
        """
        b = np.asarray(loadings, dtype=np.float64)
        q = np.array([self.net()[i] for i in self.instruments]) if desk is None else self.desk_vector(desk)
        out: npt.NDArray[np.float64] = b @ q
        return out

    def factor_risk(self, loadings: npt.ArrayLike, factor_cov: npt.ArrayLike) -> float:
        """Standard deviation ``sqrt(e' F e)`` of the CRB factor exposure ``e``."""
        e = self.factor_exposure(loadings)
        f = np.asarray(factor_cov, dtype=np.float64)
        return math.sqrt(max(float(e @ f @ e), 0.0))

    def netting_report(self, cost: CostModel) -> NettingReport:
        """Cost to flatten each desk alone versus flattening the CRB net."""
        gross = self.gross()
        stand = sum(cost.cost(i, g, self.marks[i]) for i, g in gross.items() if i in self.marks)
        net = self.net()
        crb = sum(cost.cost(i, q, self.marks[i]) for i, q in net.items() if i in self.marks)
        by_inst: dict[str, list[float]] = defaultdict(list)
        for f in self.fills:
            by_inst[f.instrument].append(f.qty)
        internal = {i: internalised_volume(by_inst.get(i, [])) for i in self.instruments}
        return NettingReport(
            standalone_cost=stand,
            crb_cost=crb,
            saved=stand - crb,
            internalised=internal,
            net=net,
            gross=gross,
        )
