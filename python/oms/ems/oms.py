"""OMS and EMS (W28c).

Plain idea: the OMS is the book of record for what the desk *wants* (parent
orders), what it *has* (positions) and *whose* it is (allocations). The EMS
turns a parent into child orders, passes each child through the kill switch
and pre-trade risk, and feeds fills back up to the parent.

Formal invariants, checked by the tests:

- for every parent, ``0 <= cum_qty <= qty`` and ``cum_qty`` equals the sum of
  its children's fills;
- an execution id is applied at most once (FIX ExecID dedupe);
- allocations of a parent's fills sum exactly to ``cum_qty`` (largest
  remainder, whole lots) and each account receives the parent's average price.

Seams: ``Slicer`` is the W3 scheduler (``qp_core`` execution); until W3 is
wired the default is an equal-size TWAP split. ``OrderIntent`` is the risk
view of the r12-l2-oms core ``Order``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from qp_oms.kill import KillSwitch
from qp_oms.risk import OrderIntent, RiskBook, Side

Slicer = Callable[[float, int], list[float]]


def twap_slices(qty: float, n: int) -> list[float]:
    """Equal whole-lot slices; the remainder goes to the first slices."""
    base, rem = divmod(int(qty), n)
    return [float(base + (1 if i < rem else 0)) for i in range(n) if base + (1 if i < rem else 0) > 0]


@dataclass(frozen=True)
class ExecReport:
    """Subset of FIX 35=8 the OMS needs."""

    exec_id: str
    cl_ord_id: str
    last_qty: float
    last_px: float
    terminal: bool = False  # OrdStatus filled/cancelled/rejected/expired


@dataclass
class ParentOrder:
    parent_id: str
    desk: str
    strategy: str
    symbol: str
    side: Side
    qty: float
    allocations: dict[str, float]  # account -> weight; weights sum to 1
    cum_qty: float = 0.0
    notional: float = 0.0
    children: list[str] = field(default_factory=list)

    @property
    def avg_px(self) -> float:
        return self.notional / self.cum_qty if self.cum_qty else 0.0


@dataclass(frozen=True)
class Child:
    intent: OrderIntent
    parent_id: str
    path: tuple[str, str, str, str]


class Oms:
    """Parents, positions and allocations; the EMS methods are on the same object."""

    def __init__(self, firm: str, risk: RiskBook, kill: KillSwitch, slicer: Slicer = twap_slices) -> None:
        self.firm = firm
        self.risk = risk
        self.kill = kill
        self.slicer = slicer
        self.parents: dict[str, ParentOrder] = {}
        self.children: dict[str, Child] = {}
        self.positions: dict[tuple[str, str], float] = {}  # (account, symbol)
        self.fills: dict[str, tuple[str, float, float]] = {}  # exec_id -> (cl_ord_id, qty, px)
        self._oid = 0
        self._child_num: dict[str, int] = {}
        self.sent: list[OrderIntent] = []
        self.rejects: list[tuple[str, str]] = []
        self._open: set[str] = set()  # children not yet terminal
        self._ids: dict[str, int] = {}
        self._cids: dict[int, str] = {}

    def new_parent(self, p: ParentOrder) -> None:
        if abs(sum(p.allocations.values()) - 1.0) > 1e-9:
            raise ValueError("allocation weights must sum to 1")
        if p.parent_id in self.parents:
            raise ValueError(f"duplicate parent {p.parent_id}")
        self.parents[p.parent_id] = p

    # ---- EMS ------------------------------------------------------------
    def work(
        self, parent_id: str, n_slices: int, px: float, now_ns: int, session: str = "S1"
    ) -> list[OrderIntent]:
        """Create children for the unworked quantity; send those risk accepts."""
        p = self.parents[parent_id]
        live = sum(
            c.intent.qty for k, c in self.children.items() if c.parent_id == parent_id and k in self._open
        )
        todo = p.qty - p.cum_qty - live
        out: list[OrderIntent] = []
        if todo <= 0:
            return out
        path = (self.firm, p.desk, p.strategy, p.symbol)
        for q in self.slicer(todo, n_slices):
            n = self._child_num.get(parent_id, 0) + 1
            self._child_num[parent_id] = n
            cid = f"{parent_id}.{n}"
            # Child orders trade in the desk's house account; allocation is post-trade.
            o = OrderIntent(cid, f"{self.firm}:{p.desk}", session, p.symbol, p.side, q, px, parent_id)
            reason = self.kill.check(path) or self.risk.check_new(o, now_ns)
            if reason is not None:
                self.rejects.append((cid, reason))
                continue
            self.children[cid] = Child(o, parent_id, path)
            self._open.add(cid)
            p.children.append(cid)
            self.kill.on_order_working(self._num(cid), path)
            self.sent.append(o)
            out.append(o)
        return out

    def _num(self, cid: str) -> int:
        """Stable integer id for the kill switch (the contract uses integers)."""
        if cid not in self._ids:
            self._ids[cid] = len(self._ids) + 1
            self._cids[self._ids[cid]] = cid
        return self._ids[cid]

    def cid_for(self, num: int) -> str:
        return self._cids[num]

    def kill_and_cancel(self, scope: tuple[str, str, str, str], actor: str) -> list[str]:
        """Engage a kill; return the child ids to cancel (mass cancel)."""
        _, nums = self.kill.engage(scope, actor)
        return [self.cid_for(n) for n in nums]

    def on_exec(self, er: ExecReport) -> bool:
        """Apply an execution report once. Returns False for a duplicate."""
        if er.exec_id in self.fills:
            return False
        c = self.children.get(er.cl_ord_id)
        if c is None:
            raise KeyError(f"exec for unknown order {er.cl_ord_id}")
        self.fills[er.exec_id] = (er.cl_ord_id, er.last_qty, er.last_px)
        if er.last_qty > 0:
            p = self.parents[c.parent_id]
            if p.cum_qty + er.last_qty > p.qty + 1e-9:
                raise ValueError(f"overfill on parent {p.parent_id}")
            p.cum_qty += er.last_qty
            p.notional += er.last_qty * er.last_px
            sign = 1.0 if p.side is Side.BUY else -1.0
            k = (c.intent.account, c.intent.symbol)
            self.positions[k] = self.positions.get(k, 0.0) + sign * er.last_qty
            if er.cl_ord_id in self._open:
                self.risk.on_fill(er.cl_ord_id, er.last_qty)
        if er.terminal and er.cl_ord_id in self._open:
            self._open.discard(er.cl_ord_id)
            self.risk.on_terminal(er.cl_ord_id)
            self.kill.on_order_terminal(self._num(er.cl_ord_id))
        return True

    # ---- allocation -----------------------------------------------------
    def allocate(self, parent_id: str) -> dict[str, tuple[float, float]]:
        """Split cum_qty across accounts in whole lots: account -> (qty, avg px)."""
        p = self.parents[parent_id]
        total = int(round(p.cum_qty))
        raw = {a: w * total for a, w in p.allocations.items()}
        base = {a: int(v) for a, v in raw.items()}
        left = total - sum(base.values())
        for a in sorted(raw, key=lambda a: (-(raw[a] - base[a]), a))[:left]:
            base[a] += 1
        return {a: (float(q), p.avg_px) for a, q in base.items()}
