"""Independent position keeper fed by the drop copy (W28c).

Plain idea: the venue sends a second copy of every execution to a process
that shares no code path with the OMS order handling. If the OMS and this
keeper disagree, one of them is wrong, and reconciliation says which fills
explain the difference.

It reads only drop-copy execution reports: (exec_id, account, symbol, side,
qty, px). It dedupes on exec_id (PossDup resends are normal on a drop copy
session) and never sees parents, children or risk state.

Source: FIA Best Practices (2024), "drop copy" section; FIX 4.4 Vol. 4
(ExecID uniqueness per session).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DropCopyFill:
    exec_id: str
    account: str
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    px: float


@dataclass
class PositionKeeper:
    positions: dict[tuple[str, str], float] = field(default_factory=dict)
    fills: dict[str, DropCopyFill] = field(default_factory=dict)
    duplicates: int = 0

    def on_fill(self, f: DropCopyFill) -> None:
        old = self.fills.get(f.exec_id)
        if old is not None:
            if old != f:
                raise ValueError(f"exec_id {f.exec_id} reused with different content")
            self.duplicates += 1
            return
        if f.side not in ("BUY", "SELL") or f.qty <= 0:
            raise ValueError(f"bad drop copy fill {f}")
        self.fills[f.exec_id] = f
        k = (f.account, f.symbol)
        self.positions[k] = self.positions.get(k, 0.0) + (f.qty if f.side == "BUY" else -f.qty)
