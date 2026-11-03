r"""Reconciliation and break reports (W28c).

Plain idea: line up the OMS's fills against an independent record (the drop
copy, or the acceptor's own book after a restart), fill by fill, then check
positions. Every difference is a break. A break is *explained* when its
exec_id is in the in-flight set (a fill the other side has not received
yet); everything else is *unexplained* and needs a person.

Formal statement. Let F_o and F_d be the fill maps exec_id -> (account,
symbol, signed qty, px). Breaks:

- MISSING_IN_OMS:      id in F_d \ F_o
- MISSING_IN_DROPCOPY: id in F_o \ F_d
- QTY_MISMATCH / PX_MISMATCH: id in both, fields differ (px tolerance 1e-9)
- POSITION_MISMATCH:   per (account, symbol), position differs by more than
  the sum of the fill breaks already reported for that key.

The last rule means a position break is only raised when the fill-level
breaks do not already account for it, so one missing fill gives one break,
not two.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

Fill = tuple[str, str, float, float]  # account, symbol, signed qty, px


@dataclass(frozen=True)
class Break:
    kind: str
    key: str
    left: str
    right: str
    explained: bool


def reconcile(
    oms_fills: dict[str, Fill],
    other_fills: dict[str, Fill],
    oms_pos: dict[tuple[str, str], float],
    other_pos: dict[tuple[str, str], float],
    in_flight: frozenset[str] = frozenset(),
    px_tol: float = 1e-9,
) -> list[Break]:
    out: list[Break] = []
    accounted: dict[tuple[str, str], float] = {}

    def note(f: Fill, sign: float) -> None:
        k = (f[0], f[1])
        accounted[k] = accounted.get(k, 0.0) + sign * f[2]

    for eid in sorted(set(oms_fills) | set(other_fills)):
        a, b = oms_fills.get(eid), other_fills.get(eid)
        exp = eid in in_flight
        if a is None and b is not None:
            out.append(Break("MISSING_IN_OMS", eid, "", repr(b), exp))
            note(b, -1.0)
        elif b is None and a is not None:
            out.append(Break("MISSING_IN_DROPCOPY", eid, repr(a), "", exp))
            note(a, 1.0)
        elif a is not None and b is not None:
            if a[:3] != b[:3]:
                out.append(Break("QTY_MISMATCH", eid, repr(a), repr(b), exp))
                note(a, 1.0)
                note(b, -1.0)
            elif abs(a[3] - b[3]) > px_tol:
                out.append(Break("PX_MISMATCH", eid, repr(a), repr(b), exp))
    for k in sorted(set(oms_pos) | set(other_pos)):
        d = oms_pos.get(k, 0.0) - other_pos.get(k, 0.0) - accounted.get(k, 0.0)
        if abs(d) > 1e-9:
            out.append(
                Break(
                    "POSITION_MISMATCH",
                    f"{k[0]}|{k[1]}",
                    str(oms_pos.get(k, 0.0)),
                    str(other_pos.get(k, 0.0)),
                    False,
                )
            )
    return out


def break_report(breaks: list[Break]) -> str:
    """JSON lines, unexplained first; the file a person works through."""
    rows = sorted(breaks, key=lambda b: (b.explained, b.kind, b.key))
    return "".join(json.dumps(asdict(b), sort_keys=True) + "\n" for b in rows)
