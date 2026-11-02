"""Risk limits: a hierarchy of limits, utilisation, warnings and breaches.

Plain idea
----------
Every number this package produces is only useful against a limit. A desk
has a VaR limit, its books have their own, and a book may also carry
sensitivity limits (delta, vega) that apply to the absolute size of the
exposure whichever way it points. The limit check answers three questions
for each node: how much of the limit is used, is it past the early-warning
level, and is it in breach.

Formal statement
----------------
For a node with limit ``L > 0`` and measured value ``x``:
``utilisation = |x| / L`` for two-sided (sensitivity) limits and ``x / L``
for one-sided (VaR, ES, stress loss) limits. Status is ``breach`` if
utilisation > 1, ``warning`` if it is ``>= warning`` (default 0.8), else
``ok``. A parent's value is the value passed for it; when none is passed it
is the sum of its children's values. That is right for stress loss and
notional, but conservative for VaR and ES, which are subadditive; pass the
aggregated risk number for those. This is a policy convention, not a
regulatory formula, and follows the limit structure described in
Basel Committee, *Principles for effective risk data aggregation and risk
reporting* (BCBS 239, 2013), principle 9 (limits reported against
exposures).

Worked example: ``tests/test_limits.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from . import _r3checks as chk


@dataclass
class Limit:
    name: str
    limit: float
    two_sided: bool = False
    warning: float = 0.8
    children: list[Limit] = field(default_factory=list)

    def __post_init__(self):
        chk.scalar("limit", self.limit, lo=0.0, strict=True)
        w = chk.scalar("warning", self.warning, lo=0.0, strict=True)
        if w > 1.0:
            raise ValueError("warning must be in (0, 1]")


@dataclass(frozen=True)
class LimitStatus:
    name: str
    value: float
    limit: float
    utilisation: float
    status: str  # "ok" | "warning" | "breach"
    depth: int


def check_limits(root: Limit, values: Mapping[str, float]) -> list[LimitStatus]:
    """Evaluate every node of the tree, parents before children (pre-order).

    ``values`` maps node names to measured values. A leaf without a value
    raises ``ValueError``; a parent without one takes the sum of its
    children.
    """
    out: list[LimitStatus] = []

    def visit(node: Limit, depth: int) -> float:
        slot = len(out)
        out.append(None)  # type: ignore[arg-type]  # filled after the children
        child_sum = sum(visit(c, depth + 1) for c in node.children)
        if node.name in values:
            x = chk.scalar(f"values[{node.name!r}]", values[node.name])
        elif node.children:
            x = child_sum
        else:
            raise ValueError(f"no value for leaf limit {node.name!r}")
        u = (abs(x) if node.two_sided else x) / node.limit
        status = "breach" if u > 1.0 else "warning" if u >= node.warning else "ok"
        out[slot] = LimitStatus(node.name, x, node.limit, u, status, depth)
        return x

    visit(root, 0)
    return out


def breaches(statuses: list[LimitStatus]) -> list[LimitStatus]:
    """Only the nodes in breach."""
    return [s for s in statuses if s.status == "breach"]
