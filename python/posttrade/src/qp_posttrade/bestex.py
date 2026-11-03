"""Best-execution report templates.

Status of the source rules, as of 2026-11 (dated, check before reuse):

- MiFID II RTS 28 ((EU) 2017/576, top-five venues report) was deleted from
  EU law by the 2024 MiFIR review (Directive (EU) 2024/790) and the FCA
  removed it in the UK (PS21/20). RTS 27 ((EU) 2017/575) was suspended.
- SEC Rule 605 was amended in 2024 (Release 34-99679); the size buckets
  below follow its share-based categories in simplified form.

So both outputs are *templates* showing the layout a desk would use, built
on synthetic fills and stamped "synthetic venue". They are not filings.

Both tables are built from the same per-fill costs as ``qp_posttrade.tca``,
and ``reconcile`` proves their totals equal the TCA totals.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from qp_posttrade import STAMP
from qp_posttrade.model import Fill
from qp_posttrade.tca import FillCost, TcaTotals, fill_costs

SIZE_BUCKETS = ((0, 100), (100, 500), (500, 2000), (2000, 10000), (10000, float("inf")))


@dataclass(frozen=True)
class Rts28Row:
    venue: str
    pct_volume: float
    pct_fills: float
    qty: float
    notional: float
    shortfall: float
    fees: float


@dataclass(frozen=True)
class Rule605Row:
    bucket: str
    fills: int
    qty: float
    avg_eff_spread_bps: float  # quantity-weighted, 2 * s * (p - m) / m
    spread: float


@dataclass(frozen=True)
class BestExReport:
    stamp: str
    rts28: list[Rts28Row]  # every venue, ranked by volume; "top five" = [:5]
    rule605: list[Rule605Row]


def build(fills: Iterable[Fill]) -> BestExReport:
    fs = list(fills)
    costs = [fill_costs(f) for f in fs]
    tq = sum(c.qty for c in costs) or 1.0
    n = len(costs) or 1
    by: dict[str, list[FillCost]] = defaultdict(list)
    for c in costs:
        by[c.venue].append(c)
    rts = [
        Rts28Row(
            v,
            100 * sum(c.qty for c in cs) / tq,
            100 * len(cs) / n,
            sum(c.qty for c in cs),
            sum(c.notional for c in cs),
            sum(c.shortfall for c in cs),
            sum(c.fee for c in cs),
        )
        for v, cs in by.items()
    ]
    rts.sort(key=lambda r: (-r.qty, r.venue))
    rows = []
    for lo, hi in SIZE_BUCKETS:
        sel = [(f, c) for f, c in zip(fs, costs, strict=True) if lo <= f.qty < hi]
        q = sum(f.qty for f, _ in sel)
        bps = (
            sum(f.qty * 2e4 * f.sign * (f.px - f.mid_at_fill) / f.mid_at_fill for f, _ in sel) / q
            if q
            else 0.0
        )
        label = f"{lo}-{int(hi) - 1}" if hi != float("inf") else f"{lo}+"
        rows.append(Rule605Row(label, len(sel), q, bps, sum(c.spread for _, c in sel)))
    return BestExReport(STAMP, rts, rows)


def reconcile(report: BestExReport, totals: TcaTotals, tol: float = 1e-6) -> list[str]:
    """Differences between the report and TCA totals; empty means reconciled."""
    diffs = []
    checks = [
        ("rts28.qty", sum(r.qty for r in report.rts28), totals.qty),
        ("rts28.notional", sum(r.notional for r in report.rts28), totals.notional),
        ("rts28.shortfall", sum(r.shortfall for r in report.rts28), totals.shortfall),
        ("rts28.fees", sum(r.fees for r in report.rts28), totals.fees),
        ("rule605.qty", sum(r.qty for r in report.rule605), totals.qty),
        ("rule605.fills", sum(r.fills for r in report.rule605), totals.fills),
        ("rule605.spread", sum(r.spread for r in report.rule605), totals.spread),
    ]
    for name, got, want in checks:
        if abs(got - want) > tol * max(1.0, abs(want)):
            diffs.append(f"{name}: report {got} vs TCA {want}")
    return diffs
