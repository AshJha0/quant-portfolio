"""The old top-level ``fills`` and ``portfolio`` imports still work (shims)."""

from __future__ import annotations

import sys
from pathlib import Path


def test_shims_reexport_the_moved_packages(monkeypatch):
    import qp_bt.fills
    import qp_bt.portfolio

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    for name in ("fills", "portfolio"):
        sys.modules.pop(name, None)
    import fills
    import portfolio

    assert fills.QueueAwareFillModel is qp_bt.fills.QueueAwareFillModel
    assert portfolio.ledoit_wolf is qp_bt.portfolio.ledoit_wolf
    assert set(fills.__all__) == set(qp_bt.fills.__all__)
