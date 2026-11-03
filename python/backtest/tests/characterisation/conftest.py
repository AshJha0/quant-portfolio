"""Import the migrated projects from this checkout, never from an installed copy."""

from __future__ import annotations

import pathlib
import sys

_PY = pathlib.Path(__file__).resolve().parents[3]
for _rel in ("foundations/02-trading-signal-backtest", "equity/05-pairs-trading",
             "equity/07-portfolio-optimization", "equity/08-algo-execution",
             "equity/10-regime-switching"):
    _src = str(_PY / _rel / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
