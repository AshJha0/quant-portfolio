"""Put the research study modules on ``sys.path`` (each notebook starts with ``import _paths``)."""

from __future__ import annotations

import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
for _d in ("panel", "tools", "vix", "tsmom"):
    _p = str(RESEARCH / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
