"""Make the repository's library packages importable from a plain checkout.

Each notebook starts with ``import _paths``. If the packages are already
installed (``pip install -e python/equity/01-options-pricing`` and so on),
this changes nothing; otherwise it puts their ``src`` folders on
``sys.path``. Notebooks use library code only, so this is the whole setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "python/core",
    "python/equity/01-options-pricing",
    "python/equity/03-var-es-engine",
    "python/equity/04-fixed-income",
    "python/equity/08-algo-execution",
)

for _pkg in PACKAGES:
    _src = str(ROOT / _pkg / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
