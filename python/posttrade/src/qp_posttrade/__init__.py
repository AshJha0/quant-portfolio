"""Post-trade, best execution and surveillance (W29).

Every report this package produces is stamped ``SYNTHETIC_VENUE``: the fills,
venues and counterparties are simulated, never real.
"""

from __future__ import annotations

STAMP = "synthetic venue"

__all__ = ["STAMP"]
