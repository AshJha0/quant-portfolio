"""The one seam to the W23 part 1 panel (``research/panel/``, lane r09-l2).

The combination needs four month x currency frames: carry, value and
momentum signals, and next-month financed excess returns, plus the
point-in-time ``live`` mask. This module asks the panel package for them by
name and does nothing else, so integration touches only this file if the
panel's names differ.

Expected interface (NEXT_RELEASE_PLAN W23 part 1): ``panel.build()`` returns
an object with attributes ``carry``, ``value``, ``momentum``, ``next_excess``
and ``live``. Until that lane merges, ``load`` raises ``PanelUnavailable``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

SIGNALS = ("carry", "value", "momentum")


class PanelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PanelInputs:
    signals: dict[str, pd.DataFrame]
    next_excess: pd.DataFrame
    live: pd.DataFrame | None


def from_object(p: object) -> PanelInputs:
    missing = [a for a in (*SIGNALS, "next_excess") if not hasattr(p, a)]
    if missing:
        raise PanelUnavailable(f"panel object lacks {missing}")
    return PanelInputs({k: getattr(p, k) for k in SIGNALS}, p.next_excess,  # type: ignore[attr-defined]
                       getattr(p, "live", None))


def load(**kwargs: object) -> PanelInputs:
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        # "research.panel", not "panel": research/lib/panel.py already owns that name.
        mod = importlib.import_module("research.panel")
        build = mod.build
    except (ImportError, AttributeError) as e:
        raise PanelUnavailable("research/panel (W23 part 1) is not merged yet") from e
    return from_object(build(**kwargs))
