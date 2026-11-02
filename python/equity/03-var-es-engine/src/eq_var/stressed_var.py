"""Stressed VaR (Basel 2.5): stress-window search and the capital formula.

Plain idea
----------
Ordinary VaR forgets a crisis once it leaves the look-back window. Stressed
VaR revalues today's book on the worst continuous 12-month period in the
bank's history, so the capital never falls below what a known stress would
cost.

Formal statement
----------------
BCBS, *Revisions to the Basel II market risk framework* (July 2009), para.
718(Lxxvi):

    capital = max(VaR_{t-1}, m_c avg60(VaR)) + max(sVaR_{t-1}, m_s avg60(sVaR))

with ``m_c, m_s >= 3`` (plus the backtesting add-on). The stress window is
chosen to maximise the VaR of the current book; we search every window of
``window`` consecutive scenario days and pick the one whose historical VaR of
the current-book P&L is largest (ties go to the earliest window).

Worked example: ``tests/test_stressed_var.py::test_planted_stress_window``.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import _r3checks as chk


@dataclass(frozen=True)
class StressWindow:
    start: int  # first scenario index of the window
    stop: int  # one past the last index
    var: float
    es: float


def find_stress_window(pnl, window: int = 250, alpha: float = 0.01) -> StressWindow:
    """Window of ``window`` consecutive days with the largest historical VaR.

    ``pnl`` is the current book revalued on every historical day (oldest
    first). Returns the window and its VaR and ES at tail ``alpha``.
    """
    a = chk.alpha_(alpha)
    w = chk.count("window", window, lo=10)
    p = chk.array("pnl", pnl, min_size=w)
    best: StressWindow | None = None
    for s in range(p.size - w + 1):
        v, e = chk.loss_quantile(p[s:s + w], a)
        if best is None or v > best.var:
            best = StressWindow(s, s + w, v, e)
    assert best is not None
    return best


def stressed_var(pnl, window: int = 250, alpha: float = 0.01) -> float:
    """Historical VaR of the current book on its worst ``window``-day period."""
    return find_stress_window(pnl, window, alpha).var


def basel25_capital(var_history, svar_history, m_c: float = 3.0, m_s: float = 3.0) -> float:
    """Para. 718(Lxxvi) capital from the last 60 daily VaR and sVaR (latest last)."""
    v = chk.array("var_history", var_history, min_size=60)[-60:]
    s = chk.array("svar_history", svar_history, min_size=60)[-60:]
    mc = chk.scalar("m_c", m_c, lo=3.0)
    ms = chk.scalar("m_s", m_s, lo=3.0)
    return float(max(v[-1], mc * v.mean()) + max(s[-1], ms * s.mean()))
