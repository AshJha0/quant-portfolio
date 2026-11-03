"""Daily P&L explain for a book of fixed cashflows on a bootstrapped curve (W14).

The plain idea
--------------
Yesterday the curve was built from quotes ``q0``; today from ``q1``, and a
day has passed. The book's P&L splits into:

* carry: what the book earns if the curve's forwards are realised (nothing
  surprising happens);
* roll-down: the extra earned if instead the curve keeps its shape by tenor,
  so each cashflow "rolls down" to a shorter, usually lower, rate;
* market: the effect of the quote moves, explained by par-rate deltas and
  quote-space cross-gamma from the risk run;
* unexplained: full revaluation minus all of the above.

The formal statement
--------------------
With ``V(t, C)`` the value at time ``t`` of the flows after ``t`` off curve
``C`` (curve times measured from ``t``), ``K`` the flows paid in
``(t0, t1]`` and ``C(q)`` the bootstrap of quotes ``q``::

    HPL       = V(t1, C(q1)) + K - V(t0, C(q0))
    carry     = V_fwd(t1)   + K - V(t0, C(q0))      V_fwd: forwards of C(q0) realised
    roll-down = V(t1, C(q0)) - V_fwd(t1)
    market    = g_q . dq + 0.5 dq^T Gamma_q dq      at (t1, C(q0)), dq = q1 - q0
    unexplained = HPL - carry - roll-down - market

``g_q`` and ``Gamma_q`` come from :mod:`fi_rates.risk.par`. Cash ``K`` is
counted at face value (no intraday reinvestment). The decomposition of the
time effect follows Andersen and Piterbarg vol. 3 (2010) section 26.6;
HPL and RTPL follow BCBS MAR32 (static positions, risk-engine Greeks).

Worked example
--------------
See ``tests/test_risk_explain.py::test_flat_curve_carry_is_the_overnight_rate``:
a single cashflow of 100 at 5y on a flat 4% curve that does not move earns
``100 e^{-0.04 (5 - dt)} - 100 e^{-0.04 x 5}`` of carry and no roll-down
(on a flat curve the forwards and the tenor-shape coincide), and the
market and unexplained terms are zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qp_core.risk import PnLExplain, explain

from ..bootstrap import Instrument, _validate, bootstrap_curve
from ..curve import DiscountCurve
from .par import par_rate_gamma, par_rate_risk, quote_jacobian

__all__ = ["CashflowBook", "explain_curve_day"]


@dataclass(frozen=True)
class CashflowBook:
    """Fixed cashflows ``amounts[i]`` at ``times[i]`` (years from an origin)."""

    times: tuple[float, ...]
    amounts: tuple[float, ...]

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        a = np.asarray(self.amounts, dtype=float)
        if t.shape != a.shape or t.ndim != 1:
            raise ValueError("times and amounts must be 1-D of equal length")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(a))):
            raise ValueError("cashflow times and amounts must be finite")

    def pv(self, curve: DiscountCurve, t: float) -> float:
        """Value at ``t`` of the flows strictly after ``t``."""
        tt = np.asarray(self.times) - t
        live = tt > 0
        if not np.any(live):
            return 0.0
        return float(np.sum(np.asarray(self.amounts)[live] * np.asarray(curve.df(tt[live]))))

    def forward_pv(self, curve: DiscountCurve, t0: float, t1: float) -> float:
        """Value at ``t1`` if the forwards of ``curve`` (built at ``t0``) are realised."""
        dt = t1 - t0
        tt = np.asarray(self.times) - t0
        live = tt > dt
        if not np.any(live):
            return 0.0
        df_dt = float(np.asarray(curve.df(dt)))
        dfs = np.asarray(curve.df(tt[live])) / df_dt
        return float(np.sum(np.asarray(self.amounts)[live] * dfs))

    def paid(self, t0: float, t1: float) -> float:
        t = np.asarray(self.times)
        return float(np.sum(np.asarray(self.amounts)[(t > t0) & (t <= t1)]))

    def __add__(self, other: CashflowBook) -> CashflowBook:
        return CashflowBook(self.times + other.times, self.amounts + other.amounts)


def explain_curve_day(
    book: CashflowBook,
    quotes0: Sequence[Instrument],
    quotes1: Sequence[Instrument],
    t0: float,
    t1: float,
    interpolation: str = "loglinear_df",
    *,
    new_trades: float = 0.0,
    with_gamma: bool = True,
    curves: tuple[DiscountCurve, DiscountCurve] | None = None,
) -> PnLExplain:
    """Explain one period's P&L of ``book`` (see the module docstring).

    ``quotes0`` and ``quotes1`` must be the same instruments (same pillars)
    with possibly different rates; risk factors are labelled by pillar.
    ``curves`` passes the two bootstrapped curves when the caller already
    has them (a daily loop builds each date's curve once); they must
    reprice their quotes, which :func:`fi_rates.risk.par.par_rate_risk`
    checks for ``c0``.
    """
    if not t1 > t0:
        raise ValueError(f"require t1 > t0, got {t0}, {t1}")
    q0 = _validate(list(quotes0))
    q1 = _validate(list(quotes1))
    if [i.pillar for i in q0] != [i.pillar for i in q1] or [type(i) for i in q0] != [type(i) for i in q1]:
        raise ValueError("quotes0 and quotes1 must be the same instruments")
    if curves is None:
        c0 = bootstrap_curve(q0, interpolation)
        c1 = bootstrap_curve(q1, interpolation)
    else:
        c0, c1 = curves
    pv0 = book.pv(c0, t0)
    cash = book.paid(t0, t1)
    hpl = book.pv(c1, t1) + cash - pv0
    v_fwd = book.forward_pv(c0, t0, t1)
    rolled = book.pv(c0, t1)

    def pv_t1(curve: DiscountCurve) -> float:
        return book.pv(curve, t1)

    jac = quote_jacobian(q0, c0)
    delta = par_rate_risk(pv_t1, q0, c0, jacobian=jac)
    gamma = (
        par_rate_gamma(pv_t1, q0, c0, jacobian=jac, quote_risk=delta) if with_gamma else None
    )
    dq = np.array([b.rate - a.rate for a, b in zip(q0, q1, strict=True)])
    labels = [f"{type(i).__name__}:{i.pillar:g}y" for i in q0]
    return explain(
        hpl=hpl,
        time={"carry": v_fwd + cash - pv0, "rolldown": rolled - v_fwd},
        delta=delta,
        move=dq,
        gamma=gamma,
        labels=labels,
        new_trades=new_trades,
    )
