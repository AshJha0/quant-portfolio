"""Sequential curve bootstrapping from market quotes.

Instruments (all quote conventions simplified and documented in
``docs/METHODOLOGY.md``):

* :class:`Deposit` — simple interest over ``[0, T]``:
  ``P(T) = 1 / (1 + r * T)``.  Closed form.
* :class:`FRA` — forward simple rate over ``[T1, T2]``:
  ``P(T2) = P(T1) / (1 + r * (T2 - T1))``.  Requires ``P(T1)`` from
  already-bootstrapped pillars (interpolated if T1 is not a pillar).
* :class:`ParSwap` — par swap with an annual (or semi/quarterly) fixed leg,
  equal accruals ``1/frequency``:
  ``r * sum_i alpha * P(t_i) + P(T) = 1``.
  Solved by 1-D root search (Brent) on the new pillar's discount factor,
  because intermediate coupon dates are interpolated off the partially built
  curve — the standard sequential bootstrap.

Bond-curve bootstrap: :func:`bootstrap_bond_curve` solves each maturity
pillar's discount factor so the quoted coupon bond reprices exactly (dirty
price), again with intermediate cashflows interpolated.

Round-trip guarantee: with matching interpolation, every input instrument
reprices to its quote within 1e-10 (tested in ``tests/test_bootstrap.py``).

Times are year fractions (ACT/365F when derived from dates); rates are
simple/par rates in decimal (0.05 = 5%).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curve import DiscountCurve, ExtrapolationWarning

__all__ = [
    "Deposit",
    "FRA",
    "ParSwap",
    "bootstrap_curve",
    "bootstrap_bond_curve",
    "reprice_instruments",
]

_DF_LO, _DF_HI = 1e-8, 20.0  # brentq bracket for a pillar discount factor


@dataclass(frozen=True)
class Deposit:
    """Money-market deposit: simple interest, ``P(T) = 1/(1 + rate * maturity)``.

    ``maturity`` in years, ``rate`` decimal simple annualised.
    """

    maturity: float
    rate: float

    @property
    def pillar(self) -> float:
        return self.maturity


@dataclass(frozen=True)
class FRA:
    """Forward rate agreement paying simple rate over ``[start, end]`` (years)."""

    start: float
    end: float
    rate: float

    @property
    def pillar(self) -> float:
        return self.end


@dataclass(frozen=True)
class ParSwap:
    """Par swap: fixed leg pays ``rate / frequency`` on a unit notional at
    ``1/frequency, 2/frequency, ..., maturity``; the (implicit) float leg is
    worth par — single-curve simplification (see METHODOLOGY assumptions)."""

    maturity: float
    rate: float
    frequency: int = 1

    @property
    def pillar(self) -> float:
        return self.maturity

    def payment_times(self) -> np.ndarray:
        n = int(round(self.maturity * self.frequency))
        if n < 1 or abs(n / self.frequency - self.maturity) > 1e-9:
            raise ValueError(
                f"swap maturity {self.maturity} is not a whole number of "
                f"periods at frequency {self.frequency}"
            )
        return np.arange(1, n + 1) / self.frequency


Instrument = Deposit | FRA | ParSwap


def _validate(instruments: list[Instrument]) -> list[Instrument]:
    if not instruments:
        raise ValueError("no instruments supplied to bootstrap")
    for ins in instruments:
        if isinstance(ins, Deposit) and ins.maturity <= 0:
            raise ValueError(f"deposit maturity must be > 0, got {ins.maturity}")
        if isinstance(ins, FRA) and not 0 <= ins.start < ins.end:
            raise ValueError(
                f"FRA requires 0 <= start < end, got [{ins.start}, {ins.end}]"
            )
        if isinstance(ins, ParSwap):
            if ins.maturity <= 0:
                raise ValueError(f"swap maturity must be > 0, got {ins.maturity}")
            if ins.frequency not in (1, 2, 4):
                raise ValueError(
                    f"swap frequency must be 1, 2 or 4, got {ins.frequency}"
                )
    ordered = sorted(instruments, key=lambda i: i.pillar)
    pillars = [i.pillar for i in ordered]
    for a, b in zip(pillars, pillars[1:]):
        if abs(a - b) < 1e-12:
            raise ValueError(
                f"duplicate pillar maturity {a}: each instrument must add a "
                "distinct pillar"
            )
    return ordered


def _pv_error(ins: Instrument, curve: DiscountCurve) -> float:
    """Pricing error: model par/forward rate minus quoted rate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        if isinstance(ins, Deposit):
            p = float(np.asarray(curve.df(ins.maturity)))
            model = (1.0 / p - 1.0) / ins.maturity
            return model - ins.rate
        if isinstance(ins, FRA):
            model = curve.simple_forward_rate(ins.start, ins.end)
            return model - ins.rate
        model = curve.par_rate(ins.maturity, ins.frequency)
        return model - ins.rate


def bootstrap_curve(
    instruments: list[Instrument],
    interpolation: str = "loglinear_df",
) -> DiscountCurve:
    """Sequentially bootstrap a discount curve.

    Instruments are sorted by pillar maturity (the result is therefore
    independent of input order); each adds one pillar whose discount factor is
    solved so the instrument reprices exactly.

    Raises
    ------
    ValueError
        On empty input, duplicate pillars, invalid instrument definitions, or
        when no discount factor in ``(1e-8, 20)`` reprices an instrument
        (unsolvable input — e.g. wildly inconsistent quotes).
    """
    ordered = _validate(instruments)
    times: list[float] = []
    dfs: list[float] = []
    for ins in ordered:
        t_new = ins.pillar

        def objective(df_new: float, _ins: Instrument = ins) -> float:
            trial = DiscountCurve(times + [t_new], dfs + [df_new], interpolation)
            return _pv_error(_ins, trial)

        try:
            lo, hi = objective(_DF_LO), objective(_DF_HI)
            if lo * hi > 0:
                raise ValueError("no sign change in bracket")
            df_solved = brentq(objective, _DF_LO, _DF_HI, xtol=1e-16, rtol=1e-15)
        except ValueError as exc:
            raise ValueError(
                f"bootstrap failed at pillar t={t_new:.4f}y for {ins!r}: "
                f"no discount factor in ({_DF_LO}, {_DF_HI}) reprices the "
                f"quote (rate={ins.rate:+.4%}). Check quote consistency "
                f"(e.g. implied forward rates may be impossible). [{exc}]"
            ) from exc
        times.append(t_new)
        dfs.append(float(df_solved))
    return DiscountCurve(times, dfs, interpolation)


@dataclass(frozen=True)
class BondQuote:
    """Coupon bond quote for bond-curve bootstrapping (time-based cashflows).

    ``maturity`` in years, ``coupon`` decimal annual coupon rate paid
    ``frequency`` times per year on unit face, ``dirty_price`` per unit face.
    """

    maturity: float
    coupon: float
    dirty_price: float
    frequency: int = 2

    @property
    def pillar(self) -> float:
        return self.maturity

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        n = int(round(self.maturity * self.frequency))
        if n < 1 or abs(n / self.frequency - self.maturity) > 1e-9:
            raise ValueError(
                f"bond maturity {self.maturity} is not a whole number of "
                f"periods at frequency {self.frequency}"
            )
        t = np.arange(1, n + 1) / self.frequency
        cf = np.full(n, self.coupon / self.frequency)
        cf[-1] += 1.0
        return t, cf


def bootstrap_bond_curve(
    quotes: list[BondQuote],
    interpolation: str = "loglinear_df",
) -> DiscountCurve:
    """Bootstrap a zero curve from coupon bond dirty prices.

    Sequential: bonds sorted by maturity; each maturity pillar's discount
    factor is solved so the bond's PV matches its dirty price, with earlier
    cashflows discounted off the partially built (interpolated) curve.
    """
    if not quotes:
        raise ValueError("no bond quotes supplied to bootstrap")
    ordered = sorted(quotes, key=lambda q: q.maturity)
    for a, b in zip(ordered, ordered[1:]):
        if abs(a.maturity - b.maturity) < 1e-12:
            raise ValueError(f"duplicate bond maturity {a.maturity}")
    times: list[float] = []
    dfs: list[float] = []
    for q in ordered:
        t_cf, cf = q.cashflows()

        def objective(df_new: float, _q: BondQuote = q) -> float:
            trial = DiscountCurve(
                times + [_q.maturity], dfs + [df_new], interpolation
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExtrapolationWarning)
                pv = float(np.sum(cf * np.asarray(trial.df(t_cf))))
            return pv - _q.dirty_price

        try:
            if objective(_DF_LO) * objective(_DF_HI) > 0:
                raise ValueError("no sign change in bracket")
            df_solved = brentq(objective, _DF_LO, _DF_HI, xtol=1e-16, rtol=1e-15)
        except ValueError as exc:
            raise ValueError(
                f"bond bootstrap failed at t={q.maturity:.4f}y "
                f"(coupon={q.coupon:.4%}, price={q.dirty_price:.6f}): "
                f"no admissible discount factor reprices the bond. [{exc}]"
            ) from exc
        times.append(q.maturity)
        dfs.append(float(df_solved))
    return DiscountCurve(times, dfs, interpolation)


def reprice_instruments(
    instruments: list[Instrument], curve: DiscountCurve
) -> list[tuple[Instrument, float]]:
    """Model-minus-quote rate error for each instrument (round-trip check)."""
    return [(ins, _pv_error(ins, curve)) for ins in instruments]
