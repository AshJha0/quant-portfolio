"""W14 acceptance: P&L explain over 250 real days passes FRTB PLA in the green zone.

Data (``tests/data/gsw_ecb_vix_2022_2023.csv``, 286 aligned days,
2022-11-01 to 2023-12-29), all free and keyless, retrieved 2026-11-03:

* Gurkaynak-Sack-Wright Svensson parameters BETA0-3, TAU1-2 and the published
  SVENY02/SVENY10 zero yields (Federal Reserve, feds200628.csv);
* ECB EURUSD reference rate (EXR D.USD.EUR.SP00.A);
* CBOE VIX close.

The book (a test fixture, not a desk position): three fixed-coupon bonds
priced off a curve bootstrapped each day from 1-30y annual par rates
implied by the GSW curve, plus a long 1y EURUSD call. The VIX level is used
as the call's implied vol: a stated proxy (VIX is S&P 500 implied vol, not
EURUSD vol), chosen because it is a free daily implied-vol series; the
point is a real, fat-tailed vol path, not an FX vol model. The EUR rate is
held at 3% (stated assumption; only the USD curve is real).

HPL is full revaluation of the start-of-day book at end-of-day market.
RTPL is the time effect plus par-rate delta and quote-space cross-gamma for
the bonds, and delta, gamma, vega, vanna, volga and USD-rate sensitivity for
the option. PLA (BCBS MAR32.42-32.44): Spearman >= 0.80 and KS <= 0.09 over
the last 250 days is green.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest
from fi_rates.bootstrap import ParSwap, bootstrap_curve
from fi_rates.risk import CashflowBook, explain_curve_day
from qp_core.black_scholes import carry_greeks
from qp_core.risk import cross_gamma, explain, fd_gradient
from scipy import stats

DATA = Path(__file__).parent / "data" / "gsw_ecb_vix_2022_2023.csv"
TENORS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
R_EUR = 0.03
PLA_SPEARMAN_GREEN, PLA_KS_GREEN = 0.80, 0.09  # eq_var/frtb_ima.py:64-65


def _rows():
    with DATA.open(newline="") as fh:
        return [
            {k: (v if k == "date" else float(v)) for k, v in r.items()}
            for r in csv.DictReader(fh)
        ]


def svensson_zero(p: dict, n: float) -> float:
    """GSW (2007) eq. 22: continuously compounded zero yield in percent."""
    b0, b1, b2, b3, t1, t2 = (p[k] for k in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"))
    x1, x2 = n / t1, n / t2
    f1 = (1 - math.exp(-x1)) / x1
    f2 = (1 - math.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - math.exp(-x1)) + b3 * (f2 - math.exp(-x2))


def _par_quotes(p: dict) -> list[ParSwap]:
    def df(n: float) -> float:
        return math.exp(-svensson_zero(p, n) / 100.0 * n)

    out = []
    for t in TENORS:
        ann = sum(df(float(i)) for i in range(1, int(t) + 1))
        out.append(ParSwap(t, (1.0 - df(t)) / ann))
    return out


def _bond(coupon: float, years: int, notional: float) -> CashflowBook:
    times = tuple(float(i) for i in range(1, years + 1))
    amounts = (*tuple(notional * coupon for _ in times[:-1]), notional * (1 + coupon))
    return CashflowBook(times, amounts)


BOOK = _bond(0.035, 10, 1e6) + _bond(0.04, 5, -2e6) + _bond(0.03, 30, 5e5)
CALL_K, CALL_T, CALL_N = 1.10, 1.25, 2e6


def _call_value(x: np.ndarray, t: float) -> float:
    spot, vol, r_usd = x
    tau = CALL_T - t
    return CALL_N * carry_greeks(spot, CALL_K, tau, r_usd, R_EUR, vol, True).price


@pytest.fixture(scope="module")
def daily():
    rows = _rows()
    d0 = dt.date.fromisoformat(rows[0]["date"])
    quotes = [_par_quotes(r) for r in rows]
    curves = [bootstrap_curve(q) for q in quotes]
    hpl, rtpl, unexplained = [], [], []
    for k, (a, b) in enumerate(zip(rows, rows[1:], strict=False)):
        t0 = (dt.date.fromisoformat(a["date"]) - d0).days / 365.0
        t1 = (dt.date.fromisoformat(b["date"]) - d0).days / 365.0
        rates = explain_curve_day(BOOK, quotes[k], quotes[k + 1], t0, t1,
                                  curves=(curves[k], curves[k + 1]))
        x0 = np.array([a["EURUSD"], a["VIX"] / 100, svensson_zero(a, CALL_T - t0) / 100])
        x1 = np.array([b["EURUSD"], b["VIX"] / 100, svensson_zero(b, CALL_T - t1) / 100])
        v0 = _call_value(x0, t0)
        opt_hpl = _call_value(x1, t1) - v0

        def at_t1(x, _t1=t1):
            return _call_value(x, _t1)

        opt = explain(
            hpl=opt_hpl,
            time={"theta": at_t1(x0) - v0},
            delta=fd_gradient(at_t1, x0, h=1e-6),
            move=x1 - x0,
            gamma=cross_gamma(at_t1, x0, h=1e-4),
            labels=["spot", "vol", "r_usd"],
        )
        hpl.append(rates.hpl + opt.hpl)
        rtpl.append(rates.rtpl + opt.rtpl)
        unexplained.append(rates.unexplained + opt.unexplained)
    return np.array(hpl), np.array(rtpl), np.array(unexplained)


def test_fixture_reproduces_published_gsw_zero_yields():
    # Independent check of the curve formula: GSW publish SVENY to 4 dp.
    for r in _rows():
        assert svensson_zero(r, 2.0) == pytest.approx(r["SVENY02"], abs=6e-5)
        assert svensson_zero(r, 10.0) == pytest.approx(r["SVENY10"], abs=6e-5)


def test_daily_curve_reprices_its_quotes():
    from fi_rates.bootstrap import reprice_instruments

    q = _par_quotes(_rows()[100])
    assert max(abs(e) for _, e in reprice_instruments(q, bootstrap_curve(q))) < 1e-10


def test_pla_green_over_250_real_days(daily):
    hpl, rtpl, unexplained = daily
    assert hpl.size >= 250
    h, r = hpl[-250:], rtpl[-250:]
    rho = stats.spearmanr(h, r).statistic
    ks = stats.ks_2samp(h, r).statistic
    assert rho >= PLA_SPEARMAN_GREEN and ks <= PLA_KS_GREEN, (rho, ks)
    # Measured 2026-11-03: Spearman 0.999997, KS 0.008; the unexplained P&L is a small
    # fraction of the HPL spread.
    assert np.std(unexplained[-250:]) < 0.005 * np.std(h)  # measured 0.00084
    try:  # the repository's own PLA test, when eq_var is installed
        from eq_var.frtb_ima import pla_test
    except ImportError:  # pragma: no cover - fi_rates-only environment
        return
    assert pla_test(h, r).zone == "green"
