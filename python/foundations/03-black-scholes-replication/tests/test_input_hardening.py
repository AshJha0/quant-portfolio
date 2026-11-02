"""Non-finite inputs must raise ``ValueError`` -- never return a NaN price.

Every comparison with NaN is ``False``, so the sign guards in ``_d1_d2``
(``if sigma <= 0 or T <= 0``, ``if S <= 0 or K <= 0``) silently accepted a
NaN spot or NaN vol and ``call_price(nan, ...)`` returned ``nan``.  The
finiteness check now runs *before* the sign guards on every public
pricing, Greeks, implied-vol and Monte Carlo entry point.
"""

from __future__ import annotations

import math

import pytest
from eq_bs_replication.black_scholes import (
    call_greeks,
    call_price,
    implied_volatility,
    put_greeks,
    put_price,
)
from eq_bs_replication.monte_carlo import mc_call_price

BASE = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
NON_FINITE = [math.nan, math.inf, -math.inf]


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("arg", ["S", "K", "r", "sigma", "T"])
@pytest.mark.parametrize("fn", [call_price, put_price, call_greeks, put_greeks])
def test_pricers_and_greeks_refuse_non_finite(fn, arg, bad):
    kwargs = dict(BASE)
    kwargs[arg] = bad
    with pytest.raises(ValueError, match=f"{arg} must be finite"):
        fn(**kwargs)


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("arg", ["price", "S", "K", "r", "T"])
def test_implied_vol_refuses_non_finite(arg, bad):
    kwargs = dict(price=10.45, S=100.0, K=100.0, r=0.05, T=1.0)
    kwargs[arg] = bad
    with pytest.raises(ValueError, match=f"{arg} must be finite"):
        implied_volatility(**kwargs)


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("arg", ["S", "K", "r", "sigma", "T"])
def test_monte_carlo_refuses_non_finite(arg, bad):
    kwargs = dict(BASE)
    kwargs[arg] = bad
    with pytest.raises(ValueError, match=f"{arg} must be finite"):
        mc_call_price(**kwargs, n_paths=100, seed=0)


def test_clean_inputs_unchanged():
    # Hardening must not move a single valid number.
    assert call_price(**BASE) == pytest.approx(10.450583572185565, rel=1e-12)
    assert put_price(**BASE) == pytest.approx(5.573526022256971, rel=1e-12)


def test_error_names_the_offending_argument_not_a_downstream_symptom():
    with pytest.raises(ValueError) as ei:
        call_price(S=float("nan"), K=100.0, r=0.05, sigma=0.2, T=1.0)
    assert "S must be finite" in str(ei.value)
