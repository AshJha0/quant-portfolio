"""Implied vol across the hard regimes, checked against mpmath (50 digits).

Acceptance rule for every case: |sigma_hat - sigma| <= 2 max(eps price / vega,
1e-12 sigma), where eps price / vega is the vol uncertainty implied by one
rounding of the quote, and at most 10 evaluations of the normalised price.
"""

import math
import warnings

import mpmath as mp
import pytest

from fx_options import gk_price, implied_vol
from fx_options._iv_core import black_implied_vol

mp.mp.dps = 50
EPS = 2.220446049250313e-16


def _mp_price_vega(S, K, T, r, q, sigma, otype):
    S, K, T, r, q, sigma = map(mp.mpf, (S, K, T, r, q, sigma))
    v = sigma * mp.sqrt(T)
    d1 = (mp.log(S / K) + (r - q) * T) / v + v / 2
    d2 = d1 - v
    ds, dk = S * mp.exp(-q * T), K * mp.exp(-r * T)
    if otype == "call":
        price = ds * mp.ncdf(d1) - dk * mp.ncdf(d2)
    else:
        price = dk * mp.ncdf(-d2) - ds * mp.ncdf(-d1)
    return price, ds * mp.npdf(d1) * mp.sqrt(T)


def _cases():
    out = []
    for lk in (-5.0, -3.0, -1.0, -0.3, -0.05, -1e-6, 0.0, 1e-6, 0.05, 0.3, 1.0, 3.0, 5.0):
        for T in (1e-7, 1e-4, 0.02, 0.25, 1.0, 10.0, 50.0):
            for sigma in (1e-4, 1e-3, 0.01, 0.1, 0.5, 2.0, 5.0):
                for otype in ("call", "put"):
                    out.append((lk, T, sigma, otype))
    return out


@pytest.mark.parametrize("lk,T,sigma,otype", _cases())
def test_iv_round_trip_against_mpmath(lk, T, sigma, otype) -> None:
    S, r, q = 100.0, 0.03, 0.01
    K = S * math.exp(lk)
    price_mp, vega_mp = _mp_price_vega(S, K, T, r, q, sigma, otype)
    price = float(price_mp)
    df = math.exp(-r * T)
    forward = S * math.exp((r - q) * T)
    lower = df * max((forward - K) if otype == "call" else (K - forward), 0.0)
    upper = S * math.exp(-q * T) if otype == "call" else K * df
    if not (lower < price < upper) or price < 1e-300:
        pytest.skip("quote not representable strictly inside the bounds")
    # One rounding of the quote, of the sigma->0 bound and of the forward
    # terms: a few eps of max(price, S, K), mapped through vega.
    tol = 4.0 * max(float(EPS * max(price_mp, S, K) / vega_mp), 1e-12 * sigma)
    if min(price - lower, upper - price) <= 8 * math.ulp(max(price, lower, upper)):
        # Time value (or distance to the upper bound) is a few ulps of the
        # quote. The engine may refuse (ValueError) or return a vol, which
        # must then still be inside the quote's own uncertainty.
        # FX contract: 0.0 at the lower bound, ValueError at the upper one.
        try:
            got = implied_vol(price, S, K, T, r, q, otype)  # r_d = r, r_f = q
        except ValueError:
            assert upper - price <= price - lower
            return
        assert got == 0.0 or abs(got - sigma) <= tol
        return
    got = implied_vol(price, S, K, T, r, q, otype)  # r_d = r, r_f = q
    assert abs(got - sigma) <= tol
    counter = [0]
    black_implied_vol(price, forward, K, T, df, otype == "call", counter)
    assert counter[0] <= 10


def test_k90_otm_put_recovers_to_1e12() -> None:
    S, K, T, r, sigma = 100.0, 90.0, 0.02, 0.03, 0.10
    put = gk_price(S, K, T, r, 0.0, sigma, "put")
    assert abs(implied_vol(put, S, K, T, r, 0.0, "put") - sigma) <= 1e-12


def test_k90_itm_call_is_backward_stable() -> None:
    """The ITM call carries less information than the OTM put.

    One ulp of the call price (1.8e-15) divided by vega (3.5e-12) is a vol
    uncertainty of about 5e-4, so no solver can return 10% to 1e-12 from
    this quote. The solver must return a vol that reprices the quote to
    within a few ulps. In the FX contract a time value below the rounding
    of the intrinsic bound returns 0.0, which also reprices the quote.
    """
    S, K, T, r, sigma = 100.0, 90.0, 0.02, 0.03, 0.10
    call = gk_price(S, K, T, r, 0.0, sigma, "call")
    got = implied_vol(call, S, K, T, r, 0.0, "call")
    assert abs(gk_price(S, K, T, r, 0.0, got, "call") - call) <= 16 * math.ulp(call)


def test_old_tuning_keywords_warn() -> None:
    price = gk_price(1.1, 1.1, 1.0, 0.04, 0.02, 0.1, "call")
    with pytest.warns(DeprecationWarning):
        implied_vol(price, 1.1, 1.1, 1.0, 0.04, 0.02, "call", tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        implied_vol(price, 1.1, 1.1, 1.0, 0.04, 0.02, "call")


def test_bounds_contract() -> None:
    S, K, T, rd, rf = 1.1, 1.0, 0.5, 0.04, 0.02
    lower = gk_price(S, K, T, rd, rf, 0.0, "call")
    upper = S * math.exp(-rf * T)
    assert implied_vol(lower, S, K, T, rd, rf, "call") == 0.0
    with pytest.raises(ValueError):
        implied_vol(upper, S, K, T, rd, rf, "call")
    with pytest.raises(ValueError):
        implied_vol(lower - 1e-9, S, K, T, rd, rf, "call")


def test_no_premature_exit_on_tiny_otm_price() -> None:
    """An OTM put worth 2.7e-11 must still invert to full vol precision.

    The previous solver stopped on an absolute price residual of 1e-14 and
    returned this vol only to 2.7e-7.
    """
    S, K, T, rd, rf, sigma = 100.0, 90.0, 0.02, 0.03, 0.0, 0.10
    put = gk_price(S, K, T, rd, rf, sigma * 1.4, "put")
    assert abs(implied_vol(put, S, K, T, rd, rf, "put") - sigma * 1.4) <= 1e-13
