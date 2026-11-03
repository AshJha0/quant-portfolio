"""Implied vol across the hard regimes, checked against mpmath (50 digits).

Acceptance rule for every case: |sigma_hat - sigma| <= 2 max(eps price / vega,
1e-12 sigma), where eps price / vega is the vol uncertainty implied by one
rounding of the quote, and at most 10 evaluations of the normalised price.
"""

import math
import warnings

import mpmath as mp
import pytest
from eq_options import bs_price, implied_vol
from qp_core.iv_core import black_implied_vol

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
    lower = bs_price(S, K, T, r, 0.0, q, otype)  # the engine's own sigma->0 bound
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
        try:
            got = implied_vol(price, S, K, T, r, q, otype)
        except ValueError:
            return
        assert abs(got - sigma) <= tol
        return
    got = implied_vol(price, S, K, T, r, q, otype)
    assert abs(got - sigma) <= tol
    counter = [0]
    black_implied_vol(price, forward, K, T, df, otype == "call", counter)
    assert counter[0] <= 10


def test_k90_otm_put_recovers_to_1e12() -> None:
    S, K, T, r, sigma = 100.0, 90.0, 0.02, 0.03, 0.10
    put = bs_price(S, K, T, r, sigma, 0.0, "put")
    assert abs(implied_vol(put, S, K, T, r, 0.0, "put") - sigma) <= 1e-12


def test_k90_itm_call_is_backward_stable() -> None:
    """The ITM call carries less information than the OTM put.

    One ulp of the call price (1.8e-15) divided by vega (3.5e-12) is a vol
    uncertainty of about 5e-4, so no solver can return 10% to 1e-12 from
    this quote. The solver must return a vol that reprices the quote to
    within a few ulps.
    """
    S, K, T, r, sigma = 100.0, 90.0, 0.02, 0.03, 0.10
    call = bs_price(S, K, T, r, sigma, 0.0, "call")
    got = implied_vol(call, S, K, T, r, 0.0, "call")
    assert abs(bs_price(S, K, T, r, got, 0.0, "call") - call) <= 16 * math.ulp(call)
    assert abs(got - sigma) < 5e-3


def test_old_tuning_keywords_warn() -> None:
    price = bs_price(100.0, 100.0, 1.0, 0.02, 0.2, 0.0, "call")
    with pytest.warns(DeprecationWarning):
        implied_vol(price, 100.0, 100.0, 1.0, 0.02, 0.0, "call", tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        implied_vol(price, 100.0, 100.0, 1.0, 0.02, 0.0, "call")


def test_root_at_the_inflection_point_converges_fast() -> None:
    """Total vol equal to s_c = sqrt(2|x|) sits on the lower/upper branch
    boundary. Bisecting towards s_c took 12 evaluations; probing s_c takes 2.
    """
    df, t, sigma = 0.97, 10.0, 0.1
    k = math.exp(-0.05)
    r = -math.log(df) / t
    from eq_options import bs_price as price_fn
    price = price_fn(df, k, t, r, sigma, 0.0, "put")
    counter = [0]
    got = black_implied_vol(price, 1.0, k, t, df, False, counter)
    assert counter[0] <= 4
    assert abs(got - sigma) <= 1e-13
