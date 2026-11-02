"""Left-tail CDF accuracy and in-the-money implied vol.

Both tests fail on the erf-based CDF and the call-only inversion that
preceded them.
"""
import math

import pytest
from eq_bs_replication import call_price, implied_volatility, put_price
from eq_bs_replication.black_scholes import _norm_cdf
from hypothesis import assume, given, settings
from hypothesis import strategies as st


# Reference values of N(x) from the asymptotic series
# N(-x) = phi(x)/x * (1 - 1/x^2 + 3/x^4 - 15/x^6 + 105/x^8 - 945/x^10),
# truncation error below 1e-9 relative for x >= 8.
def _tail_reference(x: float) -> float:
    phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    s = 1 - 1 / x**2 + 3 / x**4 - 15 / x**6 + 105 / x**8 - 945 / x**10
    return phi / x * s


@pytest.mark.parametrize("x", [8.0, 9.0, 12.0, 20.0, 37.0])
def test_norm_cdf_left_tail_keeps_relative_accuracy(x):
    assert _norm_cdf(-x) == pytest.approx(_tail_reference(x), rel=1e-8)


def test_deep_otm_call_is_positive_not_zero():
    # S=100, K=300, T=0.25, sigma=0.2: d2 is about -11, price about 3.5e-28.
    c = call_price(100.0, 300.0, 0.0, 0.2, 0.25)
    assert 0.0 < c < 1e-26


def test_deep_itm_call_quote_inverts_to_its_true_vol():
    # K=80 with sigma=5%: almost all of the call price is intrinsic. The old
    # call-side inversion returned about 0.0501. The OTM-side inversion
    # recovers 0.05 to 1e-8.
    S, K, r, T, sigma = 100.0, 80.0, 0.03, 1.0, 0.05
    c = call_price(S, K, r, sigma, T)
    assert implied_volatility(c, S, K, r, T) == pytest.approx(sigma, abs=1e-8)


def test_unresolvable_itm_quote_warns_and_returns_nan():
    # K=50 priced at sigma=5%: time value is about 1e-48, far below what a
    # double can resolve on a price of 51. The old code returned 0.175.
    S, K, r, T = 100.0, 50.0, 0.03, 1.0
    c = call_price(S, K, r, 0.05, T)
    with pytest.warns(RuntimeWarning, match="not determined"):
        assert math.isnan(implied_volatility(c, S, K, r, T))


@settings(max_examples=200, deadline=None)
@given(
    K=st.floats(60.0, 160.0),
    sigma=st.floats(0.08, 1.5),
    T=st.floats(0.1, 3.0),
    r=st.floats(-0.01, 0.06),
    put=st.booleans(),
)
def test_implied_vol_round_trip_calls_and_puts(K, sigma, T, r, put):
    S = 100.0
    otm = min(call_price(S, K, r, sigma, T), put_price(S, K, r, sigma, T))
    assume(otm >= 1e-3)
    price = put_price(S, K, r, sigma, T) if put else call_price(S, K, r, sigma, T)
    iv = implied_volatility(price, S, K, r, T, option_type="put" if put else "call")
    # The OTM price is at least 1e-3 (assumed above) and vega is not tiny, so a 1e-10
    # relative price residual bounds the vol error well below 1e-6.
    assert iv == pytest.approx(sigma, abs=1e-6)


def test_unknown_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        implied_volatility(5.0, 100.0, 100.0, 0.0, 1.0, option_type="straddle")
