"""Edge-case and property tests added in the review pass (project 09).

Gaps targeted (beyond the existing `test_edge_cases.py`):

* RR/BF -> strike conversion at extreme skew and near-degenerate quotes,
* butterfly-arbitrage detection as a *property* (monotone in planted skew),
* wing behaviour: SVI linear-total-variance asymptotics, wing extrapolation,
* T -> 0 limits for price, delta, vega and the DNS strike,
* NaN/Inf rejection across the pricer, greeks and quote conversion.
"""

import math

import numpy as np
import pytest

from fx_surface.garman_kohlhagen import (
    gk_delta,
    gk_forward,
    gk_gamma,
    gk_price,
    gk_vanna,
    gk_vega,
    gk_volga,
    implied_vol,
)
from fx_surface.smile import SVIParams, SVISmile
from fx_surface.smile_from_quotes import (
    SmileQuotes,
    atm_dns_strike,
    pa_call_delta_max,
    quotes_from_vols,
    solve_pillar_strikes,
    strike_from_delta,
    strike_from_delta_pa_candidates,
    vols_from_quotes,
)

S0, RD, RF = 1.10, 0.03, 0.02


# ---------------------------------------------------------------------------
# RR / BF -> strike conversion edge cases
# ---------------------------------------------------------------------------

def test_rr_sign_flips_which_wing_strike_carries_the_higher_vol():
    """Positive RR = calls richer; the 25d call strike must carry more vol."""
    pos = vols_from_quotes(SmileQuotes(0.10, +0.02, 0.005, +0.035, 0.015))
    neg = vols_from_quotes(SmileQuotes(0.10, -0.02, 0.005, -0.035, 0.015))
    assert pos["25c"] > pos["25p"]
    assert neg["25p"] > neg["25c"]
    # the ATM pillar is untouched by RR
    assert pos["atm"] == neg["atm"] == 0.10


def test_bf_lifts_both_wings_equally_leaving_rr_unchanged():
    base = SmileQuotes(0.10, 0.02, 0.005, 0.035, 0.015)
    lifted = SmileQuotes(0.10, 0.02, 0.015, 0.035, 0.025)
    vb, vl = vols_from_quotes(base), vols_from_quotes(lifted)
    assert vl["25c"] - vb["25c"] == pytest.approx(0.010)
    assert vl["25p"] - vb["25p"] == pytest.approx(0.010)
    assert quotes_from_vols(vl).rr25 == pytest.approx(base.rr25)


def test_extreme_skew_still_produces_ordered_pillar_strikes():
    """A heavily skewed EM-style quote set must keep K10P<K25P<KATM<K25C<K10C."""
    vols = vols_from_quotes(SmileQuotes(0.18, 0.06, 0.012, 0.11, 0.035))
    strikes = solve_pillar_strikes(vols, S0, 0.5, RD, RF, "spot")
    ks = [strikes[k] for k in ("10p", "25p", "atm", "25c", "10c")]
    assert all(a < b for a, b in zip(ks, ks[1:]))


def test_quote_set_driving_a_wing_negative_is_rejected():
    """RR big enough to push a wing vol <= 0 must raise, not produce a NaN vol."""
    with pytest.raises(ValueError, match="non-positive vol"):
        vols_from_quotes(SmileQuotes(0.10, 0.50, 0.005, 0.50, 0.01))


def test_degenerate_zero_rr_zero_bf_gives_all_pillars_at_atm():
    vols = vols_from_quotes(SmileQuotes(0.12, 0.0, 0.0, 0.0, 0.0))
    assert all(v == pytest.approx(0.12) for v in vols.values())


def test_quotes_vols_round_trip_survives_extreme_skew():
    q = SmileQuotes(0.20, 0.09, 0.02, 0.16, 0.05)
    back = quotes_from_vols(vols_from_quotes(q))
    for attr in ("atm", "rr25", "bf25", "rr10", "bf10"):
        assert getattr(back, attr) == pytest.approx(getattr(q, attr), abs=1e-15)


def test_strike_from_delta_is_monotone_decreasing_in_call_delta():
    """A lower call delta is a more OTM (higher) strike."""
    ks = [
        strike_from_delta(d, +1, 0.10, S0, 1.0, RD, RF, "spot")
        for d in (0.40, 0.25, 0.10, 0.05)
    ]
    assert all(a < b for a, b in zip(ks, ks[1:]))


def test_strike_from_delta_is_monotone_increasing_in_put_delta():
    ks = [
        strike_from_delta(d, -1, 0.10, S0, 1.0, RD, RF, "spot")
        for d in (0.05, 0.10, 0.25, 0.40)
    ]
    assert all(a < b for a, b in zip(ks, ks[1:]))


def test_higher_vol_pushes_wing_strikes_further_from_the_forward():
    """Same delta, more vol => the 25d call strike moves further OTM."""
    lo = strike_from_delta(0.25, +1, 0.08, S0, 1.0, RD, RF, "spot")
    hi = strike_from_delta(0.25, +1, 0.20, S0, 1.0, RD, RF, "spot")
    assert hi > lo > gk_forward(S0, 1.0, RD, RF)


def test_pa_market_branch_is_the_more_otm_of_the_two_candidates():
    K_low, K_mkt = strike_from_delta_pa_candidates(0.25, 0.12, S0, 1.0, RD, RF, "spot_pa")
    K_max, _ = pa_call_delta_max(0.12, S0, 1.0, RD, RF, "spot_pa")
    assert K_low < K_max < K_mkt
    assert strike_from_delta(0.25, +1, 0.12, S0, 1.0, RD, RF, "spot_pa") == pytest.approx(K_mkt)


def test_atm_dns_pa_sits_below_the_forward_and_unadjusted_above():
    F = gk_forward(S0, 1.0, RD, RF)
    assert atm_dns_strike(F, 0.12, 1.0, premium_adjusted=False) > F
    assert atm_dns_strike(F, 0.12, 1.0, premium_adjusted=True) < F


def test_atm_dns_strikes_collapse_to_the_forward_as_vol_vanishes():
    F = gk_forward(S0, 1.0, RD, RF)
    for pa in (False, True):
        assert atm_dns_strike(F, 1e-8, 1.0, premium_adjusted=pa) == pytest.approx(F, rel=1e-12)


# ---------------------------------------------------------------------------
# Butterfly arbitrage
# ---------------------------------------------------------------------------

def test_durrleman_is_positive_on_a_calm_well_behaved_smile():
    svi = SVISmile(SVIParams(a=0.010, b=0.05, rho=-0.2, m=0.0, s=0.20), F=1.10, T=1.0)
    ok, g_min = svi.is_butterfly_arbitrage_free()
    assert ok and g_min > 0.0


def test_butterfly_arbitrage_appears_as_curvature_is_pushed_up():
    """Property: raising b (wing slope) past a point breaks Durrleman's condition."""
    results = []
    for b in (0.02, 0.05, 0.20, 0.60, 1.20):
        svi = SVISmile(SVIParams(a=0.010, b=b, rho=-0.3, m=0.0, s=0.05), F=1.10, T=1.0)
        results.append(svi.is_butterfly_arbitrage_free()[1])
    # min g is decreasing in b, and the most extreme case is an arb violation
    assert all(a >= b for a, b in zip(results, results[1:]))
    assert results[-1] < 0.0


def test_arbitrage_free_flag_agrees_with_the_sign_of_min_g():
    for b in (0.02, 0.30, 1.5):
        svi = SVISmile(SVIParams(a=0.010, b=b, rho=-0.3, m=0.0, s=0.05), F=1.10, T=1.0)
        ok, g_min = svi.is_butterfly_arbitrage_free()
        assert ok == (g_min >= -1e-10)


def test_flat_smile_is_always_butterfly_arbitrage_free():
    svi = SVISmile(SVIParams(a=0.0144, b=0.0, rho=0.0, m=0.0, s=0.1), F=1.10, T=1.0)
    ok, g_min = svi.is_butterfly_arbitrage_free()
    assert ok and g_min > 0.0


def test_svi_rejects_parameters_with_nonpositive_minimum_variance():
    with pytest.raises(ValueError, match="minimum total variance"):
        SVIParams(a=-1.0, b=0.05, rho=-0.2, m=0.0, s=0.2)


# ---------------------------------------------------------------------------
# Wings
# ---------------------------------------------------------------------------

def test_svi_total_variance_is_asymptotically_linear_in_log_moneyness():
    """Raw SVI wings: w(k) -> b(rho +/- 1) k, so the slope tends to a constant."""
    p = SVIParams(a=0.010, b=0.08, rho=-0.3, m=0.0, s=0.15)
    svi = SVISmile(p, F=1.10, T=1.0)
    for k, expected in ((60.0, p.b * (p.rho + 1.0)), (-60.0, p.b * (p.rho - 1.0))):
        slope = float(svi.w_prime(k))
        assert slope == pytest.approx(expected, rel=1e-3)


def test_svi_wing_variance_stays_positive_far_out():
    svi = SVISmile(SVIParams(a=0.010, b=0.08, rho=-0.3, m=0.0, s=0.15), F=1.10, T=1.0)
    k = np.linspace(-8.0, 8.0, 401)
    assert np.all(svi.total_variance(k) > 0.0)
    assert np.all(np.isfinite(svi.vol_logm(k)))


def test_svi_curvature_decays_in_the_wings():
    """w'' -> 0 far from the money: the smile straightens out."""
    svi = SVISmile(SVIParams(a=0.010, b=0.08, rho=-0.3, m=0.0, s=0.15), F=1.10, T=1.0)
    assert float(svi.w_second(0.0)) > float(svi.w_second(2.0)) > float(svi.w_second(10.0)) > 0.0


def test_deep_wing_option_prices_stay_within_no_arbitrage_bounds():
    T = 1.0
    F = gk_forward(S0, T, RD, RF)
    df_d = math.exp(-RD * T)
    for K in (0.20, 0.50, 2.5, 6.0):
        c = gk_price(S0, K, T, RD, RF, 0.15, +1)
        p = gk_price(S0, K, T, RD, RF, 0.15, -1)
        assert max(df_d * (F - K), 0.0) - 1e-12 <= c <= S0 * math.exp(-RF * T) + 1e-12
        assert max(df_d * (K - F), 0.0) - 1e-12 <= p <= df_d * K + 1e-12
        # put-call parity holds in the deep wings too
        assert c - p == pytest.approx(df_d * (F - K), abs=1e-12)


def test_deep_otm_wing_price_is_positive_but_tiny():
    c = gk_price(S0, 10.0, 1.0, RD, RF, 0.10, +1)
    assert 0.0 < c < 1e-9


# ---------------------------------------------------------------------------
# T -> 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1e-2, 1e-4, 1e-6, 1e-8])
def test_price_converges_to_intrinsic_as_expiry_vanishes(T):
    itm = gk_price(S0, 1.00, T, RD, RF, 0.10, +1)
    otm = gk_price(S0, 1.20, T, RD, RF, 0.10, +1)
    assert itm == pytest.approx(S0 - 1.00, abs=5e-3 * math.sqrt(T / 1e-2) + 1e-6)
    assert otm < 1e-2
    # An OTM premium may legitimately UNDERFLOW to exactly 0.0 at
    # microscopic expiries (N(d2) with d2 ~ -1e4); it must never go
    # negative, and must never be NaN.
    assert otm >= 0.0 and math.isfinite(otm)


def test_atm_price_vanishes_like_sqrt_t():
    """ATM value ~ S*sigma*sqrt(T/2pi): halving T scales price by 1/sqrt(2)."""
    p1 = gk_price(S0, gk_forward(S0, 1e-4, RD, RF), 1e-4, RD, RF, 0.10, +1)
    p2 = gk_price(S0, gk_forward(S0, 5e-5, RD, RF), 5e-5, RD, RF, 0.10, +1)
    assert p1 / p2 == pytest.approx(math.sqrt(2.0), rel=1e-3)


def test_delta_approaches_step_function_as_expiry_vanishes():
    T = 1e-8
    assert gk_delta(S0, 0.90, T, RD, RF, 0.10, +1, "spot") == pytest.approx(1.0, abs=1e-6)
    assert gk_delta(S0, 1.30, T, RD, RF, 0.10, +1, "spot") == pytest.approx(0.0, abs=1e-6)
    assert gk_delta(S0, 0.90, T, RD, RF, 0.10, -1, "spot") == pytest.approx(0.0, abs=1e-6)


def test_vega_vanishes_as_expiry_vanishes():
    vegas = [gk_vega(S0, S0, T, RD, RF, 0.10) for T in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert all(a > b > 0 for a, b in zip(vegas, vegas[1:]))
    assert vegas[-1] < 1e-4


def test_implied_vol_round_trips_at_very_short_expiry():
    T = 1e-6
    K = gk_forward(S0, T, RD, RF)
    for sigma in (0.05, 0.15, 0.40):
        px = gk_price(S0, K, T, RD, RF, sigma, +1)
        assert implied_vol(px, S0, K, T, RD, RF, +1) == pytest.approx(sigma, rel=1e-6)


def test_zero_and_negative_expiry_rejected_everywhere():
    for T in (0.0, -1.0):
        with pytest.raises(ValueError, match="time to expiry"):
            gk_price(S0, 1.10, T, RD, RF, 0.10, +1)
        with pytest.raises(ValueError, match="time to expiry"):
            gk_vega(S0, 1.10, T, RD, RF, 0.10)


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["S", "K", "T", "r_d", "r_f", "sigma"])
def test_pricer_rejects_nonfinite_inputs(field, bad):
    """A NaN slips past every ``<= 0`` guard: it must be caught explicitly."""
    kwargs = dict(S=S0, K=1.10, T=1.0, r_d=RD, r_f=RF, sigma=0.10)
    kwargs[field] = bad
    with pytest.raises(ValueError):
        gk_price(cp=+1, **kwargs)


@pytest.mark.parametrize("greek", [gk_vega, gk_gamma, gk_vanna, gk_volga])
def test_greeks_reject_nonfinite_rates(greek):
    with pytest.raises(ValueError, match="must be finite"):
        greek(S0, 1.10, 1.0, np.nan, RF, 0.10)
    with pytest.raises(ValueError, match="must be finite"):
        greek(S0, 1.10, 1.0, RD, np.inf, 0.10)


def test_delta_rejects_nonfinite_inputs():
    with pytest.raises(ValueError):
        gk_delta(S0, 1.10, 1.0, RD, RF, np.nan, +1, "spot")
    with pytest.raises(ValueError):
        gk_delta(np.nan, 1.10, 1.0, RD, RF, 0.10, +1, "spot")


def test_forward_rejects_nonfinite_rates():
    with pytest.raises(ValueError, match="must be finite"):
        gk_forward(S0, 1.0, np.nan, RF)
    with pytest.raises(ValueError, match="must be finite"):
        gk_forward(S0, 1.0, RD, np.nan)


def test_strike_solving_rejects_nonfinite_vol_and_out_of_range_delta():
    with pytest.raises(ValueError):
        strike_from_delta(0.25, +1, np.nan, S0, 1.0, RD, RF, "spot")
    for d in (0.0, 1.0, -0.25, 1.5):
        with pytest.raises(ValueError, match="delta magnitude"):
            strike_from_delta(d, +1, 0.10, S0, 1.0, RD, RF, "spot")


def test_nonpositive_atm_quote_rejected():
    for atm in (0.0, -0.05):
        with pytest.raises(ValueError, match="ATM vol must be positive"):
            SmileQuotes(atm, 0.01, 0.005, 0.02, 0.01)


def test_unknown_delta_convention_rejected():
    with pytest.raises(ValueError, match="unknown delta convention"):
        gk_delta(S0, 1.10, 1.0, RD, RF, 0.10, +1, "spot_premium_adj")
    with pytest.raises(ValueError, match="unknown delta convention"):
        strike_from_delta(0.25, +1, 0.10, S0, 1.0, RD, RF, "fwd_pa")
