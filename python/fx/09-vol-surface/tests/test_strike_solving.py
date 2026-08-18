"""Delta -> strike solving under all four FX delta conventions."""

import math

import numpy as np
import pytest

from fx_surface import (
    DELTA_CONVENTIONS,
    atm_dns_strike,
    gk_delta,
    gk_forward,
    pa_call_delta_max,
    solve_pillar_strikes,
    strike_from_delta,
    strike_from_delta_pa_candidates,
    vols_from_quotes,
)

S, T, RD, RF, SIG = 1.10, 0.5, 0.045, 0.033, 0.09
JPY = dict(S=150.0, T=0.75, r_d=0.002, r_f=0.044, sigma=0.101)


@pytest.mark.parametrize("convention", DELTA_CONVENTIONS)
@pytest.mark.parametrize("cp", [+1, -1])
@pytest.mark.parametrize("delta", [0.10, 0.25])
def test_delta_strike_delta_round_trip(convention, cp, delta):
    K = strike_from_delta(delta, cp, SIG, S, T, RD, RF, convention)
    d = gk_delta(S, K, T, RD, RF, SIG, cp, convention)
    assert abs(abs(d) - delta) < 1e-8
    assert math.copysign(1, d) == cp


@pytest.mark.parametrize("convention", ["spot_pa", "forward_pa"])
@pytest.mark.parametrize("cp", [+1, -1])
def test_pa_round_trip_jpy_levels(convention, cp):
    K = strike_from_delta(0.25, cp, JPY["sigma"], JPY["S"], JPY["T"],
                          JPY["r_d"], JPY["r_f"], convention)
    d = gk_delta(JPY["S"], K, JPY["T"], JPY["r_d"], JPY["r_f"],
                 JPY["sigma"], cp, convention)
    assert abs(abs(d) - 0.25) < 1e-8


def test_atm_dns_unadjusted_formula_and_neutrality():
    F = gk_forward(S, T, RD, RF)
    K = atm_dns_strike(F, SIG, T, premium_adjusted=False)
    assert K == pytest.approx(F * math.exp(0.5 * SIG**2 * T), rel=1e-15)
    for conv in ("spot", "forward"):
        dc = gk_delta(S, K, T, RD, RF, SIG, +1, conv)
        dp = gk_delta(S, K, T, RD, RF, SIG, -1, conv)
        assert abs(dc + dp) < 1e-12  # straddle is delta-neutral


def test_atm_dns_premium_adjusted_formula_and_neutrality():
    F = gk_forward(S, T, RD, RF)
    K = atm_dns_strike(F, SIG, T, premium_adjusted=True)
    assert K == pytest.approx(F * math.exp(-0.5 * SIG**2 * T), rel=1e-15)
    assert K < F  # pa DNS sits below the forward
    for conv in ("spot_pa", "forward_pa"):
        dc = gk_delta(S, K, T, RD, RF, SIG, +1, conv)
        dp = gk_delta(S, K, T, RD, RF, SIG, -1, conv)
        assert abs(dc + dp) < 1e-12


def test_pa_call_delta_has_interior_maximum():
    K_max, d_max = pa_call_delta_max(SIG, S, T, RD, RF, "spot_pa")
    eps = 1e-4 * K_max
    d_at = gk_delta(S, K_max, T, RD, RF, SIG, 1, "spot_pa")
    assert d_at == pytest.approx(d_max, abs=1e-12)
    assert gk_delta(S, K_max - eps, T, RD, RF, SIG, 1, "spot_pa") < d_max
    assert gk_delta(S, K_max + eps, T, RD, RF, SIG, 1, "spot_pa") < d_max


def test_pa_call_two_candidates_share_delta_market_branch_selected():
    K_low, K_mkt = strike_from_delta_pa_candidates(0.25, SIG, S, T, RD, RF, "spot_pa")
    K_max, _ = pa_call_delta_max(SIG, S, T, RD, RF, "spot_pa")
    for K in (K_low, K_mkt):
        d = gk_delta(S, K, T, RD, RF, SIG, 1, "spot_pa")
        assert abs(d - 0.25) < 1e-10  # both really are 25-delta strikes
    assert K_low < K_max < K_mkt  # candidates straddle the maximum
    # strike_from_delta must return the market-standard (OTM, falling
    # branch) candidate and reject the low-strike one.
    K = strike_from_delta(0.25, +1, SIG, S, T, RD, RF, "spot_pa")
    assert K == pytest.approx(K_mkt, rel=1e-12)
    assert not np.isclose(K, K_low, rtol=1e-3)


def test_pa_call_delta_above_maximum_raises():
    _, d_max = pa_call_delta_max(SIG, S, T, RD, RF, "spot_pa")
    with pytest.raises(ValueError, match="unattainable"):
        strike_from_delta(d_max + 0.01, +1, SIG, S, T, RD, RF, "spot_pa")


@pytest.mark.parametrize("convention", ["spot", "spot_pa"])
def test_pillar_strike_ordering(convention):
    vols = vols_from_quotes(
        __import__("fx_surface").SmileQuotes(0.101, -0.017, 0.003, -0.0306, 0.0087)
    )
    strikes = solve_pillar_strikes(vols, JPY["S"], JPY["T"], JPY["r_d"],
                                   JPY["r_f"], convention)
    ks = [strikes[p] for p in ("10p", "25p", "atm", "25c", "10c")]
    assert all(a < b for a, b in zip(ks, ks[1:]))


def test_pillar_ordering_all_preset_slices(eurusd_surface, usdjpy_surface):
    for surf in (eurusd_surface, usdjpy_surface):
        for sl in surf.slices:
            ks = [sl.strikes[p] for p in ("10p", "25p", "atm", "25c", "10c")]
            assert all(a < b for a, b in zip(ks, ks[1:]))


def test_pa_strikes_differ_from_unadjusted():
    """Same quotes, different convention => materially different strikes."""
    vols = {"10p": 0.13, "25p": 0.113, "atm": 0.10, "25c": 0.094, "10c": 0.094}
    k_un = solve_pillar_strikes(vols, **{k: JPY[k] for k in ("S", "T", "r_d", "r_f")},
                                convention="spot")
    k_pa = solve_pillar_strikes(vols, **{k: JPY[k] for k in ("S", "T", "r_d", "r_f")},
                                convention="spot_pa")
    assert abs(k_pa["25c"] - k_un["25c"]) / k_un["25c"] > 1e-3
    assert k_pa["atm"] < k_un["atm"]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="delta magnitude"):
        strike_from_delta(1.2, 1, SIG, S, T, RD, RF, "spot")
    with pytest.raises(ValueError, match="convention"):
        strike_from_delta(0.25, 1, SIG, S, T, RD, RF, "premium")
    with pytest.raises(ValueError, match="cp"):
        strike_from_delta(0.25, 2, SIG, S, T, RD, RF, "spot")
    with pytest.raises(ValueError):
        atm_dns_strike(-1.0, SIG, T)


def test_spot_delta_target_beyond_forward_cap_raises():
    # spot delta targets close to 1 can exceed N(.) reach once divided
    # by the foreign discount factor
    with pytest.raises(ValueError, match="unattainable"):
        strike_from_delta(0.999999, 1, 0.02, 1.0, 10.0, 0.0, 0.08, "spot")
