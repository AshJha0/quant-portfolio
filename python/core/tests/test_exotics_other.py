"""Vanna-volga barriers, Asians, LSM Americans and TARFs."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.exotics.asian import (
    asian_mc,
    geometric_asian_price,
    turnbull_wakeman_continuous,
    turnbull_wakeman_price,
)
from qp_core.exotics.barriers import barrier_price, vanilla_price
from qp_core.exotics.lsm import crr_american, lsm_american
from qp_core.exotics.tarf import tarf_mc
from qp_core.exotics.vanna_volga import vanna_volga_barrier, vanna_volga_vanilla

# --------------------------------------------------------------- vanna-volga
S, T, RD, RF = 1.30, 0.5, 0.03, 0.02
PK = (1.24, 1.305, 1.37)
PV = (0.112, 0.100, 0.104)  # skewed: 25P richer than 25C


def test_vv_reproduces_pillars_exactly():
    for K, v in zip(PK, PV, strict=True):
        mkt = vanilla_price("call", S, K, T, RD, RD - RF, v)
        assert vanna_volga_vanilla("call", S, K, T, RD, RF, PK, PV) == pytest.approx(mkt, abs=1e-12)


def test_vv_put_call_parity_and_smile_between_pillars():
    c = vanna_volga_vanilla("call", S, 1.33, T, RD, RF, PK, PV)
    p = vanna_volga_vanilla("put", S, 1.33, T, RD, RF, PK, PV)
    assert c - p == pytest.approx(S * math.exp(-RF * T) - 1.33 * math.exp(-RD * T), abs=1e-12)
    # With a steep put wing the vanna-volga smile has its minimum between ATM
    # and the 25C pillar, so the vol at 1.33 may sit just under the ATM vol.
    lo = vanilla_price("call", S, 1.33, T, RD, RD - RF, 0.095)
    hi = vanilla_price("call", S, 1.33, T, RD, RD - RF, PV[2])
    assert lo < c < hi


@pytest.mark.parametrize("weighted", [True, False])
def test_vv_flat_smile_is_black_scholes(weighted):
    flat = (0.1, 0.1, 0.1)
    bs = barrier_price("up_out", "call", S, 1.30, 1.40, T, RD, RD - RF, 0.1)
    vv = vanna_volga_barrier("up_out", "call", S, 1.30, 1.40, T, RD, RF, PK, flat, survival_weighted=weighted)
    assert vv == pytest.approx(bs, abs=1e-12)


def test_vv_knock_in_out_parity_and_far_barrier():
    for kind_out, kind_in, H in (("down_out", "down_in", 1.18), ("up_out", "up_in", 1.45)):
        o = vanna_volga_barrier(kind_out, "call", S, 1.30, H, T, RD, RF, PK, PV)
        i = vanna_volga_barrier(kind_in, "call", S, 1.30, H, T, RD, RF, PK, PV)
        assert o + i == pytest.approx(vanna_volga_vanilla("call", S, 1.30, T, RD, RF, PK, PV), abs=1e-12)
    far = vanna_volga_barrier("down_out", "put", S, 1.30, 0.3, T, RD, RF, PK, PV)
    assert far == pytest.approx(vanna_volga_vanilla("put", S, 1.30, T, RD, RF, PK, PV), abs=1e-7)


def test_vv_survival_weight_shrinks_adjustment_near_barrier():
    bs = barrier_price("up_out", "call", S, 1.30, 1.36, T, RD, RD - RF, PV[1])
    w = vanna_volga_barrier("up_out", "call", S, 1.30, 1.36, T, RD, RF, PK, PV)
    u = vanna_volga_barrier("up_out", "call", S, 1.30, 1.36, T, RD, RF, PK, PV, survival_weighted=False)
    assert abs(w - bs) < abs(u - bs)
    assert w >= 0.0
    with pytest.raises(ValueError):
        vanna_volga_vanilla("call", S, 1.3, T, RD, RF, (1.3, 1.2, 1.4), PV)


# --------------------------------------------------------------- Asians
FIX = [i / 12 for i in range(1, 13)]


@pytest.mark.parametrize("cp", ["call", "put"])
@pytest.mark.parametrize("X", [90.0, 100.0, 110.0])
def test_geometric_closed_form_vs_plain_mc(cp, X):
    """The geometric price is exact; plain MC must be within 3 SE."""
    rng = np.random.default_rng(4)
    t = np.array(FIX)
    dt = np.diff(np.concatenate([[0.0], t]))
    z = rng.standard_normal((200_000, t.size))
    logs = math.log(100) + np.cumsum((0.03 - 0.5 * 0.25**2) * dt + 0.25 * np.sqrt(dt) * z, axis=1)
    g = np.exp(logs.mean(axis=1))
    sign = 1 if cp == "call" else -1
    y = math.exp(-0.05) * np.maximum(sign * (g - X), 0.0)
    cf = geometric_asian_price(cp, 100, X, FIX, 1.0, 0.05, 0.03, 0.25)
    assert abs(y.mean() - cf) < 3 * y.std(ddof=1) / math.sqrt(y.size)


@pytest.mark.parametrize("cp", ["call", "put"])
@pytest.mark.parametrize("X", [90.0, 100.0, 110.0])
def test_turnbull_wakeman_vs_control_variate_mc(cp, X):
    tw = turnbull_wakeman_price(cp, 100, X, FIX, 1.0, 0.05, 0.03, 0.25)
    est, se = asian_mc(cp, 100, X, FIX, 1.0, 0.05, 0.03, 0.25, n_paths=200_000, seed=9)
    plain_est, plain_se = asian_mc(cp, 100, X, FIX, 1.0, 0.05, 0.03, 0.25, n_paths=200_000, seed=9,
                                   control_variate=False)
    assert se < plain_se / 5  # the geometric control variate earns its keep
    assert abs(plain_est - est) < 3 * plain_se
    # TW is an approximation, not a pricer to 3 SE. Measured against this
    # MC (S=100, 12 monthly fixings, sigma 25%, T=1): +0.050, +0.024, -0.017
    # for X = 90, 100, 110, the same for puts (put-call parity holds in both).
    # The error grows with sigma^2 T (Levy 1992; Haug 2007 s.4.20).
    assert abs(tw - est) < 0.06, (tw, est, se)


def test_tw_discrete_tends_to_continuous():
    fine = [i / 2000 for i in range(1, 2001)]
    d = turnbull_wakeman_price("call", 100, 100, fine, 1.0, 0.05, 0.03, 0.25)
    c = turnbull_wakeman_continuous("call", 100, 100, 1.0, 0.05, 0.03, 0.25)
    assert d == pytest.approx(c, abs=5e-3)
    c0 = turnbull_wakeman_continuous("put", 100, 100, 1.0, 0.05, 1e-10, 0.25)
    c1 = turnbull_wakeman_continuous("put", 100, 100, 1.0, 0.05, 1e-6, 0.25)
    assert c0 == pytest.approx(c1, abs=1e-4)


def test_asian_input_checks():
    with pytest.raises(ValueError):
        geometric_asian_price("call", 100, 100, [0.5, 0.25], 1.0, 0.05, 0.03, 0.2)
    with pytest.raises(ValueError):
        turnbull_wakeman_price("call", 100, 100, [0.5, 1.5], 1.0, 0.05, 0.03, 0.2)
    with pytest.raises(ValueError):
        turnbull_wakeman_continuous("straddle", 100, 100, 1.0, 0.05, 0.03, 0.2)


# --------------------------------------------------------------- LSM
def test_lsm_longstaff_schwartz_table_1():
    est, se = lsm_american("put", 36.0, 40.0, 1.0, 0.06, 0.0, 0.2, n_paths=100_000, n_steps=50, seed=1)
    # LS (2001) Table 1: finite difference 4.478, LSM 4.472 (s.e. 0.010).
    assert abs(est - 4.478) < 3 * se + 0.02, (est, se)


@pytest.mark.parametrize(("S0", "sig", "T"), [(36, 0.2, 1.0), (40, 0.4, 2.0), (44, 0.2, 2.0)])
def test_lsm_vs_crr_tree(S0, sig, T):
    tree = crr_american("put", S0, 40.0, T, 0.06, 0.0, sig, n_steps=2000)
    est, se = lsm_american("put", S0, 40.0, T, 0.06, 0.0, sig, n_paths=100_000, n_steps=50, seed=2)
    # 50 exercise dates vs a continuous-exercise tree: LSM is low by the
    # Bermudan gap (about 0.01 here) plus its own small bias.
    assert -0.04 - 3 * se < est - tree < 3 * se, (est, tree, se)


def test_american_call_no_dividend_equals_european():
    tree = crr_american("call", 100, 100, 1.0, 0.05, 0.0, 0.2, n_steps=2000)
    euro = vanilla_price("call", 100, 100, 1.0, 0.05, 0.05, 0.2)
    assert tree == pytest.approx(euro, abs=5e-3)
    est, se = lsm_american("call", 100, 100, 1.0, 0.05, 0.0, 0.2, n_paths=50_000, seed=3)
    assert abs(est - euro) < 3 * se + 0.01


# --------------------------------------------------------------- TARF
TF = [i / 12 for i in range(1, 13)]


def test_tarf_unreachable_target_is_forward_strip():
    v, se = tarf_mc(1.10, 1.06, TF, 0.03, 0.01, 0.08, 1e9, leverage=1.0, n_paths=100_000, seed=1)
    exact = sum(math.exp(-0.03 * t) * (1.10 * math.exp(0.02 * t) - 1.06) for t in TF)
    assert abs(v - exact) < 3 * se


def test_tarf_orderings():
    args = (1.10, 1.06, TF, 0.03, 0.01, 0.08, 0.15)
    full, _ = tarf_mc(*args, leverage=2.0, final="full", n_paths=50_000, seed=2)
    capped, _ = tarf_mc(*args, leverage=2.0, final="capped", n_paths=50_000, seed=2)
    none, _ = tarf_mc(*args, leverage=2.0, final="none", n_paths=50_000, seed=2)
    assert none <= capped <= full
    lev1, _ = tarf_mc(*args, leverage=1.0, final="full", n_paths=50_000, seed=2)
    assert lev1 > full  # leverage only adds losses
    with pytest.raises(ValueError):
        tarf_mc(*args, final="partial")
