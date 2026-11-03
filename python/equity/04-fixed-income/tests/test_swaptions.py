"""fi_rates.swaptions: SABR cube on an ILLUSTRATIVE surface (not market data).

No free swaption quotes exist (plan W15). The node parameters below were
chosen by hand to look like a rates smile; the tests check the machinery
(fit, interpolation, parity), not market levels.
"""

from __future__ import annotations

import numpy as np
import pytest
from fi_rates.curve import DiscountCurve
from fi_rates.rate_options import forward_swap_rate, swap_annuity
from fi_rates.swaptions import SabrSwaptionCube, SmileQuote, cube_swaption_price
from qp_core.vol.sabr import sabr_vol

T = np.arange(1, 161) * 0.25
DISC = DiscountCurve(T, np.exp(-0.03 * T))
PROJ = DiscountCurve(T, np.exp(-0.0325 * T))
BETA, SHIFT = 0.5, 0.02
EXP, TEN = (1.0, 5.0, 10.0), (2.0, 10.0)
# illustrative, not market data
TRUE = {(e, t): (0.03 + 0.002 * i, -0.45 + 0.1 * i + 0.05 * j, 0.55 - 0.1 * i - 0.05 * j)
        for i, e in enumerate(EXP) for j, t in enumerate(TEN)}


def _quotes():
    out = {}
    for (e, t), (a, r, n) in TRUE.items():
        F = forward_swap_rate(DISC, PROJ, e, t)
        ks = tuple(F + d for d in (-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02))
        out[(e, t)] = SmileQuote(F, ks, tuple(sabr_vol(F, k, e, a, BETA, r, n, SHIFT) for k in ks))
    return out


CUBE = SabrSwaptionCube.calibrate(_quotes(), BETA, SHIFT)


def test_nodes_recover_parameters():
    for i, e in enumerate(EXP):
        for j, t in enumerate(TEN):
            a, r, n = TRUE[(e, t)]
            assert (CUBE.alpha[i, j], CUBE.rho[i, j], CUBE.nu[i, j]) == pytest.approx((a, r, n), abs=1e-6)
    assert CUBE.rmse is not None and np.max(CUBE.rmse) < 1e-9


def test_interpolation_is_bilinear_and_flat_outside():
    _, r, _ = CUBE.params(3.0, 6.0)
    want = 0.25 * (CUBE.rho[0, 0] + CUBE.rho[1, 0] + CUBE.rho[0, 1] + CUBE.rho[1, 1])
    assert r == pytest.approx(want, abs=1e-15)
    assert CUBE.params(0.1, 1.0) == CUBE.params(1.0, 2.0)
    assert CUBE.params(30.0, 30.0) == CUBE.params(10.0, 10.0)


def test_node_vol_is_the_quote():
    q = _quotes()[(5.0, 10.0)]
    for k, v in zip(q.strikes, q.vols, strict=True):
        assert CUBE.vol(5.0, 10.0, k, q.forward) == pytest.approx(v, abs=1e-9)


def test_put_call_parity():
    e, t, K = 3.0, 6.0, 0.028
    _, A = swap_annuity(DISC, e, t, 1)
    F = forward_swap_rate(DISC, PROJ, e, t)
    pay = cube_swaption_price(CUBE, DISC, PROJ, e, t, K, True)
    rec = cube_swaption_price(CUBE, DISC, PROJ, e, t, K, False)
    assert pay - rec == pytest.approx(A * (F - K), abs=1e-15)


def test_validation():
    with pytest.raises(ValueError):
        SabrSwaptionCube((1.0,), (2.0,), np.zeros((2, 1)), np.zeros((1, 1)), np.zeros((1, 1)), 0.5)
    with pytest.raises(ValueError):
        SabrSwaptionCube.calibrate(dict(list(_quotes().items())[:-1]), BETA, SHIFT)
