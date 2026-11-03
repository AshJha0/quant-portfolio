"""Compare qp_core.capital.frtb_sbm with the independent MAR21 oracle (W17).

250 seeded random portfolios per risk class and measure, every correlation
scenario plus the final charge, to 1e-10 relative. Portfolios mix signs so
the MAR21.4(5) negative-sum fallback is exercised, and include duplicated
risk factors so netting is exercised.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from qp_core.capital import frtb_sbm as impl

from . import mar21_oracle as ref

N = 250
KEYS = ("medium", "high", "low", "charge")
CCYS = ("USD", "EUR", "GBP", "JPY", "CHF", "NZD", "MXN")
PAIRS = ("EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDTRY", "USDPLN", "USDTHB")
VEGA_T = (0.5, 1.0, 3.0, 5.0, 10.0)


def _close(a: dict[str, float], b: dict[str, float]) -> None:
    for k in KEYS:
        assert a[k] == pytest.approx(b[k], rel=1e-10, abs=1e-12), (k, a, b)


def _amt(r: random.Random) -> float:
    return r.choice((-1, 1)) * 10 ** r.uniform(2, 7)


@pytest.mark.parametrize("relief", [True, False])
@pytest.mark.parametrize("seed", range(N))
def test_girr_delta(seed: int, relief: bool) -> None:
    r = random.Random(seed)
    rows = [
        (
            r.choice(CCYS[: r.randint(1, 7)]),
            r.choice(("OIS", "LIBOR3M", "SOFR")),
            r.choice(ref.GIRR_TENORS),
            _amt(r),
        )
        for _ in range(r.randint(1, 40))
    ]
    got = impl.girr_delta([impl.GirrDelta(*x) for x in rows], specified_relief=relief)
    _close(got, ref.girr_delta(rows, relief))


@pytest.mark.parametrize("seed", range(N))
def test_girr_vega(seed: int) -> None:
    r = random.Random(1000 + seed)
    rows = [
        (r.choice(CCYS[: r.randint(1, 7)]), r.choice(VEGA_T), r.choice(VEGA_T), _amt(r))
        for _ in range(r.randint(1, 40))
    ]
    _close(impl.girr_vega([impl.GirrVega(*x) for x in rows]), ref.girr_vega(rows))


@pytest.mark.parametrize("relief", [True, False])
@pytest.mark.parametrize("seed", range(N))
def test_fx_delta(seed: int, relief: bool) -> None:
    r = random.Random(2000 + seed)
    rows = [(r.choice(PAIRS), _amt(r)) for _ in range(r.randint(1, 12))]
    got = impl.fx_delta([impl.FxDelta(*x) for x in rows], specified_relief=relief)
    _close(got, ref.fx_delta(rows, relief))


@pytest.mark.parametrize("seed", range(N))
def test_fx_vega(seed: int) -> None:
    r = random.Random(3000 + seed)
    rows = [(r.choice(PAIRS), r.choice(VEGA_T), _amt(r)) for _ in range(r.randint(1, 30))]
    _close(impl.fx_vega([impl.FxVega(*x) for x in rows]), ref.fx_vega(rows))


@pytest.mark.parametrize("seed", range(N))
def test_across_bucket_fallback(seed: int) -> None:
    """MAR21.4(5)(b): the clipped-S_b branch.

    With the GIRR and FX parameters of MAR21.46-50 and MAR21.89 the branch
    cannot fire from tenor sensitivities alone (K_b^2 >= |S_b|^2 / 2.5 while
    gamma <= 0.625), so the aggregation step is compared directly with
    K_b and S_b drawn so that roughly half the cases take the branch.
    """
    r = random.Random(5000 + seed)
    n = r.randint(2, 6)
    kb = [r.uniform(0.1, 1.0) for _ in range(n)]
    sb = [r.choice((-1, 1)) * r.uniform(0.0, 4.0) for _ in range(n)]
    gamma = r.uniform(0.0, 1.0)
    want = ref.across(kb, sb, gamma)
    g = np.full((n, n), gamma)
    np.fill_diagonal(g, 1.0)
    assert impl.across_buckets(kb, sb, g) == pytest.approx(want, rel=1e-10, abs=1e-12)


def test_negative_sum_fallback_is_exercised() -> None:
    """Hand check of the MAR21.4(5)(b) fallback in the oracle itself."""
    kb, sb = [1.0, 1.0], [3.0, -3.0]
    # 1 + 1 + 2*0.5*(-9) = -7 < 0 -> S clipped to +-1 -> 2 - 1 = 1
    assert ref.across(kb, sb, 0.5) == pytest.approx(1.0)
