"""W3 (FX): impact does not depend on how finely the same order is sliced.

Up to v2.0.0 the simulator scaled both impact terms by the *bucket* vol
``sigma * sqrt(dt)`` while participation ``|q| / (depth * dt)`` was already a
rate.  The same 2-hour TWAP therefore paid less temporary impact on a finer
grid (``~ sqrt(dt)``) and more permanent impact (``~ 1 / sqrt(dt)``).  Both
terms now use the vol over a fixed reference horizon
(``impact_horizon_minutes``, default 5 = the default bucket), so a fixed
trading-rate profile costs the same at any bucket width.
"""

from __future__ import annotations

import numpy as np
import pytest
from fx_algo.execution import MarketSimulator, twap_schedule
from fx_algo.sessions import EURUSD

PARENT = 200.0  # mm EUR over two London-NY overlap hours


def costs(dt, **kw):
    sim = MarketSimulator(EURUSD, start_hour=13.0, horizon_hours=2.0, dt_minutes=dt, **kw)
    res = sim.execute(twap_schedule(PARENT, sim.n_buckets))
    temp = float(np.sum(res.temp_pips * res.qty) / PARENT)
    return temp, res.perm_total_pips


@pytest.mark.parametrize("dt", [1.0, 2.5, 15.0, 30.0])
def test_twap_impact_independent_of_bucket_width(dt):
    t_ref, p_ref = costs(5.0)
    t, p = costs(dt)
    assert t == pytest.approx(t_ref, rel=1e-12)
    assert p == pytest.approx(p_ref, rel=1e-12)


def test_default_bucket_numbers_unchanged():
    """At the default 5-minute bucket the reference horizon equals the bucket,
    so every v2.0.0 number produced on a 5-minute grid is unchanged."""
    sim = MarketSimulator(EURUSD, start_hour=13.0, horizon_hours=2.0, dt_minutes=5.0)
    q = twap_schedule(PARENT, sim.n_buckets)
    res = sim.execute(q)
    part = np.abs(q) / sim.depth_bucket
    np.testing.assert_allclose(res.temp_pips, 0.35 * sim.sigma_bucket_pips * np.sqrt(part),
                               rtol=1e-12)


def test_linear_exponent_option():
    sim = MarketSimulator(EURUSD, start_hour=13.0, horizon_hours=1.0, dt_minutes=5.0, impact_exponent=1.0)
    q = twap_schedule(60.0, sim.n_buckets)
    res = sim.execute(q)
    part = np.abs(q) / sim.depth_bucket
    np.testing.assert_allclose(res.temp_pips, 0.35 * EURUSD.vol_at(13.5)[0] * np.sqrt(5.0) * part,
                               rtol=1e-12)
    with pytest.raises(ValueError):
        MarketSimulator(EURUSD, impact_exponent=0.0)
    with pytest.raises(ValueError):
        MarketSimulator(EURUSD, impact_horizon_minutes=-1.0)
