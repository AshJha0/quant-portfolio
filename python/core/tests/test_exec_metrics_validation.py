"""Tests for qp_core.execution, qp_core.metrics and qp_core.validation."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as hst
from hypothesis.extra.numpy import arrays

from qp_core.execution import ExecutionConfig, fill_price
from qp_core.metrics import (
    annualisation_factor,
    check_periods_per_year,
    downside_deviation,
    drawdown_series,
    max_drawdown_from_equity,
    max_drawdown_from_pnl,
    max_drawdown_from_returns,
    wealth_curve,
)
from qp_core.validation import finite_scalar, finite_series, require_finite

small_returns = arrays(np.float64, hst.integers(1, 60), elements=hst.floats(-0.5, 0.5))


class TestExecution:
    def test_defaults(self):
        c = ExecutionConfig()
        assert (c.mode, c.fill_fraction, c.auction_cost_bp) == ("next", 0.5, 10.0)
        assert c.effective_fraction == 0.5
        assert ExecutionConfig(mode="close_auction").effective_fraction == 0.0

    @pytest.mark.parametrize(
        "kwargs, msg",
        [
            ({"mode": "same_close"}, "mode"),
            ({"fill_fraction": 0.0}, "fill_fraction"),
            ({"fill_fraction": 1.5}, "fill_fraction"),
            ({"fill_fraction": float("nan")}, "fill_fraction"),
            ({"auction_cost_bp": -1.0}, "auction_cost_bp"),
            ({"auction_cost_bp": float("inf")}, "auction_cost_bp"),
        ],
    )
    def test_rejects(self, kwargs, msg):
        with pytest.raises(ValueError, match=msg):
            ExecutionConfig(**kwargs)

    def test_fill_price(self):
        assert fill_price(100.0, 104.0, ExecutionConfig(fill_fraction=0.25)) == 101.0
        assert fill_price(100.0, 104.0, ExecutionConfig(fill_fraction=1.0)) == 104.0
        assert fill_price(100.0, 104.0, ExecutionConfig(mode="close_auction")) == 100.0
        got = fill_price(np.array([1.0, 2.0]), np.array([3.0, 0.0]), ExecutionConfig())
        np.testing.assert_array_equal(got, [2.0, 1.0])

    @given(hst.floats(1.0, 1e4), hst.floats(1.0, 1e4), hst.floats(1e-6, 1.0))
    def test_fill_between_closes(self, p0, p1, f):
        p = fill_price(p0, p1, ExecutionConfig(fill_fraction=f))
        assert min(p0, p1) - 1e-9 * p0 <= p <= max(p0, p1) + 1e-9 * p0


class TestMetrics:
    def test_downside_deviation_full_sample(self):
        # Squares of the shortfalls (0.02, 0.04) over all four observations.
        r = [0.01, -0.02, 0.03, -0.04]
        assert downside_deviation(r) == pytest.approx(math.sqrt((0.0004 + 0.0016) / 4), rel=1e-15)
        # MAR shifts the threshold: shortfalls below 0.01 are 0.03 and 0.05.
        assert downside_deviation(r, mar=0.01) == pytest.approx(math.sqrt((0.0009 + 0.0025) / 4), rel=1e-14)
        assert downside_deviation([0.01, 0.02]) == 0.0

    @given(small_returns, hst.floats(-0.1, 0.1))
    def test_downside_deviation_bounded_by_rms(self, r, mar):
        dd = downside_deviation(r, mar)
        assert 0.0 <= dd <= math.sqrt(np.mean((r - mar) ** 2)) * (1 + 1e-12) + 1e-300

    def test_annualisation(self):
        assert annualisation_factor(252) == math.sqrt(252)
        for bad in (0, -1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="periods_per_year"):
                check_periods_per_year(bad)

    def test_wealth_and_drawdown_known(self):
        r = [-0.1, 0.5, -0.5]
        np.testing.assert_allclose(wealth_curve(r), [1.0, 0.9, 1.35, 0.675], rtol=1e-15)
        assert wealth_curve(r, include_start=False).shape == (3,)
        # Peak 1.35 to trough 0.675 is 50%.
        assert max_drawdown_from_returns(r) == pytest.approx(0.5, rel=1e-15)
        # A first-period loss counts only when the start is in the peak.
        assert max_drawdown_from_returns([-0.1]) == pytest.approx(0.1, rel=1e-15)
        assert max_drawdown_from_returns([-0.1], include_start=False) == 0.0
        lr = [math.log(0.8), math.log(1.5)]
        assert max_drawdown_from_returns(lr, "log") == pytest.approx(0.2, rel=1e-14)
        with pytest.raises(ValueError, match="compounding"):
            wealth_curve(r, "continuous")  # type: ignore[arg-type]

    def test_equity_and_pnl_drawdown(self):
        np.testing.assert_array_equal(drawdown_series([1.0, 3.0, 2.0, 4.0]), [0.0, 0.0, -1.0, 0.0])
        assert max_drawdown_from_equity([1.0, 3.0, 2.0, 4.0]) == 1.0
        dd = max_drawdown_from_equity([1.0, 2.0])
        assert dd == 0.0 and math.copysign(1.0, dd) == 1.0
        # Equity from 0: -1 then -3 is a drawdown of 3 from the starting 0.
        assert max_drawdown_from_pnl([-1.0, -2.0, 5.0]) == 3.0
        assert max_drawdown_from_pnl([]) == 0.0

    @given(small_returns)
    def test_drawdown_in_unit_interval(self, r):
        for start in (True, False):
            dd = max_drawdown_from_returns(r, include_start=start)
            assert 0.0 <= dd <= 1.0
        assert max_drawdown_from_returns(r, include_start=True) >= max_drawdown_from_returns(
            r, include_start=False
        )


class TestValidation:
    def test_require_finite(self):
        require_finite(a=1.0, b=None, c=[1.0, 2.0])
        with pytest.raises(ValueError, match="c must be finite"):
            require_finite(a=1.0, c=[1.0, float("inf")])

    def test_finite_scalar(self):
        assert finite_scalar("x", 2) == 2.0
        with pytest.raises(ValueError, match="x must be finite"):
            finite_scalar("x", float("nan"))

    def test_finite_series(self):
        np.testing.assert_array_equal(finite_series([1, 2], "p"), [1.0, 2.0])
        with pytest.raises(ValueError, match="one-dimensional"):
            finite_series([[1.0]], "p")
        with pytest.raises(ValueError, match="p contains NaN or infinite values; fix it"):
            finite_series([1.0, np.nan], "p", nonfinite_hint="; fix it")
        with pytest.raises(ValueError, match=r"p must be strictly positive \(logged\), found min=0.0"):
            finite_series([1.0, 0.0], "p", positive=True, positive_reason=" (logged)")
