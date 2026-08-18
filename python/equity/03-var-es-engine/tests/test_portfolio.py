"""Portfolio construction, P&L revaluation and delta-gamma approximation."""

import numpy as np
import pytest

from eq_var import (
    EquityPosition,
    FuturePosition,
    OptionPosition,
    Portfolio,
    RiskFactor,
    bs_greeks,
    bs_price,
)
from eq_var.data import demo_portfolio


def make_linear_portfolio() -> Portfolio:
    factors = {
        "STK": RiskFactor("STK", "equity", 100.0),
        "IDX": RiskFactor("IDX", "index", 2000.0),
    }
    positions = [
        EquityPosition(name="stk", factor="STK", shares=1000.0),
        FuturePosition(name="fut", factor="IDX", contracts=-2.0, multiplier=50.0),
    ]
    return Portfolio(positions, factors)


def make_option_portfolio(contracts: float = 10.0, kind: str = "call") -> Portfolio:
    factors = {
        "SPX": RiskFactor("SPX", "index", 5000.0),
        "SPX_IV": RiskFactor("SPX_IV", "vol", 0.20),
    }
    opt = OptionPosition(
        name="opt",
        underlier="SPX",
        vol_factor="SPX_IV",
        strike=5000.0,
        expiry=0.5,
        rate=0.03,
        div_yield=0.01,
        kind=kind,
        contracts=contracts,
        multiplier=100.0,
    )
    return Portfolio([opt], factors)


class TestLinearPnl:
    def test_equity_pnl_exact(self):
        pf = make_linear_portfolio()
        scen = np.array([[0.02, 0.0]])
        # 1000 shares * 100 * 2% = 2000
        assert pf.pnl(scen, "full")[0] == pytest.approx(2000.0, abs=1e-10)

    def test_future_pnl_exact(self):
        pf = make_linear_portfolio()
        scen = np.array([[0.0, -0.01]])
        # -2 * 50 * 2000 * -1% = +2000
        assert pf.pnl(scen, "full")[0] == pytest.approx(2000.0, abs=1e-10)

    def test_linear_positions_identical_under_both_methods(self):
        pf = make_linear_portfolio()
        rng = np.random.default_rng(0)
        scen = rng.normal(0, 0.02, size=(50, 2))
        np.testing.assert_allclose(pf.pnl(scen, "full"), pf.pnl(scen, "delta_gamma"))

    def test_delta_exposures_hand_computed(self):
        pf = make_linear_portfolio()
        np.testing.assert_allclose(
            pf.delta_exposures(), [1000 * 100.0, -2 * 50 * 2000.0]
        )


class TestBlackScholes:
    def test_put_call_parity(self):
        s, k, tau, r, q, v = 100.0, 95.0, 0.5, 0.03, 0.01, 0.25
        call = bs_price(s, k, tau, r, q, v, "call")
        put = bs_price(s, k, tau, r, q, v, "put")
        parity = s * np.exp(-q * tau) - k * np.exp(-r * tau)
        assert call - put == pytest.approx(parity, abs=1e-10)

    def test_greeks_vs_finite_differences(self):
        s, k, tau, r, q, v = 100.0, 105.0, 0.25, 0.02, 0.0, 0.3
        g = bs_greeks(s, k, tau, r, q, v, "put")
        h = 1e-4
        delta_fd = (bs_price(s + h, k, tau, r, q, v, "put") - bs_price(s - h, k, tau, r, q, v, "put")) / (2 * h)
        gamma_fd = (
            bs_price(s + h, k, tau, r, q, v, "put")
            - 2 * bs_price(s, k, tau, r, q, v, "put")
            + bs_price(s - h, k, tau, r, q, v, "put")
        ) / h**2
        vega_fd = (bs_price(s, k, tau, r, q, v + h, "put") - bs_price(s, k, tau, r, q, v - h, "put")) / (2 * h)
        assert g["delta"] == pytest.approx(delta_fd, abs=1e-6)
        assert g["gamma"] == pytest.approx(gamma_fd, abs=1e-5)
        assert g["vega"] == pytest.approx(vega_fd, abs=1e-5)

    def test_zero_tau_is_intrinsic(self):
        assert bs_price(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, "call") == pytest.approx(10.0)
        assert bs_price(90.0, 100.0, 0.0, 0.05, 0.0, 0.2, "call") == 0.0

    def test_negative_inputs_raise(self):
        with pytest.raises(ValueError):
            bs_price(100.0, 100.0, -0.1, 0.0, 0.0, 0.2, "call")
        with pytest.raises(ValueError):
            bs_price(100.0, 100.0, 0.1, 0.0, 0.0, -0.2, "call")


class TestOptionPnl:
    def test_full_reval_matches_direct_bs_difference(self):
        pf = make_option_portfolio()
        opt = pf.positions[0]
        scen = np.array([[-0.05, 0.03]])
        pnl = pf.pnl(scen, "full")[0]
        p0 = bs_price(5000.0, 5000.0, 0.5, 0.03, 0.01, 0.20, "call")
        p1 = bs_price(4750.0, 5000.0, 0.5, 0.03, 0.01, 0.23, "call")
        assert pnl == pytest.approx(10 * 100 * (p1 - p0), rel=1e-12)

    def test_delta_gamma_matches_taylor_hand_computed(self):
        pf = make_option_portfolio()
        g = bs_greeks(5000.0, 5000.0, 0.5, 0.03, 0.01, 0.20, "call")
        ds, dv = 5000.0 * 0.01, 0.02
        expected = 10 * 100 * (g["delta"] * ds + 0.5 * g["gamma"] * ds**2 + g["vega"] * dv)
        assert pf.pnl(np.array([[0.01, 0.02]]), "delta_gamma")[0] == pytest.approx(expected, rel=1e-12)

    def test_approximation_error_grows_with_shock_size(self):
        pf = make_option_portfolio()
        errs = []
        for shock in (0.01, 0.05, 0.10, 0.20):
            scen = np.array([[-shock, 0.0]])
            errs.append(abs(pf.approximation_error(scen)[0]))
        assert errs == sorted(errs)
        assert errs[-1] > errs[0] * 10  # error is O(dS^3): strongly increasing

    def test_long_gamma_reduces_loss_vs_pure_delta(self):
        pf = make_option_portfolio(contracts=10.0)
        scen = np.array([[-0.10, 0.0]])
        dg_pnl = pf.pnl(scen, "delta_gamma")[0]
        delta_only = float(pf.delta_exposures() @ scen[0])
        assert dg_pnl > delta_only  # +0.5*gamma*dS^2 > 0 for a long option

    def test_short_gamma_worsens_loss_vs_pure_delta(self):
        pf = make_option_portfolio(contracts=-10.0)
        scen = np.array([[-0.10, 0.0]])
        assert pf.pnl(scen, "delta_gamma")[0] < float(pf.delta_exposures() @ scen[0])

    def test_gamma_matrix_sign(self):
        long_g = make_option_portfolio(10.0).gamma_matrix()
        short_g = make_option_portfolio(-10.0).gamma_matrix()
        assert long_g[0, 0] > 0
        assert short_g[0, 0] < 0
        assert long_g[1, 1] == 0.0  # no vol-vol gamma for vanilla BS


class TestPortfolioStructure:
    def test_unknown_factor_raises(self):
        with pytest.raises(ValueError, match="unknown risk factors"):
            Portfolio([EquityPosition(name="x", factor="MISSING", shares=1.0)], {})

    def test_scenario_shape_mismatch_raises(self):
        pf = make_linear_portfolio()
        with pytest.raises(ValueError, match="columns"):
            pf.pnl(np.zeros((5, 3)))

    def test_invalid_method_raises(self):
        pf = make_linear_portfolio()
        with pytest.raises(ValueError, match="method"):
            pf.pnl(np.zeros((1, 2)), method="quadratic")

    def test_negative_factor_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            RiskFactor("BAD", "equity", -5.0)

    def test_value_hand_computed(self):
        pf = make_linear_portfolio()
        assert pf.value() == pytest.approx(1000 * 100.0)  # futures contribute 0

    def test_demo_portfolio_structure(self):
        pf = demo_portfolio()
        assert pf.n_factors == 4
        assert pf.factor_names == ["AAPL", "JPM", "SPX", "SPX_IV"]
        expos = pf.delta_exposures()
        assert expos[0] > 0 and expos[1] > 0  # long equities
        assert expos[2] < 0  # short index hedge (future + put delta)
        assert expos[3] > 0  # long vega from puts
