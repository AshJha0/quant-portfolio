"""Analytic Greeks vs finite differences; FX-specific rho signs."""

import math

import pytest

from fx_options import (analytic_greeks, finite_difference_greeks, gamma,
                        gk_price, vanna, vega, volga)

CASES = [
    dict(S=1.10, K=1.10, T=0.5, r_d=0.0425, r_f=0.0290, sigma=0.0825),
    dict(S=1.10, K=1.25, T=1.0, r_d=0.0425, r_f=0.0290, sigma=0.0825),
    dict(S=147.5, K=150.0, T=0.5, r_d=0.0050, r_f=0.0525, sigma=0.1075),
    dict(S=1.08, K=1.05, T=1.0, r_d=-0.0075, r_f=-0.0050, sigma=0.065),
]


class TestVsFiniteDifferences:
    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_all_greeks_match_fd(self, case, option_type):
        g = analytic_greeks(**case, option_type=option_type).as_dict()
        fd = finite_difference_greeks(**case, option_type=option_type)
        scale = max(1.0, case["S"])
        for name, fd_val in fd.items():
            tol = 5e-5 * scale if name in ("gamma", "vanna", "volga") \
                else 1e-6 * scale
            assert g[name] == pytest.approx(fd_val, abs=tol), name


class TestRhoSigns:
    def test_call_rho_domestic_positive_rho_foreign_negative(self):
        g = analytic_greeks(**CASES[0], option_type="call")
        assert g.rho_domestic > 0
        assert g.rho_foreign < 0

    def test_put_rho_domestic_negative_rho_foreign_positive(self):
        g = analytic_greeks(**CASES[0], option_type="put")
        assert g.rho_domestic < 0
        assert g.rho_foreign > 0

    def test_rho_foreign_is_minus_st_delta(self):
        # rho_f = -T * S * delta_spot for calls and puts alike.
        for ot in ("call", "put"):
            g = analytic_greeks(**CASES[1], option_type=ot)
            assert g.rho_foreign == pytest.approx(
                -CASES[1]["T"] * CASES[1]["S"] * g.delta_spot, abs=1e-12)


class TestStructure:
    def test_vega_gamma_positive_and_put_call_identical(self):
        c = analytic_greeks(**CASES[0], option_type="call")
        p = analytic_greeks(**CASES[0], option_type="put")
        assert c.vega > 0 and c.gamma > 0
        assert c.vega == pytest.approx(p.vega, abs=1e-14)
        assert c.gamma == pytest.approx(p.gamma, abs=1e-14)
        assert c.vanna == pytest.approx(p.vanna, abs=1e-14)
        assert c.volga == pytest.approx(p.volga, abs=1e-14)

    def test_standalone_helpers_match_dataclass(self):
        case = CASES[2]
        g = analytic_greeks(**case, option_type="call")
        assert vega(**case) == pytest.approx(g.vega, abs=1e-14)
        assert gamma(**case) == pytest.approx(g.gamma, abs=1e-14)
        assert vanna(**case) == pytest.approx(g.vanna, abs=1e-14)
        assert volga(**case) == pytest.approx(g.volga, abs=1e-14)

    def test_forward_delta_relation(self):
        case = CASES[0]
        g = analytic_greeks(**case, option_type="call")
        assert g.delta_forward == pytest.approx(
            g.delta_spot * math.exp(case["r_f"] * case["T"]), abs=1e-14)

    def test_price_field_matches_gk(self):
        case = CASES[3]
        g = analytic_greeks(**case, option_type="put")
        assert g.price == pytest.approx(gk_price(**case, option_type="put"),
                                        abs=1e-14)

    def test_atm_wing_volga_signs(self):
        # Volga ~ 0 near the ATM-DNS point (d1*d2 < 0 between d1=0 and
        # d2=0 strikes) and positive in the wings.
        wing = analytic_greeks(S=1.10, K=1.35, T=0.5, r_d=0.0425,
                               r_f=0.0290, sigma=0.0825, option_type="call")
        assert wing.volga > 0

    def test_theta_recovers_price_decay(self):
        # Price at T-dt ~ price at T + theta*dt (theta is per year, dV/dt).
        case = CASES[0]
        dt = 1e-4
        g = analytic_greeks(**case, option_type="call")
        p_now = gk_price(**case, option_type="call")
        later = dict(case, T=case["T"] - dt)
        p_later = gk_price(**later, option_type="call")
        assert p_later - p_now == pytest.approx(g.theta * dt, abs=1e-7)


class TestValidation:
    def test_t_zero_raises(self):
        with pytest.raises(ValueError):
            analytic_greeks(1.1, 1.1, 0.0, 0.03, 0.01, 0.1, "call")

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            analytic_greeks(1.1, 1.1, 0.5, 0.03, 0.01, 0.0, "call")

    def test_bad_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            analytic_greeks(1.1, 1.1, 0.5, 0.03, 0.01, 0.1, "collar")
