"""Monte Carlo: statistical agreement with GK, variance reduction, digitals."""

import numpy as np
import pytest

from fx_options import (digital_price, gk_price, mc_digital_price, mc_price)

MKT = dict(S=1.10, K=1.12, T=0.5, r_d=0.0425, r_f=0.0290, sigma=0.0825)


class TestVanillaMC:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_within_three_standard_errors(self, option_type):
        res = mc_price(**MKT, option_type=option_type, n_paths=200_000, rng=42)
        gk = gk_price(**MKT, option_type=option_type)
        assert abs(res.price - gk) < 3.0 * res.std_error

    def test_negative_rates_within_three_se(self):
        args = dict(S=1.08, K=1.08, T=1.0, r_d=-0.0075, r_f=-0.005,
                    sigma=0.065)
        res = mc_price(**args, option_type="call", n_paths=200_000, rng=1)
        assert abs(res.price - gk_price(**args, option_type="call")) \
            < 3.0 * res.std_error

    def test_variance_reduction_works(self):
        # ITM call: strong payoff/S_T correlation makes the control
        # variate bite hard (observed ~70% SE reduction).
        itm = dict(MKT, K=1.05)
        plain = mc_price(**itm, option_type="call", n_paths=50_000, rng=7,
                         antithetic=False, control_variate=False)
        cv = mc_price(**itm, option_type="call", n_paths=50_000, rng=7,
                      antithetic=True, control_variate=True)
        assert cv.std_error < 0.5 * plain.std_error

    def test_variance_reduction_never_hurts_otm(self):
        plain = mc_price(**MKT, option_type="call", n_paths=50_000, rng=7,
                         antithetic=False, control_variate=False)
        cv = mc_price(**MKT, option_type="call", n_paths=50_000, rng=7,
                      antithetic=True, control_variate=True)
        assert cv.std_error < plain.std_error

    def test_seed_reproducibility(self):
        a = mc_price(**MKT, option_type="call", n_paths=10_000, rng=123)
        b = mc_price(**MKT, option_type="call", n_paths=10_000, rng=123)
        assert a.price == b.price and a.std_error == b.std_error

    def test_generator_accepted(self):
        gen = np.random.default_rng(5)
        res = mc_price(**MKT, option_type="call", n_paths=10_000, rng=gen)
        assert res.n_paths == 10_000

    def test_ci_brackets_price(self):
        res = mc_price(**MKT, option_type="put", n_paths=20_000, rng=3)
        assert res.ci_low < res.price < res.ci_high
        assert res.ci_high - res.ci_low == pytest.approx(2 * 1.96 * res.std_error)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="T > 0"):
            mc_price(1.1, 1.1, 0.0, 0.03, 0.01, 0.1, "call")
        with pytest.raises(ValueError, match="n_paths"):
            mc_price(**MKT, option_type="call", n_paths=1)
        with pytest.raises(ValueError):
            mc_price(-1.1, 1.1, 0.5, 0.03, 0.01, 0.1, "call")


class TestDigitals:
    @pytest.mark.parametrize("payout", ["domestic", "foreign"])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_mc_matches_analytic(self, payout, option_type):
        res = mc_digital_price(**MKT, option_type=option_type,
                               payout_currency=payout, n_paths=400_000, rng=42)
        analytic = digital_price(**MKT, option_type=option_type,
                                 payout_currency=payout)
        assert abs(res.price - analytic) < 3.0 * res.std_error

    def test_foreign_digital_uses_n_d1_not_n_d2(self):
        # Measure care: foreign-cash digital = S e^{-r_f T} N(d1); pricing
        # it as e^{-r_f T} N(d2) (the naive 'discount foreign cash at r_f'
        # error) misprices materially.
        from fx_options import d1, d2
        import math
        from scipy.stats import norm
        good = digital_price(**MKT, option_type="call",
                             payout_currency="foreign")
        _d1 = d1(**MKT)
        expected = MKT["S"] * math.exp(-MKT["r_f"] * MKT["T"]) * norm.cdf(_d1)
        assert good == pytest.approx(expected, abs=1e-14)
        naive = math.exp(-MKT["r_f"] * MKT["T"]) * norm.cdf(d2(**MKT))
        assert abs(good - naive) > 1e-3

    def test_digital_parity_domestic(self):
        # Digital call + digital put = discount factor.
        import math
        c = digital_price(**MKT, option_type="call", payout_currency="domestic")
        p = digital_price(**MKT, option_type="put", payout_currency="domestic")
        assert c + p == pytest.approx(math.exp(-MKT["r_d"] * MKT["T"]),
                                      abs=1e-14)

    def test_vanilla_decomposition(self):
        # call = (foreign-cash digital) - K * (domestic-cash digital).
        c = gk_price(**MKT, option_type="call")
        asset_leg = digital_price(**MKT, option_type="call",
                                  payout_currency="foreign")
        cash_leg = digital_price(**MKT, option_type="call",
                                 payout_currency="domestic")
        assert c == pytest.approx(asset_leg - MKT["K"] * cash_leg, abs=1e-14)

    def test_invalid_payout_currency_raises(self):
        with pytest.raises(ValueError, match="payout_currency"):
            digital_price(**MKT, option_type="call", payout_currency="quanto")
        with pytest.raises(ValueError, match="payout_currency"):
            mc_digital_price(**MKT, option_type="call",
                             payout_currency="quanto")
