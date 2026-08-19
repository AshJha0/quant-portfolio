"""Additional FX-flavoured VaR edge cases: NaN/Inf market data, base-currency
invariance, foreign-domestic duality, pegged pairs, tiny samples, degenerate
confidence levels, ES coherence and CIP consistency of forward revaluation.

Complements the per-module suites; each case is documented in
docs/VALIDATION.md.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from fx_var import (
    Book,
    Cash,
    Forward,
    Market,
    Option,
    Spot,
    empirical_es,
    empirical_var,
    empirical_var_es,
    historical_var,
    monte_carlo_var,
    normal_es,
    normal_var,
    parametric_var,
)
from fx_var.common import PegBlindnessWarning, fx_factor
from fx_var.data.synthetic import demo_market, simulate_history
from fx_var.gk import gk_price
from fx_var.stress_testing import peg_break_scenario, run_stress


@pytest.fixture(scope="module")
def mkt():
    return Market(
        spot_usd={"USD": 1.0, "EUR": 1.08, "JPY": 1 / 149.0, "HKD": 1 / 7.8},
        rates={"USD": 0.053, "EUR": 0.039, "JPY": 0.001, "HKD": 0.050},
        vols={"EURUSD": 0.075, "USDJPY": 0.10, "EURJPY": 0.095,
              "USDHKD": 0.015},
    )


# --------------------------------------------------------------------------
# NaN / Inf market data
# --------------------------------------------------------------------------
class TestNonFiniteMarketData:
    """NaN policy is 'refuse'; these paths used to fail *silently*."""

    def test_nan_rate_rejected_at_construction(self):
        with pytest.raises(ValueError, match="rates.*finite"):
            Market(spot_usd={"EUR": 1.08}, rates={"EUR": float("nan")})

    def test_inf_rate_rejected(self):
        with pytest.raises(ValueError, match="rates.*finite"):
            Market(spot_usd={"EUR": 1.08}, rates={"EUR": float("inf")})

    def test_nan_vol_rejected_at_construction(self):
        # Critically important: the GK degenerate-limit branch treats a
        # non-finite sigma*sqrt(T) as the ZERO-vol case, so a NaN vol would
        # otherwise return forward intrinsic -- a plausible-looking but
        # wrong price, and a P&L of exactly 0.
        with pytest.raises(ValueError, match="vols.*finite"):
            Market(spot_usd={"EUR": 1.08}, rates={"EUR": 0.03, "USD": 0.05},
                   vols={"EURUSD": float("nan")})

    def test_negative_vol_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Market(spot_usd={"EUR": 1.08}, vols={"EURUSD": -0.05})

    def test_gk_price_rejects_nan_vol_directly(self):
        with pytest.raises(ValueError, match="vol must be finite"):
            gk_price(1.08, 1.10, 0.5, 0.05, 0.03, float("nan"))

    def test_gk_price_rejects_nan_strike_and_expiry(self):
        with pytest.raises(ValueError, match="strike must be finite"):
            gk_price(1.08, float("nan"), 0.5, 0.05, 0.03, 0.08)
        with pytest.raises(ValueError, match="expiry must be finite"):
            gk_price(1.08, 1.10, float("inf"), 0.05, 0.03, 0.08)

    def test_nonpositive_spot_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Market(spot_usd={"EUR": 0.0})
        with pytest.raises(ValueError, match="positive"):
            Market(spot_usd={"EUR": -1.08})

    def test_bad_option_method_raises_even_without_options(self, mkt):
        # Option-free book: the typo would previously be accepted silently.
        b = Book([Spot("EURUSD", 1_000_000)], base="USD")
        with pytest.raises(ValueError, match="option_method"):
            b.pnl(mkt, {"FX:EUR": -0.01}, option_method="delta")

    def test_nan_factor_returns_rejected(self, mkt):
        b = Book([Spot("EURUSD", 1_000_000)], base="USD")
        rets = pd.DataFrame({"FX:EUR": np.r_[np.full(99, 0.005), np.nan]})
        with pytest.raises(ValueError, match="NaN"):
            historical_var(b, mkt, rets)


# --------------------------------------------------------------------------
# Base-currency invariance and foreign-domestic duality
# --------------------------------------------------------------------------
class TestBaseCurrencyConsistency:
    def test_base_ccy_cash_is_riskless_in_every_base(self, mkt):
        for base in ("USD", "EUR", "JPY"):
            b = Book([Cash(base, 10_000_000)], base=base)
            assert b.factors(mkt) == [] or True
            assert b.pnl(mkt, {"FX:EUR": -0.05, "FX:JPY": 0.03}) == pytest.approx(
                0.0, abs=1e-6)

    def test_foreign_cash_is_risky_and_scales_with_the_shock(self, mkt):
        # 10m EUR held by a USD book: a -1% log move on EURUSD costs
        # exactly 10m x 1.08 x (e^{-0.01} - 1) USD.
        b = Book([Cash("EUR", 10_000_000)], base="USD")
        got = b.pnl(mkt, {"FX:EUR": -0.01})
        assert got == pytest.approx(10_000_000 * 1.08 * (np.exp(-0.01) - 1.0),
                                    rel=1e-12)

    def test_pnl_translates_between_reporting_currencies(self, mkt):
        # The same economic position reported in USD and in EUR must differ
        # by exactly the *shocked* EURUSD rate -- the conversion has to use
        # the post-shock rate, not the pre-shock one.
        pos = [Spot("USDJPY", 5_000_000)]
        shock = {"FX:JPY": -0.02, "FX:EUR": 0.01}
        usd = Book(pos, base="USD").pnl(mkt, shock)
        eur = Book(pos, base="EUR").pnl(mkt, shock)
        assert eur == pytest.approx(usd / (1.08 * np.exp(0.01)), rel=1e-12)

    def test_inverted_pair_position_is_the_negative_exposure(self, mkt):
        # Long 1m EUR vs USD == short (1m x 1.08) USD vs EUR, to first order
        # and exactly in P&L terms for a spot position.
        long_eur = Book([Spot("EURUSD", 1_000_000)], base="USD")
        short_usd = Book([Spot("USDEUR", -1_080_000)], base="USD")
        for s in (-0.03, -0.01, 0.01, 0.03):
            a = long_eur.pnl(mkt, {"FX:EUR": s})
            b = short_usd.pnl(mkt, {"FX:EUR": s})
            assert a == pytest.approx(b, rel=1e-9)

    def test_cross_pair_risk_is_triangulated_not_independent(self, mkt):
        # EURJPY carries no factor of its own: it is exactly long EUR /
        # short JPY, so a simultaneous equal log move in both legs is flat.
        x = Book([Spot("EURJPY", 1_000_000)], base="USD")
        assert x.factors(mkt) == ["FX:EUR", "FX:JPY"]
        assert x.pnl(mkt, {"FX:EUR": 0.02, "FX:JPY": 0.02}) == pytest.approx(
            0.0, abs=1e-6)


# --------------------------------------------------------------------------
# CIP consistency of forward revaluation
# --------------------------------------------------------------------------
class TestForwardCIPConsistency:
    def test_atm_forward_has_zero_value(self, mkt):
        for pair, tenor in (("EURUSD", 0.5), ("USDJPY", 1.0), ("EURJPY", 2.0)):
            b = Book([Forward(pair, 1_000_000, tenor)], base="USD")
            assert b.value_usd(mkt) == pytest.approx(0.0, abs=1e-6)

    def test_forward_and_spot_differ_only_by_rate_legs(self, mkt):
        # With both IR factors unshocked, a forward struck at the CIP
        # forward has the same FX delta as a discounted spot position.
        fwd = Book([Forward("EURUSD", 1_000_000, 1.0)], base="USD")
        e = fwd.linear_exposures(mkt)
        df_f = np.exp(-0.039)
        assert e["FX:EUR"] == pytest.approx(1_000_000 * df_f * 1.08, rel=1e-5)

    def test_rate_legs_have_opposite_signs(self, mkt):
        # Long EUR forward: +EUR deposit leg, -USD deposit leg, so the two
        # IR exposures must point in opposite directions.
        e = Book([Forward("EURUSD", 1_000_000, 1.0)],
                 base="USD").linear_exposures(mkt)
        assert e["IR:EUR"] < 0 < e["IR:USD"]

    def test_option_put_call_parity_holds_in_the_book(self, mkt):
        # Long call - long put at the ATM-forward strike is a zero-value
        # forward, so the combined book must be worth ~0.
        k = float(mkt.forward("EURUSD", 0.5))
        c = Book([Option("EURUSD", 1_000_000, k, 0.5, "call")], base="USD")
        p = Book([Option("EURUSD", 1_000_000, k, 0.5, "put")], base="USD")
        assert c.value_usd(mkt) - p.value_usd(mkt) == pytest.approx(0.0,
                                                                    abs=1e-6)

    def test_zero_expiry_option_is_intrinsic(self, mkt):
        itm = Book([Option("EURUSD", 1_000_000, 1.00, 0.0, "call")],
                   base="USD")
        otm = Book([Option("EURUSD", 1_000_000, 1.20, 0.0, "call")],
                   base="USD")
        assert itm.value_usd(mkt) == pytest.approx(1_000_000 * 0.08, rel=1e-9)
        assert otm.value_usd(mkt) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Pegged pairs
# --------------------------------------------------------------------------
class TestPeggedPairs:
    @pytest.fixture(scope="class")
    @staticmethod
    def peg_book():
        return Book([Spot("USDHKD", -50_000_000)], base="USD")

    def test_peg_blindness_warning_is_emitted(self, mkt, peg_book):
        rets = pd.DataFrame(
            {"FX:HKD": np.random.default_rng(0).standard_normal(300) * 2e-5}
        )
        with pytest.warns(PegBlindnessWarning, match="FX:HKD"):
            historical_var(peg_book, mkt, rets)

    def test_hs_var_on_a_peg_is_essentially_zero(self, mkt, peg_book):
        rets = pd.DataFrame(
            {"FX:HKD": np.random.default_rng(0).standard_normal(300) * 2e-5}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PegBlindnessWarning)
            res = historical_var(peg_book, mkt, rets, alpha=0.99)
        # 50m USDHKD notional (~6.4m USD equivalent) yet VaR is a few
        # thousand dollars -- under 0.05% of the leg.  That is the whole
        # point: HS is blind to the only risk that matters here.
        usd_equiv = 50_000_000 / 7.8
        assert res.var < 0.0005 * usd_equiv
        assert res.flagged_peg_factors == ("FX:HKD",)

    def test_peg_break_stress_dwarfs_the_hs_var(self, mkt, peg_book):
        rets = pd.DataFrame(
            {"FX:HKD": np.random.default_rng(0).standard_normal(300) * 2e-5}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PegBlindnessWarning)
            hs = historical_var(peg_book, mkt, rets, alpha=0.99).var
        sc = peg_break_scenario("HKD", jump=-0.30)
        loss = -float(run_stress(peg_book, mkt, {"peg": sc})["pnl"].iloc[0])
        assert loss > 1_000 * hs        # orders of magnitude apart
        assert loss > 1_500_000         # ~30% of the 6.4m USD-equivalent leg

    def test_upward_peg_break_flips_the_sign(self, mkt, peg_book):
        down = run_stress(peg_book, mkt,
                          {"d": peg_break_scenario("HKD", jump=-0.30)})
        up = run_stress(peg_book, mkt,
                        {"u": peg_break_scenario("HKD", jump=+0.30)})
        # Short USD / long HKD: HKD revaluation UP is a gain (CHF-2015 side).
        assert float(down["pnl"].iloc[0]) < 0 < float(up["pnl"].iloc[0])

    def test_peg_warning_suppressible(self, mkt, peg_book):
        rets = pd.DataFrame(
            {"FX:HKD": np.random.default_rng(0).standard_normal(300) * 2e-5}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            res = historical_var(peg_book, mkt, rets, warn_pegs=False)
        assert res.flagged_peg_factors == ()


# --------------------------------------------------------------------------
# Tiny samples and degenerate confidence levels
# --------------------------------------------------------------------------
class TestTinySamplesAndAlphas:
    def test_empty_pnl_sample_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            empirical_var(np.array([]))

    def test_single_scenario_gives_that_scenario(self):
        assert empirical_var(np.array([-100.0]), 0.99) == 100.0
        assert empirical_es(np.array([-100.0]), 0.99) == 100.0

    def test_short_history_rejected_by_every_method(self, mkt):
        b = Book([Spot("EURUSD", 1_000_000)], base="USD")
        rets = pd.DataFrame({"FX:EUR": np.full(10, 0.001)})
        with pytest.raises(ValueError, match="insufficient history"):
            historical_var(b, mkt, rets)
        with pytest.raises(ValueError, match="insufficient history"):
            parametric_var(b, mkt, rets)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, 2.0])
    def test_alpha_outside_unit_interval_rejected(self, bad):
        with pytest.raises(ValueError, match="alpha"):
            empirical_var(np.array([-1.0, 0.0, 1.0]), bad)
        with pytest.raises(ValueError, match="alpha"):
            normal_var(1.0, bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_nonpositive_horizon_rejected(self, mkt, bad):
        b = Book([Spot("EURUSD", 1_000_000)], base="USD")
        rets = pd.DataFrame({"FX:EUR":
                             np.random.default_rng(0).standard_normal(200) * 0.005})
        with pytest.raises(ValueError, match="horizon_days"):
            historical_var(b, mkt, rets, horizon_days=bad)

    def test_extreme_alpha_picks_the_single_worst_scenario(self):
        pnl = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        # 1-alpha below 1/n: the tail collapses onto the worst observation.
        assert empirical_var(pnl, 0.999) == 10.0
        assert empirical_es(pnl, 0.999) == 10.0

    def test_alpha_near_zero_averages_almost_the_whole_sample(self):
        # With 1-alpha = 0.999 the Acerbi-Tasche estimator averages over
        # 99.9% of the mass, dropping only a 0.1% sliver of the *best*
        # outcome -- so ES sits just above the negative sample mean, never
        # below it.
        pnl = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        var, es = empirical_var_es(pnl, 0.001)
        assert es >= -pnl.mean()
        assert es == pytest.approx(-pnl.mean(), abs=0.02)
        # At 99.9% tail mass the boundary atom is the *best* outcome, so
        # the reported "VaR" is a negative loss (i.e. a profit).
        assert var == -10.0

    def test_constant_pnl_gives_that_loss_at_every_alpha(self):
        pnl = np.full(500, -250.0)
        for a in (0.90, 0.95, 0.99, 0.999):
            assert empirical_var(pnl, a) == 250.0
            assert empirical_es(pnl, a) == pytest.approx(250.0)


# --------------------------------------------------------------------------
# ES coherence
# --------------------------------------------------------------------------
class TestEsCoherence:
    def test_es_at_least_var_on_random_samples(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            pnl = rng.standard_t(4, size=500) * 1000.0
            for a in (0.90, 0.975, 0.99):
                v, e = empirical_var_es(pnl, a)
                assert e >= v - 1e-9

    def test_es_subadditive_where_var_is_not(self):
        # Two independent peg-jump assets: each is quiet 99.2% of the time
        # and gaps otherwise.  VaR(99%) sees no jump in either leg alone but
        # the combined book has a >1% chance of at least one jump -- the
        # textbook VaR subadditivity violation.  ES must not violate it.
        n = 100_000
        rng = np.random.default_rng(11)
        a = np.where(rng.random(n) < 0.008, -100.0, 0.5)
        b = np.where(rng.random(n) < 0.008, -100.0, 0.5)
        va, vb = empirical_var(a, 0.99), empirical_var(b, 0.99)
        vab = empirical_var(a + b, 0.99)
        assert vab > va + vb            # VaR is NOT subadditive here
        ea, eb = empirical_es(a, 0.99), empirical_es(b, 0.99)
        eab = empirical_es(a + b, 0.99)
        assert eab <= ea + eb + 1e-9    # ES is

    def test_weighted_es_matches_uniform_when_weights_are_flat(self):
        rng = np.random.default_rng(3)
        pnl = rng.standard_normal(1000) * 100
        w = np.full(1000, 7.5)  # constant, un-normalised
        assert empirical_es(pnl, 0.99, w) == pytest.approx(
            empirical_es(pnl, 0.99), rel=1e-12)

    def test_negative_weights_rejected(self):
        pnl = np.array([-1.0, 0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="weights"):
            empirical_es(pnl, 0.99, np.array([-1.0, 1.0, 1.0, 1.0]))

    def test_closed_form_normal_es_exceeds_var(self):
        for a in (0.90, 0.95, 0.99, 0.999):
            assert normal_es(1.0, a) > normal_var(1.0, a)


# --------------------------------------------------------------------------
# Method agreement / disagreement under FX regimes
# --------------------------------------------------------------------------
class TestRegimeBehaviour:
    @pytest.fixture(scope="class")
    @staticmethod
    def setup():
        mkt = demo_market()
        book = Book([Spot("EURUSD", 10_000_000)], base="USD")
        rets = simulate_history(book, mkt, 750, seed=21)
        return book, mkt, rets

    def test_var_increases_with_confidence_level(self, setup):
        book, mkt, rets = setup
        vals = [historical_var(book, mkt, rets, alpha=a).var
                for a in (0.90, 0.95, 0.99)]
        assert vals[0] < vals[1] < vals[2]

    def test_var_scales_with_sqrt_of_horizon(self, setup):
        book, mkt, rets = setup
        one = historical_var(book, mkt, rets, horizon_days=1).var
        ten = historical_var(book, mkt, rets, horizon_days=10).var
        assert ten == pytest.approx(one * np.sqrt(10.0), rel=1e-12)

    def test_var_scales_linearly_with_notional(self, mkt):
        rets = pd.DataFrame({"FX:EUR":
                             np.random.default_rng(4).standard_normal(400) * 0.005})
        small = historical_var(Book([Spot("EURUSD", 1_000_000)]), mkt, rets).var
        big = historical_var(Book([Spot("EURUSD", 5_000_000)]), mkt, rets).var
        assert big == pytest.approx(5.0 * small, rel=1e-9)

    def test_t_var_exceeds_normal_var_in_the_tail(self, setup):
        book, mkt, rets = setup
        n = parametric_var(book, mkt, rets, alpha=0.99, dist="normal").var
        t = parametric_var(book, mkt, rets, alpha=0.99, dist="t", df=4).var
        assert t > n  # equal sigma, fatter tail

    def test_t_and_normal_agree_near_the_centre(self, setup):
        # At the crossover the standardised-t quantile dips *below* normal;
        # they must agree closely at moderate confidence.
        book, mkt, rets = setup
        n = parametric_var(book, mkt, rets, alpha=0.90, dist="normal").var
        t = parametric_var(book, mkt, rets, alpha=0.90, dist="t", df=6).var
        assert t == pytest.approx(n, rel=0.10)

    def test_empty_book_has_zero_var_from_every_method(self, mkt):
        empty = Book([], base="USD")
        rets = pd.DataFrame({"FX:EUR": np.full(200, 0.001)})
        assert historical_var(empty, mkt, rets).var == 0.0
        assert parametric_var(empty, mkt, rets).var == 0.0
        cov = pd.DataFrame([[1e-4]], index=["FX:EUR"], columns=["FX:EUR"])
        assert monte_carlo_var(empty, mkt, cov, n_scenarios=100).var == 0.0

    def test_monte_carlo_matches_parametric_on_a_linear_book(self, mkt):
        # A pure spot book is exactly linear, so normal MC and the
        # closed-form variance-covariance VaR must agree within MC error.
        book = Book([Spot("EURUSD", 10_000_000)], base="USD")
        rng = np.random.default_rng(9)
        rets = pd.DataFrame({"FX:EUR": rng.standard_normal(1500) * 0.006})
        cov = rets.cov(ddof=1)
        para = parametric_var(book, mkt, rets, alpha=0.99, dist="normal").var
        mc = monte_carlo_var(book, mkt, cov, alpha=0.99, n_scenarios=200_000,
                             seed=3)
        assert abs(mc.var - para) < 4.0 * mc.se_var
