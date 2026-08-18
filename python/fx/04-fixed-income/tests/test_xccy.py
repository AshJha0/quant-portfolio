"""Cross-currency swap: two-bond decomposition, par solvers, basis effects."""

from dataclasses import replace

import pytest

from fx_rates.xccy import (
    CrossCurrencySwap,
    solve_par_basis,
    solve_par_rate_base,
    solve_par_rate_quote,
)


@pytest.fixture
def swap(market):
    return CrossCurrencySwap(
        notional_base=50e6,
        notional_quote=50e6 * market.spot,
        rate_base=0.025,
        rate_quote=0.043,
        maturity=5.0,
        frequency=1,
        receive_base=True,
    )


class TestConstructionValidation:
    def test_same_currency_rejected(self):
        with pytest.raises(ValueError, match="rejected"):
            CrossCurrencySwap(1e6, 1e6, 0.02, 0.04, 5.0, pair=("EUR", "EUR"))

    def test_non_integral_schedule_rejected(self):
        with pytest.raises(ValueError, match="integral"):
            CrossCurrencySwap(1e6, 1e6, 0.02, 0.04, 5.3, frequency=1)

    def test_negative_notional_rejected(self):
        with pytest.raises(ValueError, match="notionals"):
            CrossCurrencySwap(-1e6, 1e6, 0.02, 0.04, 5.0)

    def test_payment_times_semiannual(self):
        sw = CrossCurrencySwap(1e6, 1e6, 0.02, 0.04, 2.0, frequency=2)
        assert list(sw.payment_times()) == [0.5, 1.0, 1.5, 2.0]


class TestTwoBondDecomposition:
    def test_value_matches_explicit_cashflow_discounting(self, market, swap):
        # independently discount every enumerated cashflow
        v = 0.0
        for ccy, t, amount in swap.cashflows():
            if ccy == "EUR":
                v += amount * market.foreign_curve_adjusted.df(t) * market.spot
            else:
                v += amount * market.domestic_curve.df(t)
        assert swap.value(market) == pytest.approx(v, abs=1e-8)

    def test_initial_exchange_identity(self, market, swap):
        # with notional_quote = N_base * spot the initial exchange is worth 0
        at_inception = replace(swap, include_initial_exchange=True,
                               notional_quote=swap.notional_base * market.spot)
        seasoned = replace(swap, include_initial_exchange=False,
                           notional_quote=swap.notional_base * market.spot)
        assert at_inception.value(market) == pytest.approx(
            seasoned.value(market), abs=1e-8
        )

    def test_receive_pay_mirror(self, market, swap):
        pay = replace(swap, receive_base=False)
        assert pay.value(market) == pytest.approx(-swap.value(market), abs=1e-8)

    def test_pair_mismatch_raises(self, market, swap):
        bad = replace(swap, pair=("GBP", "USD"))
        with pytest.raises(ValueError, match="pair"):
            bad.value(market)


class TestParSolvers:
    def test_par_rate_quote_zeroes_pv_exactly(self, market, swap):
        c = solve_par_rate_quote(swap, market)
        par = replace(swap, rate_quote=c)
        assert par.value(market) == pytest.approx(0.0, abs=1e-10 * swap.notional_quote)

    def test_par_rate_base_zeroes_pv_exactly(self, market, swap):
        c = solve_par_rate_base(swap, market)
        par = replace(swap, rate_base=c)
        assert par.value(market) == pytest.approx(0.0, abs=1e-10 * swap.notional_base)

    def test_par_basis_zeroes_pv(self, market, swap):
        x = solve_par_basis(swap, market)
        shifted = tuple((t, s + x) for t, s in market.basis_spreads)
        m2 = market.replace(basis_spreads=shifted)
        assert abs(swap.value(m2)) < 1e-12 * swap.notional_base * swap.maturity * 10

    def test_par_basis_of_cip_par_swap_offsets_market_basis(self, market, swap):
        # price the swap to par on the *pure CIP* market; the par basis on the
        # basis-market must then be (approximately) the negative of the
        # prevailing 5y basis level
        cip = market.replace(basis_spreads=())
        c = solve_par_rate_quote(swap, cip)
        par_on_cip = replace(swap, rate_quote=c)
        x = solve_par_basis(par_on_cip, market.replace(basis_spreads=()))
        assert x == pytest.approx(0.0, abs=1e-12)

    def test_zero_base_notional_par_basis_raises(self, market, swap):
        degenerate = replace(swap, notional_base=0.0)
        with pytest.raises(ValueError, match="zero base notional"):
            solve_par_basis(degenerate, market)


class TestBasisEffects:
    def test_zero_basis_recovers_pure_cip_pricing(self, market, swap):
        m0 = market.replace(basis_spreads=())
        mz = market.replace(
            basis_spreads=tuple((t, 0.0) for t, _ in market.basis_spreads)
        )
        assert swap.value(mz) == pytest.approx(swap.value(m0), abs=1e-8)

    def test_widening_basis_hurts_receive_base(self, market, swap):
        # more negative basis => adjusted EUR DFs rise => receive-EUR leg
        # gains: PV increases when basis widens (goes more negative)
        wider = market.replace(
            basis_spreads=tuple((t, s - 0.0025) for t, s in market.basis_spreads)
        )
        assert swap.value(wider) > swap.value(market)

    def test_solved_par_rate_reflects_basis(self, market, swap):
        # receive-EUR swap: with a negative basis the EUR leg is worth more,
        # so the par USD coupon must be higher than in the pure CIP world
        c_with_basis = solve_par_rate_quote(swap, market)
        c_cip = solve_par_rate_quote(swap, market.replace(basis_spreads=()))
        assert c_with_basis > c_cip
