"""CIP arbitrage detector: no false positives inside costs, planted
violations detected with exact P&L."""

import pytest

from fx_rates.arbitrage import (
    CIPQuotes,
    detect_cip_arbitrage,
    no_arb_bounds,
)


def make_quotes(fwd_mid=None, spread_pips=2.0, rate_spread_bp=5.0, tau=0.5,
                spot_mid=1.0850, r_d=0.040, r_f=0.025, fwd_bid=None, fwd_ask=None):
    """Consistent two-sided market around CIP mid, unless forwards overridden."""
    half_s = spread_pips * 1e-4 / 2
    half_r = rate_spread_bp * 1e-4 / 2
    if fwd_mid is None:
        fwd_mid = spot_mid * (1 + r_d * tau) / (1 + r_f * tau)
    if fwd_bid is None:
        fwd_bid = fwd_mid - half_s
    if fwd_ask is None:
        fwd_ask = fwd_mid + half_s
    return CIPQuotes(
        spot_bid=spot_mid - half_s, spot_ask=spot_mid + half_s,
        fwd_bid=fwd_bid, fwd_ask=fwd_ask,
        dom_rate_bid=r_d - half_r, dom_rate_ask=r_d + half_r,
        for_rate_bid=r_f - half_r, for_rate_ask=r_f + half_r,
        tau=tau,
    )


class TestNoFalsePositives:
    def test_cip_consistent_quotes_no_arbitrage(self):
        res = detect_cip_arbitrage(make_quotes())
        assert not res.is_arbitrage
        assert res.direction == "none"
        assert res.pnl == 0.0

    def test_forward_anywhere_inside_band_no_arbitrage(self):
        q = make_quotes()
        lo, hi = no_arb_bounds(q)
        for f in [lo + 1e-6, (lo + hi) / 2, hi - 1e-6]:
            res = detect_cip_arbitrage(make_quotes(fwd_bid=f, fwd_ask=f))
            assert not res.is_arbitrage

    def test_wider_spreads_absorb_bigger_deviation(self):
        # a mid-CIP deviation that flags with tight spreads is absorbed by
        # wide ones — the post-2008 'CIP arbitrage that is not' in miniature
        q_mid = make_quotes()
        deviated = q_mid.fwd_ask + 8e-4  # forward 8 pips above CIP mid
        tight = make_quotes(fwd_bid=deviated, fwd_ask=deviated + 1e-4,
                            spread_pips=1.0, rate_spread_bp=1.0)
        wide = make_quotes(fwd_bid=deviated, fwd_ask=deviated + 1e-4,
                           spread_pips=10.0, rate_spread_bp=50.0)
        assert detect_cip_arbitrage(tight).is_arbitrage
        assert not detect_cip_arbitrage(wide).is_arbitrage

    def test_min_pnl_threshold_suppresses_marginal_arb(self):
        q = make_quotes()
        f = no_arb_bounds(q)[1] + 2e-5  # just above the band
        marginal = make_quotes(fwd_bid=f, fwd_ask=f + 1e-5)
        assert detect_cip_arbitrage(marginal).is_arbitrage
        assert not detect_cip_arbitrage(marginal, min_pnl=1e-3).is_arbitrage


class TestPlantedViolations:
    def test_rich_forward_detected_with_exact_pnl(self):
        q0 = make_quotes()
        f = no_arb_bounds(q0)[1] + 50e-4  # 50 pips above the upper bound
        q = make_quotes(fwd_bid=f, fwd_ask=f + 1e-4)
        res = detect_cip_arbitrage(q)
        assert res.is_arbitrage and res.direction == "sell_forward"
        # independent hand computation of the round trip per 1 USD borrowed
        eur_bought = 1.0 / q.spot_ask
        eur_at_t = eur_bought * (1.0 + q.for_rate_bid * q.tau)
        usd_received = eur_at_t * q.fwd_bid
        usd_owed = 1.0 + q.dom_rate_ask * q.tau
        assert res.pnl == pytest.approx(usd_received - usd_owed, abs=1e-10)

    def test_cheap_forward_detected_with_exact_pnl(self):
        q0 = make_quotes()
        f = no_arb_bounds(q0)[0] - 50e-4
        q = make_quotes(fwd_bid=f - 1e-4, fwd_ask=f)
        res = detect_cip_arbitrage(q)
        assert res.is_arbitrage and res.direction == "buy_forward"
        # per 1 EUR borrowed: sell spot, invest USD, buy EUR repayment forward
        usd_deployed = q.spot_bid
        usd_at_t = usd_deployed * (1.0 + q.dom_rate_bid * q.tau)
        usd_cost_of_repay = q.fwd_ask * (1.0 + q.for_rate_ask * q.tau)
        expected = (usd_at_t - usd_cost_of_repay) / usd_deployed
        assert res.pnl == pytest.approx(expected, abs=1e-10)

    def test_bounds_are_the_detection_frontier(self):
        q0 = make_quotes()
        lo, hi = no_arb_bounds(q0)
        eps = 1e-7
        assert detect_cip_arbitrage(make_quotes(fwd_bid=hi + eps, fwd_ask=hi + eps)).is_arbitrage
        assert not detect_cip_arbitrage(make_quotes(fwd_bid=hi - eps, fwd_ask=hi - eps)).is_arbitrage
        assert detect_cip_arbitrage(make_quotes(fwd_bid=lo - eps, fwd_ask=lo - eps)).is_arbitrage
        assert not detect_cip_arbitrage(make_quotes(fwd_bid=lo + eps, fwd_ask=lo + eps)).is_arbitrage

    def test_negative_foreign_rates_supported(self):
        # EUR 2019: negative deposit rates; arbitrage math must still work
        res = detect_cip_arbitrage(
            make_quotes(r_d=0.02, r_f=-0.005)
        )
        assert not res.is_arbitrage
        f = no_arb_bounds(make_quotes(r_d=0.02, r_f=-0.005))[1] + 30e-4
        res2 = detect_cip_arbitrage(
            make_quotes(r_d=0.02, r_f=-0.005, fwd_bid=f, fwd_ask=f + 1e-4)
        )
        assert res2.is_arbitrage


class TestValidation:
    def test_crossed_quotes_rejected(self):
        with pytest.raises(ValueError, match="exceeds ask"):
            CIPQuotes(1.09, 1.08, 1.10, 1.11, 0.03, 0.031, 0.02, 0.021, 0.5)

    def test_bad_tau_rejected(self):
        with pytest.raises(ValueError, match="tau"):
            CIPQuotes(1.08, 1.09, 1.10, 1.11, 0.03, 0.031, 0.02, 0.021, 0.0)

    def test_nonpositive_price_rejected(self):
        with pytest.raises(ValueError, match="> 0"):
            CIPQuotes(1.08, 1.09, -1.10, 1.11, 0.03, 0.031, 0.02, 0.021, 0.5)
