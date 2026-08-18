"""Delta-hedging simulator: P&L statistics, wrong vol, transaction costs."""

import math

import pytest

from fx_options import hedge_frequency_study, simulate_delta_hedge

ARGS = dict(S0=1.10, K=1.1075, T=0.5, r_d=0.0425, r_f=0.0290,
            sigma_true=0.0825)


class TestCorrectVolHedge:
    def test_mean_pnl_near_zero_at_true_vol(self):
        res = simulate_delta_hedge(**ARGS, option_type="call",
                                   n_rebalances=100, n_paths=4000, rng=7)
        se = res.std_pnl / math.sqrt(res.n_paths)
        assert abs(res.mean_pnl) < 3.0 * se
        # And economically small vs the premium.
        assert abs(res.mean_pnl) < 0.02 * res.option_premium

    def test_std_scales_like_inverse_sqrt_n(self):
        lo = simulate_delta_hedge(**ARGS, option_type="call",
                                  n_rebalances=25, n_paths=4000, rng=11)
        hi = simulate_delta_hedge(**ARGS, option_type="call",
                                  n_rebalances=400, n_paths=4000, rng=11)
        ratio = lo.std_pnl / hi.std_pnl
        assert ratio == pytest.approx(math.sqrt(400 / 25), rel=0.25)

    def test_frequency_study_monotone_std(self):
        rows = hedge_frequency_study(**ARGS, option_type="call",
                                     frequencies=(4, 25, 100), n_paths=2000,
                                     rng=3)
        stds = [r["std_pnl"] for r in rows]
        assert stds[0] > stds[1] > stds[2]

    def test_put_hedge_also_flat(self):
        res = simulate_delta_hedge(**ARGS, option_type="put",
                                   n_rebalances=100, n_paths=4000, rng=9)
        se = res.std_pnl / math.sqrt(res.n_paths)
        assert abs(res.mean_pnl) < 3.0 * se


class TestWrongVolAndCosts:
    def test_selling_rich_vol_earns_positive_pnl(self):
        # Sell at 10.25%, realised 8.25%: short gamma at overstated vol
        # collects excess premium -> positive mean P&L.
        res = simulate_delta_hedge(**ARGS, option_type="call",
                                   sigma_hedge=ARGS["sigma_true"] + 0.02,
                                   n_rebalances=100, n_paths=4000, rng=7)
        se = res.std_pnl / math.sqrt(res.n_paths)
        assert res.mean_pnl > 3.0 * se

    def test_selling_cheap_vol_loses(self):
        res = simulate_delta_hedge(**ARGS, option_type="call",
                                   sigma_hedge=ARGS["sigma_true"] - 0.02,
                                   n_rebalances=100, n_paths=4000, rng=7)
        se = res.std_pnl / math.sqrt(res.n_paths)
        assert res.mean_pnl < -3.0 * se

    def test_transaction_costs_reduce_mean_pnl(self):
        free = simulate_delta_hedge(**ARGS, option_type="call",
                                    n_rebalances=100, n_paths=2000, rng=5)
        costly = simulate_delta_hedge(**ARGS, option_type="call",
                                      n_rebalances=100, n_paths=2000, rng=5,
                                      transaction_cost_pips=1.0)
        assert costly.mean_pnl < free.mean_pnl
        assert costly.total_transaction_costs > 0
        assert free.total_transaction_costs == 0.0
        # Same paths (same seed): difference equals average costs.
        assert free.mean_pnl - costly.mean_pnl == pytest.approx(
            costly.total_transaction_costs, rel=0.35)

    def test_costs_grow_with_rebalance_frequency(self):
        rows = hedge_frequency_study(**ARGS, option_type="call",
                                     frequencies=(10, 100), n_paths=1000,
                                     rng=13, transaction_cost_pips=1.0)
        assert rows[1]["mean_costs"] > rows[0]["mean_costs"]

    def test_jpy_pip_size(self):
        res = simulate_delta_hedge(S0=147.5, K=144.0, T=0.5, r_d=0.005,
                                   r_f=0.0525, sigma_true=0.1075,
                                   option_type="call", n_rebalances=50,
                                   n_paths=500, rng=2,
                                   transaction_cost_pips=1.0, pip_size=1e-2)
        assert res.total_transaction_costs > 0


class TestValidation:
    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError):
            simulate_delta_hedge(**ARGS, option_type="call", n_rebalances=0)
        with pytest.raises(ValueError):
            simulate_delta_hedge(**ARGS, option_type="call", n_paths=1)
        with pytest.raises(ValueError, match="sigma_hedge"):
            simulate_delta_hedge(**ARGS, option_type="call", sigma_hedge=-0.1)
        with pytest.raises(ValueError):
            simulate_delta_hedge(**ARGS, option_type="call",
                                 transaction_cost_pips=-1.0)
        with pytest.raises(ValueError):
            simulate_delta_hedge(S0=1.1, K=1.1, T=0.0, r_d=0.03, r_f=0.01,
                                 sigma_true=0.1, option_type="call")
