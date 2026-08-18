"""Z-scores, entry/exit/stop state machine, position sizing."""

import numpy as np
import pandas as pd
import pytest

from eq_pairs.data import business_index, simulate_ou
from eq_pairs.signals import (
    SignalRules,
    generate_signals,
    size_positions,
    time_stop_bars,
    zscore_ou,
    zscore_rolling,
)
from eq_pairs.spread import fit_ou_ols


def _series(vals) -> pd.Series:
    return pd.Series(np.asarray(vals, dtype=float), index=business_index(len(vals)))


class TestZScores:
    def test_rolling_zscore_hand_computed(self):
        s = _series([1.0, 2.0, 3.0, 4.0, 10.0])
        z = zscore_rolling(s, window=3)
        # window at t=4: [3, 4, 10], mean = 17/3, std = sqrt(sum(d^2)/2)
        mean = 17.0 / 3.0
        std = np.sqrt(((3 - mean) ** 2 + (4 - mean) ** 2 + (10 - mean) ** 2) / 2.0)
        assert z.iloc[4] == pytest.approx((10.0 - mean) / std, abs=1e-12)

    def test_rolling_warmup_is_nan(self):
        z = zscore_rolling(_series(np.arange(10.0)), window=5)
        assert z.iloc[:4].isna().all()
        assert z.iloc[4:].notna().all()

    def test_rolling_zero_variance_window_nan_not_inf(self):
        z = zscore_rolling(_series([1.0] * 8 + [2.0]), window=4)
        assert not np.isinf(z.to_numpy()[~np.isnan(z.to_numpy())]).any()
        assert z.iloc[5].item() != z.iloc[5].item() or np.isnan(z.iloc[5])

    def test_rolling_window_validation(self):
        with pytest.raises(ValueError, match="window"):
            zscore_rolling(_series([1.0, 2.0, 3.0]), window=2)

    def test_ou_zscore_known_values(self):
        path = simulate_ou(3000, kappa=0.1, sigma=1.0, mu=2.0, seed=60)
        fit = fit_ou_ols(path)
        s = _series(path[:10])
        z = zscore_ou(s, fit)
        expected = (path[:10] - fit.mu) / fit.stationary_std
        np.testing.assert_allclose(z.to_numpy(), expected, atol=1e-12)

    def test_ou_zscore_no_warmup_nans(self):
        path = simulate_ou(500, kappa=0.1, sigma=1.0, seed=61)
        z = zscore_ou(_series(path), fit_ou_ols(path))
        assert z.notna().all()

    def test_ou_zscore_non_mean_reverting_raises(self):
        fit = fit_ou_ols(1.02 ** np.arange(300))
        with pytest.raises(ValueError, match="not mean-reverting"):
            zscore_ou(_series(np.arange(10.0)), fit)


class TestStateMachine:
    def test_entry_short_and_long(self):
        z = _series([0.0, 2.1, 1.0, -0.1, -1.0, -2.5, 0.1])
        out = generate_signals(z)
        # short at 2.1; exit when z crosses 0 (-0.1); -1.0 re-arms;
        # long at -2.5; exit when z crosses back through 0 (0.1)
        assert list(out["position"]) == [0, -1, -1, 0, 0, 1, 0]
        assert out["event"].iloc[1] == "entry_short"
        assert out["event"].iloc[3] == "exit_mean"
        assert out["event"].iloc[5] == "entry_long"
        assert out["event"].iloc[6] == "exit_mean"

    def test_exit_band(self):
        z = _series([0.0, 2.5, 1.0, 0.6, 0.4])
        out = generate_signals(z, SignalRules(entry_z=2.0, exit_z=0.5, stop_z=4.0))
        # short at 2.5; 1.0 and 0.6 still above exit band; 0.4 <= 0.5 exits
        assert list(out["position"]) == [0, -1, -1, -1, 0]
        assert out["event"].iloc[4] == "exit_mean"

    def test_stop_loss_path(self):
        z = _series([0.0, 2.5, 3.0, 4.2, 3.9])
        out = generate_signals(z)
        assert list(out["position"]) == [0, -1, -1, 0, 0]
        assert out["event"].iloc[3] == "exit_stop"

    def test_time_stop_path(self):
        z = _series([0.0, 2.5, 2.4, 2.3, 2.2, 2.1])
        out = generate_signals(z, SignalRules(max_holding=3))
        assert list(out["position"]) == [0, -1, -1, -1, 0, 0]
        assert out["event"].iloc[4] == "exit_time"

    def test_no_reentry_until_rearmed_after_stop(self):
        z = _series([0.0, 2.5, 4.5, 4.4, 2.6, 1.0, 2.5])
        out = generate_signals(z)
        # stop at 4.5; 4.4 and 2.6 remain outside the entry band -> stay flat;
        # 1.0 re-arms; 2.5 re-enters
        assert list(out["position"]) == [0, -1, 0, 0, 0, 0, -1]
        assert out["event"].iloc[2] == "exit_stop"
        assert out["event"].iloc[6] == "entry_short"

    def test_nan_z_forces_flat_no_nan_leak(self):
        z = _series([np.nan, np.nan, -2.5, np.nan, -2.6, 0.1, np.nan])
        out = generate_signals(z)
        assert out["position"].notna().all()
        assert list(out["position"]) == [0, 0, 1, 0, 1, 0, 0]
        assert out["event"].iloc[3] == "exit_nan"

    def test_position_uses_info_through_t_only(self):
        # entry happens exactly on the crossing bar, not before
        z = _series([1.9, 1.99, 2.0, 0.0])
        out = generate_signals(z)
        assert list(out["position"]) == [0, 0, -1, 0]

    def test_symmetric_long_stop(self):
        z = _series([0.0, -2.5, -4.1, -1.0])
        out = generate_signals(z)
        assert list(out["position"]) == [0, 1, 0, 0]
        assert out["event"].iloc[2] == "exit_stop"

    def test_rules_validation(self):
        with pytest.raises(ValueError, match="entry_z"):
            SignalRules(entry_z=0.0)
        with pytest.raises(ValueError, match="exit_z"):
            SignalRules(entry_z=2.0, exit_z=2.5)
        with pytest.raises(ValueError, match="stop_z"):
            SignalRules(entry_z=2.0, stop_z=1.5)
        with pytest.raises(ValueError, match="max_holding"):
            SignalRules(max_holding=0)


class TestTimeStopBars:
    def test_multiple_of_half_life(self):
        assert time_stop_bars(10.0, k=3.0) == 30
        assert time_stop_bars(10.5, k=3.0) == 32  # ceil(31.5)

    def test_infinite_half_life_capped(self):
        assert time_stop_bars(np.inf, k=3.0, cap=252) == 252

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            time_stop_bars(10.0, k=0.0)


class TestSizing:
    def test_dollar_neutral(self):
        q_y, q_x = size_positions(50.0, 200.0, +1, beta=1.5, gross=1000.0, mode="dollar")
        assert q_y * 50.0 == pytest.approx(500.0, abs=1e-10)
        assert q_x * 200.0 == pytest.approx(-500.0, abs=1e-10)
        # net dollar exposure exactly zero
        assert q_y * 50.0 + q_x * 200.0 == pytest.approx(0.0, abs=1e-10)

    def test_beta_neutral_ratio_and_gross(self):
        q_y, q_x = size_positions(50.0, 200.0, -1, beta=1.5, gross=1000.0, mode="beta")
        assert q_x == pytest.approx(-1.5 * q_y, abs=1e-12)
        gross = abs(q_y) * 50.0 + abs(q_x) * 200.0
        assert gross == pytest.approx(1000.0, abs=1e-9)
        assert q_y < 0 < q_x  # short spread: short y, long x

    def test_direction_zero(self):
        assert size_positions(50.0, 200.0, 0, 1.0) == (0.0, 0.0)

    def test_validations(self):
        with pytest.raises(ValueError, match="direction"):
            size_positions(50.0, 200.0, 2, 1.0)
        with pytest.raises(ValueError, match="prices"):
            size_positions(-50.0, 200.0, 1, 1.0)
        with pytest.raises(ValueError, match="gross"):
            size_positions(50.0, 200.0, 1, 1.0, gross=0.0)
        with pytest.raises(ValueError, match="beta"):
            size_positions(50.0, 200.0, 1, -1.0, mode="beta")
        with pytest.raises(ValueError, match="mode"):
            size_positions(50.0, 200.0, 1, 1.0, mode="vol")
