"""Strategy tests: hysteresis, allocation mapping, vol targeting, turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_regime.strategy import (
    build_weights,
    hysteresis_regime,
    naive_threshold_regime,
    regime_target_weight,
    turnover,
    vol_target_scale,
)

TRADING_DAYS = 252


class TestHysteresis:
    def test_constructed_path(self):
        p = np.array([0.1, 0.75, 0.5, 0.65, 0.2, 0.9, 0.31, 0.29])
        out = hysteresis_regime(p, enter=0.7, exit_=0.3)
        # enters at 0.75; stays through 0.5 / 0.65 (inside band); exits at
        # 0.2; re-enters at 0.9; stays at 0.31; exits at 0.29
        np.testing.assert_array_equal(
            out, [False, True, True, True, False, True, True, False]
        )

    def test_boundary_values_do_not_flip(self):
        """At exactly enter/exit the state must NOT change (strict inequalities)."""
        p = np.array([0.70, 0.70, 0.71, 0.30, 0.30, 0.29])
        out = hysteresis_regime(p, enter=0.7, exit_=0.3)
        np.testing.assert_array_equal(out, [False, False, True, True, True, False])

    def test_reduces_turnover_vs_naive_threshold(self):
        """Noisy probability path oscillating around 0.5: hysteresis must
        trade strictly less than the naive single-threshold rule."""
        rng = np.random.default_rng(0)
        p = np.clip(0.5 + 0.15 * rng.standard_normal(500), 0.0, 1.0)
        naive = naive_threshold_regime(p).astype(float)
        hyst = hysteresis_regime(p).astype(float)
        assert turnover(hyst) < 0.2 * turnover(naive)

    def test_reduces_turnover_on_regimelike_path(self):
        """Path that genuinely switches twice plus noise: hysteresis still
        cuts turnover while capturing both switches."""
        rng = np.random.default_rng(1)
        base = np.concatenate([np.full(100, 0.1), np.full(100, 0.9), np.full(100, 0.1)])
        p = np.clip(base + 0.25 * rng.standard_normal(300), 0.0, 1.0)
        naive = naive_threshold_regime(p).astype(float)
        hyst = hysteresis_regime(p).astype(float)
        assert turnover(hyst) < turnover(naive)
        assert hyst[150] == 1.0 and hyst[50] == 0.0 and hyst[280] == 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match="exit < enter"):
            hysteresis_regime(np.array([0.5]), enter=0.3, exit_=0.7)
        with pytest.raises(ValueError, match="0, 1"):
            hysteresis_regime(np.array([1.5]))


class TestAllocation:
    def test_regime_target_weight_mapping(self):
        regimes = np.array(["bull", "transition", "bear"])
        w = regime_target_weight(regimes)
        np.testing.assert_allclose(w, [1.0, 0.5, 0.0])

    def test_bear_flag_overrides(self):
        regimes = np.array(["bull", "bull", "bear"])
        flag = np.array([False, True, False])
        w = regime_target_weight(regimes, bear_flag=flag)
        # flagged bull -> bear weight; unconfirmed bear label -> transition weight
        np.testing.assert_allclose(w, [1.0, 0.0, 0.5])

    def test_bear_flag_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            regime_target_weight(np.array(["bull"]), bear_flag=np.array([True, False]))


class TestVolTargeting:
    def test_hits_ex_ante_target(self):
        """Constant-vol returns: scaled position realizes ~ the target vol."""
        rng = np.random.default_rng(2)
        ann_vol = 0.20
        r = pd.Series(
            rng.standard_normal(2000) * ann_vol / np.sqrt(TRADING_DAYS),
            index=pd.bdate_range("2015-01-01", periods=2000),
        )
        scale = vol_target_scale(r, target_vol=0.10, window=63, max_leverage=3.0)
        scaled = (scale.shift(1) * r).dropna()
        realized = scaled.std(ddof=1) * np.sqrt(TRADING_DAYS)
        assert realized == pytest.approx(0.10, rel=0.15)

    def test_clipped_at_max_leverage(self):
        r = pd.Series(
            1e-6 * np.random.default_rng(3).standard_normal(300),
            index=pd.bdate_range("2020-01-01", periods=300),
        )
        scale = vol_target_scale(r, target_vol=0.10, window=21, max_leverage=1.5)
        assert scale.max() <= 1.5 + 1e-12

    def test_validation(self):
        r = pd.Series([0.01, -0.01])
        with pytest.raises(ValueError, match="target_vol"):
            vol_target_scale(r, target_vol=0.0)
        with pytest.raises(ValueError, match="max_leverage"):
            vol_target_scale(r, max_leverage=-1.0)


class TestBuildWeights:
    def _detection(self, n=100, seed=4):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-01", periods=n)
        p_bear = np.clip(rng.beta(2, 5, n), 0, 1)
        regime = np.where(p_bear > 0.5, "bear", "bull")
        return pd.DataFrame(
            {"p_bull": 1 - p_bear, "p_transition": 0.0, "p_bear": p_bear,
             "regime": regime},
            index=idx,
        )

    def test_output_aligned_and_bounded(self):
        det = self._detection()
        r = pd.Series(0.001, index=det.index)
        w = build_weights(det, r, target_vol=None)
        assert w.index.equals(det.index)
        assert ((w >= 0) & (w <= 1)).all()

    def test_missing_columns_raise(self):
        det = self._detection().drop(columns=["p_bear"])
        with pytest.raises(ValueError, match="columns"):
            build_weights(det, pd.Series(dtype=float))

    def test_turnover_hand_checked(self):
        assert turnover(np.array([1.0, 1.0, 0.0, 0.5])) == pytest.approx(1.0 + 1.0 + 0.5)
        assert turnover(np.array([])) == 0.0
