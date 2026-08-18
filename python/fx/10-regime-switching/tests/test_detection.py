"""Detection tests: causality (mutation), labeling, hysteresis, edge cases."""

import numpy as np
import pandas as pd
import pytest

from fx_regime import (
    DetectionConfig,
    apply_hysteresis,
    build_features,
    generate_roro_panel,
    label_states,
    run_detection,
)

FEATS = ["avg_vol", "carry_ret", "haven_rs", "usd_corr", "em_g10", "fwd_ts"]


def test_filtered_only_past_mutation():
    """CRITICAL: the full detection output at t must not see the future."""
    panel = generate_roro_panel(700, n_states=2, seed=9)
    feats = build_features(panel.returns, panel.deposit_rates)
    cfg = DetectionConfig(n_states=2, min_train=180, refit_every=60)
    det1 = run_detection(feats, cfg, seed=0)

    cut = 450  # row index into feats, after min_train
    feats2 = feats.copy()
    feats2.iloc[cut:] = -feats2.iloc[cut:] * 3.0 + 1.0
    det2 = run_detection(feats2, cfg, seed=0)

    cut_date = feats.index[cut]
    pd.testing.assert_frame_equal(
        det1.probs.loc[det1.probs.index < cut_date],
        det2.probs.loc[det2.probs.index < cut_date],
    )
    pd.testing.assert_series_equal(
        det1.regimes.loc[det1.regimes.index < cut_date],
        det2.regimes.loc[det2.regimes.index < cut_date],
    )
    # sanity: the future DID change
    assert not np.allclose(
        det1.probs.loc[det1.probs.index >= cut_date].to_numpy(),
        det2.probs.loc[det2.probs.index >= cut_date].to_numpy(),
    )


def test_labeling_high_vol_high_corr_is_risk_off():
    means = np.array(
        [
            [-0.5, 0.4, -0.3, -0.4, 0.2, 0.1],  # low vol  -> risk_on
            [1.8, -1.2, 1.5, 1.6, -1.0, -0.2],  # high vol, haven bid -> risk_off
        ]
    )
    labels = label_states(means, FEATS, 2)
    assert labels[0] == "risk_on"
    assert labels[1] == "risk_off"


def test_labeling_three_state_squeeze_split():
    means = np.array(
        [
            [-0.5, 0.4, -0.3, -0.4, 0.2, 0.1],   # risk_on
            [1.8, -1.2, 1.5, 1.6, -1.0, -0.2],   # risk_off (haven_rs high)
            [1.5, -1.0, -1.2, 1.4, -0.8, -0.1],  # squeeze (haven_rs low)
        ]
    )
    labels = label_states(means, FEATS, 3)
    assert labels[0] == "risk_on"
    assert labels[1] == "risk_off"
    assert labels[2] == "usd_squeeze"


def test_labeling_invalid_inputs_raise():
    with pytest.raises(ValueError):
        label_states(np.zeros((2, 6)), FEATS, 4)
    with pytest.raises(ValueError):
        label_states(np.zeros((3, 6)), FEATS, 2)
    with pytest.raises(ValueError):
        label_states(np.zeros((2, 2)), ["a", "b"], 2)


def _prob_frame(p_off):
    idx = pd.bdate_range("2020-01-01", periods=len(p_off))
    p_off = np.asarray(p_off)
    return pd.DataFrame(
        {"risk_on": 1.0 - p_off, "risk_off": p_off}, index=idx
    )


def test_hysteresis_cuts_turnover_vs_argmax():
    """Oscillating probabilities: argmax flips constantly, hysteresis not."""
    p_off = np.tile([0.45, 0.55], 30)  # argmax flips every day
    probs = _prob_frame(p_off)
    raw = probs.idxmax(axis=1)
    committed = apply_hysteresis(
        probs, enter_threshold=0.7, exit_threshold=0.3, min_duration=1
    )
    n_raw = int((raw != raw.shift()).sum()) - 1
    n_committed = int((committed != committed.shift()).sum()) - 1
    assert n_raw >= 50
    assert n_committed == 0


def test_min_duration_removes_flicker():
    """A one-day probability spike must not flip the committed regime."""
    p_off = np.zeros(20)
    p_off[10] = 0.95  # single-day spike
    probs = _prob_frame(p_off)
    with_confirm = apply_hysteresis(probs, min_duration=3)
    assert (with_confirm == "risk_on").all()
    without = apply_hysteresis(probs, min_duration=1)
    assert without.iloc[10] == "risk_off"


def test_hysteresis_switches_on_persistent_signal():
    p_off = np.concatenate([np.zeros(10), np.full(10, 0.9)])
    probs = _prob_frame(p_off)
    out = apply_hysteresis(probs, min_duration=2)
    assert out.iloc[9] == "risk_on"
    assert out.iloc[11] == "risk_off"  # confirmed on 2nd qualifying day
    assert out.iloc[10] == "risk_on"   # still pending on the 1st


def test_threshold_boundary_exact():
    """Probability exactly at enter_threshold qualifies (>=)."""
    at = apply_hysteresis(
        _prob_frame([0.0, 0.70]), enter_threshold=0.70, min_duration=1
    )
    assert at.iloc[1] == "risk_off"
    below = apply_hysteresis(
        _prob_frame([0.0, 0.699999]), enter_threshold=0.70, min_duration=1
    )
    assert below.iloc[1] == "risk_on"


def test_exit_threshold_path():
    """Incumbent collapse (< exit) lets a sub-enter challenger through."""
    # p_off = 0.65: argmax is risk_off, below enter=0.7, but risk_on
    # prob 0.35 is NOT below exit=0.3 -> no switch
    no_switch = apply_hysteresis(
        _prob_frame([0.0, 0.65, 0.65]), enter_threshold=0.7,
        exit_threshold=0.3, min_duration=1,
    )
    assert (no_switch == "risk_on").all()
    # p_off = 0.75 > enter -> switch; equivalently incumbent 0.25 < exit
    switch = apply_hysteresis(
        _prob_frame([0.0, 0.72, 0.72]), enter_threshold=0.7,
        exit_threshold=0.3, min_duration=1,
    )
    assert switch.iloc[1] == "risk_off"


def test_detection_recovers_regimes(panel2, feats2, det2):
    """Committed regimes track the truth on a well-specified panel."""
    true = pd.Series(
        [panel2.state_names[s] for s in panel2.states],
        index=panel2.returns.index,
    ).reindex(det2.regimes.index)
    acc = (det2.regimes == true).mean()
    assert acc > 0.75
    # both labels actually used
    assert set(det2.regimes.unique()) == {"risk_on", "risk_off"}


def test_hysteresis_fewer_switches_than_raw(det2):
    n_raw = int((det2.raw_regimes != det2.raw_regimes.shift()).sum())
    n_committed = int((det2.regimes != det2.regimes.shift()).sum())
    assert n_committed <= n_raw


def test_probs_rows_sum_to_one(det2):
    assert np.allclose(det2.probs.sum(axis=1), 1.0, atol=1e-8)
    assert not det2.probs.isna().any().any()


def test_all_risk_on_sample_handled():
    """One state never visited: the k=2 model is unidentified, but the
    detector must run to completion with valid, finite probabilities.

    (The detector WILL split noise into two pseudo-states here — that is
    the documented failure mode; the economic guard is the null-panel
    strategy test in test_risk.py, which shows the split carries no
    P&L edge.)"""
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    panel = generate_roro_panel(500, n_states=2, seed=3, transition=P)
    assert (panel.states == 0).all()
    feats = build_features(panel.returns, panel.deposit_rates)
    det = run_detection(
        feats, DetectionConfig(n_states=2, min_train=150, refit_every=100),
        seed=0,
    )
    assert np.allclose(det.probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(det.probs.to_numpy()).all()
    assert not det.regimes.isna().any()
    # hysteresis still throttles switching relative to raw argmax
    n_raw = int((det.raw_regimes != det.raw_regimes.shift()).sum())
    n_committed = int((det.regimes != det.regimes.shift()).sum())
    assert n_committed <= n_raw


def test_config_validation():
    with pytest.raises(ValueError):
        DetectionConfig(n_states=5)
    with pytest.raises(ValueError):
        DetectionConfig(enter_threshold=0.3, exit_threshold=0.7)
    with pytest.raises(ValueError):
        DetectionConfig(min_duration=0)
    with pytest.raises(ValueError):
        apply_hysteresis(_prob_frame([0.5]), enter_threshold=0.2,
                         exit_threshold=0.4)


def test_short_series_raises(feats2):
    with pytest.raises(ValueError):
        run_detection(feats2.iloc[:100],
                      DetectionConfig(n_states=2, min_train=252))
