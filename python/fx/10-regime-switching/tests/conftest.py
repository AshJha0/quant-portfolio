"""Shared fixtures: session-scoped synthetic panels and fitted detectors.

Everything is seeded and offline; session scope keeps the expensive
expanding-refit detections to one run each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx_regime import (
    DetectionConfig,
    StrategyConfig,
    build_features,
    generate_roro_panel,
    oracle_regimes,
    run_backtest,
    run_detection,
    static_carry_regimes,
)


@pytest.fixture(scope="session")
def panel2():
    """2-state RORO panel, 1100 business days."""
    return generate_roro_panel(1100, n_states=2, seed=42)


@pytest.fixture(scope="session")
def panel3():
    """3-state RORO panel (with USD-squeeze), 1100 business days."""
    return generate_roro_panel(1100, n_states=3, seed=7)


@pytest.fixture(scope="session")
def feats2(panel2):
    return build_features(panel2.returns, panel2.deposit_rates)


@pytest.fixture(scope="session")
def feats3(panel3):
    return build_features(panel3.returns, panel3.deposit_rates)


@pytest.fixture(scope="session")
def det2(feats2):
    """Expanding-refit detection on the 2-state panel."""
    return run_detection(
        feats2,
        DetectionConfig(n_states=2, min_train=252, refit_every=42),
        seed=0,
    )


@pytest.fixture(scope="session")
def true_labels2(panel2):
    return pd.Series(
        [panel2.state_names[s] for s in panel2.states],
        index=panel2.returns.index,
        name="regime",
    )


@pytest.fixture(scope="session")
def backtests2(panel2, det2):
    """Filtered / oracle / static backtests on the 2-state panel."""
    cfg = StrategyConfig()
    filtered = run_backtest(
        panel2.returns, panel2.deposit_rates, det2.regimes, cfg
    )
    oracle = run_backtest(
        panel2.returns,
        panel2.deposit_rates,
        oracle_regimes(
            panel2.returns.index, panel2.states, panel2.state_names
        ).loc[det2.regimes.index[0]:],
        cfg,
    )
    static = run_backtest(
        panel2.returns,
        panel2.deposit_rates,
        static_carry_regimes(det2.regimes.index),
        cfg,
    )
    return {"filtered": filtered, "oracle": oracle, "static": static}
