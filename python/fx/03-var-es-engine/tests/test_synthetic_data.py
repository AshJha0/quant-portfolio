"""Synthetic generators: blocks, regimes, GARCH, pegs, calibration replication."""

import numpy as np
import pandas as pd
import pytest

from fx_var import Book, Forward, Option, Spot, ewma_volatility
from fx_var.data.synthetic import (
    ANNUAL_VOLS,
    EM,
    G10,
    PEGGED,
    default_correlation,
    demo_book,
    demo_em_book,
    demo_market,
    simulate_fx_returns,
    simulate_history,
)


def test_universe_has_vols():
    for c in G10 + EM + PEGGED:
        assert c in ANNUAL_VOLS


def test_correlation_matrices_valid():
    for regime in ("calm", "stress"):
        c = default_correlation(regime=regime)
        m = c.to_numpy()
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(m).min() > -1e-10  # PSD
    with pytest.raises(ValueError):
        default_correlation(regime="panic")


def test_stress_regime_raises_correlations_and_flips_jpy():
    calm = default_correlation(regime="calm")
    stress = default_correlation(regime="stress")
    assert stress.loc["EUR", "GBP"] > calm.loc["EUR", "GBP"]
    assert stress.loc["MXN", "BRL"] > calm.loc["MXN", "BRL"]
    # safe-haven JPY: correlation to carry currencies goes negative in stress
    assert calm.loc["JPY", "AUD"] > 0
    assert stress.loc["JPY", "AUD"] < 0


def test_seed_reproducibility():
    a = simulate_fx_returns(["EUR", "JPY"], 200, seed=5)
    b = simulate_fx_returns(["EUR", "JPY"], 200, seed=5)
    c = simulate_fx_returns(["EUR", "JPY"], 200, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_calibration_replication_vols_and_correlation():
    """Loose-tolerance replication: the simulator's sample moments recover
    the calibration inputs (vols within 10%, block corr within 0.10)."""
    ccys = ["EUR", "GBP", "JPY", "MXN", "BRL"]
    r = simulate_fx_returns(ccys, 6000, seed=42)
    ann = r.std(ddof=1) * np.sqrt(252)
    for c in ccys:
        assert ann[f"FX:{c}"] == pytest.approx(ANNUAL_VOLS[c], rel=0.10)
    corr = r.corr()
    assert corr.loc["FX:EUR", "FX:GBP"] == pytest.approx(0.55, abs=0.10)
    assert corr.loc["FX:MXN", "FX:BRL"] == pytest.approx(0.45, abs=0.10)


def test_garch_produces_volatility_clustering():
    """ACF of squared returns: strong under GARCH, ~0 under constant vol."""
    g = simulate_fx_returns(["EUR"], 4000, seed=11, garch=True)["FX:EUR"].to_numpy()
    i = simulate_fx_returns(["EUR"], 4000, seed=11, garch=False)["FX:EUR"].to_numpy()

    def acf1(x):
        x2 = x**2 - (x**2).mean()
        return float(np.corrcoef(x2[:-1], x2[1:])[0, 1])

    assert acf1(g) > 0.10
    assert abs(acf1(i)) < 0.05


def test_garch_ewma_tracks_true_sigma():
    """Calibration cross-check: EWMA vol correlates strongly with the true
    conditional sigma of the GARCH simulator."""
    r, state = simulate_fx_returns(["EUR"], 3000, seed=13, garch=True,
                                   return_state=True)
    sig_hat, _ = ewma_volatility(r, 0.94)
    corr = np.corrcoef(sig_hat["FX:EUR"].to_numpy()[50:],
                       state["FX:EUR"].to_numpy()[50:])[0, 1]
    assert corr > 0.7


def test_regime_switching_vols():
    r, state = simulate_fx_returns(["EUR", "AUD"], 4000, seed=17,
                                   regime_switching=True, return_state=True)
    regime = state["regime"].to_numpy()
    assert 0 < regime.sum() < len(regime)  # both regimes visited
    x = r["FX:EUR"].to_numpy()
    assert x[regime == 1].std() > 1.5 * x[regime == 0].std()


def test_peg_simulator_near_zero_vol_with_rare_jumps():
    r = simulate_fx_returns(["HKD"], 4000, seed=23, peg_jump_prob=0.005)
    x = r["FX:HKD"].to_numpy()
    n_jumps = int((np.abs(x) > 0.04).sum())
    assert 5 <= n_jumps <= 50  # ~20 expected
    quiet = x[np.abs(x) < 0.04]
    assert quiet.std() * np.sqrt(252) < 0.01  # near-zero vol otherwise


def test_peg_simulator_no_jumps_by_default():
    x = simulate_fx_returns(["HKD"], 3000, seed=2)["FX:HKD"].to_numpy()
    assert np.abs(x).max() < 0.01


def test_unknown_ccy_raises():
    with pytest.raises(ValueError, match="XAU"):
        simulate_fx_returns(["XAU"], 100, seed=0)


# ------------------------------------------------------------ demo objects
def test_demo_market_snapshot():
    m = demo_market()
    assert m.spot("USD") == 1.0
    assert m.cross("USDJPY") == pytest.approx(149.0, rel=1e-9)
    assert m.rate("TRY") > m.rate("JPY")  # EM carry
    assert m.vol("USDTRY") > m.vol("EURUSD")


def test_demo_book_composition():
    b = demo_book()
    kinds = [type(p).__name__ for p in b.positions]
    assert "Spot" in kinds and "Forward" in kinds and "Option" in kinds
    ccys = b.currencies()
    assert "MXN" in ccys  # EM position
    assert "HKD" in ccys  # peg position
    m = demo_market()
    f = b.factors(m)
    assert "VOL:EURUSD" in f and "IR:USD" in f


def test_demo_em_book_is_long_em_fx():
    b = demo_em_book()
    m = demo_market()
    w = b.linear_exposures(m)
    for f in ("FX:MXN", "FX:BRL", "FX:TRY", "FX:ZAR"):
        assert w[f] > 0  # long the local currency = short USD legs


def test_simulate_history_covers_book_factors():
    m = demo_market()
    b = demo_book()
    h = simulate_history(b, m, 300, seed=1)
    assert list(h.columns) == b.factors(m)
    assert not h.isna().any().any()
    assert (h.filter(like="IR:").std() < 0.001).all()  # bp-scale rate moves


def test_live_loader_guard_blocks_network(monkeypatch):
    """The Frankfurter loader must refuse to touch the network unless the
    caller opts in explicitly - no test ever opens the guard."""
    monkeypatch.delenv("FX_VAR_ALLOW_NETWORK", raising=False)
    from fx_var.data import live

    with pytest.raises(RuntimeError, match="Network access is disabled"):
        live.load_frankfurter(["EUR"])
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        live.frankfurter_factor_returns(["EUR", "JPY"])
