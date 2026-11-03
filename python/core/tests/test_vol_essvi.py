"""eSSVI by slice and per-slice SVI against fx_surface VALIDATION F10.

F10: power-law SSVI misses five-point USDJPY smiles by about one vol point.
The pillar data are the stylised fx_surface presets that define F10 (frozen
in tests/data/fx_pillar_smiles.json). No free FX option quotes exist; ECB
publishes spot reference rates only, so "ECB-derived" smiles are not
available and these presets stand in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from qp_core.vol.essvi import (
    ESSVISlice,
    SVISlice,
    calendar_arbitrage_free,
    essvi_total_variance,
    fit_essvi_slices,
    fit_svi_slices,
)

DATA = json.loads((Path(__file__).parent / "data" / "fx_pillar_smiles.json").read_text())


def _inputs(pair: str):
    s = DATA["markets"][pair]["slices"]
    return ([x["strikes"] for x in s], [x["vols"] for x in s], [x["forward"] for x in s], [x["T"] for x in s])


@pytest.mark.parametrize("pair", ["USDJPY", "EURUSD"])
def test_svi_per_slice_closes_f10_within_01_vol(pair):
    fit = fit_svi_slices(*_inputs(pair))
    assert max(fit.max_vol_error) < 0.001  # 0.1 vol point
    assert fit.butterfly_free and fit.calendar_free


def test_essvi_by_slice_improves_f10_but_shape_limits_usdjpy():
    """eSSVI (three numbers per slice) cuts USDJPY's worst miss from about
    1 vol to about 0.115 vol; the 1y slice cannot go lower even without the
    arbitrage constraints (an unconstrained least squares gives the same
    0.115), so the limit is the slice shape, not the constraints."""
    fit = fit_essvi_slices(*_inputs("USDJPY"))
    assert max(fit.max_vol_error) < 0.0012
    assert max(fit.max_vol_error) > 0.001  # documented: does not meet 0.1 vol
    assert fit.butterfly_free and fit.calendar_free
    eur = fit_essvi_slices(*_inputs("EURUSD"))
    assert max(eur.max_vol_error) < 0.001
    assert eur.butterfly_free and eur.calendar_free


def test_essvi_slice_is_ssvi_and_checks():
    s = ESSVISlice(0.01, -0.3, 0.2)
    phi = 0.2 / 0.01
    k = np.linspace(-1, 1, 21)
    ssvi = 0.5 * 0.01 * (1 - 0.3 * phi * k + np.sqrt((phi * k - 0.3) ** 2 + 1 - 0.09))
    assert np.allclose(s.total_variance(k), ssvi, rtol=0, atol=1e-16)
    assert not ESSVISlice(0.0001, -0.3, 0.2).butterfly_free()
    assert calendar_arbitrage_free([s, ESSVISlice(0.02, -0.3, 0.25)])
    assert not calendar_arbitrage_free([s, ESSVISlice(0.011, -0.3, 0.1)])
    for bad in [(0.0, 0.0, 0.1), (0.01, 1.0, 0.1), (0.01, 0.0, 0.0)]:
        with pytest.raises(ValueError):
            ESSVISlice(*bad)
    assert float(essvi_total_variance(0.0, 0.03, 0.5, 0.4)) == pytest.approx(0.03, abs=1e-16)


def test_svi_density_factor_flags_arbitrage():
    good = SVISlice(0.004, 0.08, -0.4, 0.01, 0.15)
    assert np.all(good.density_factor(np.linspace(-1, 1, 101)) > 0)
    # Axel Vogt's example (Gatheral and Jacquier 2014, s.2.2) has butterfly arbitrage
    vogt = SVISlice(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    assert np.any(vogt.density_factor(np.linspace(-1.5, 1.5, 301)) < 0)


def test_fit_input_checks():
    with pytest.raises(ValueError, match="bracket"):
        fit_essvi_slices([[1.2, 1.3, 1.4]], [[0.1, 0.1, 0.1]], [1.1], [1.0])
    with pytest.raises(ValueError, match="three"):
        fit_essvi_slices([[1.0, 1.2]], [[0.1, 0.1]], [1.1], [1.0])
    with pytest.raises(ValueError, match="increasing"):
        fit_essvi_slices([[1.0, 1.1, 1.2]] * 2, [[0.1] * 3] * 2, [1.1, 1.1], [1.0, 0.5])
    with pytest.raises(ValueError, match="five"):
        fit_svi_slices([[1.0, 1.1, 1.2]], [[0.1] * 3], [1.1], [1.0])


def test_exotics_golden_is_current():
    gen = Path(__file__).parent / "gen_exotics_golden.py"
    src = Path(__file__).resolve().parents[1] / "src"
    r = subprocess.run([sys.executable, str(gen), "--check"], env={"PYTHONPATH": str(src), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, "rust/core/tests/data/exotics_golden.txt is stale: rerun gen_exotics_golden.py" + r.stderr
