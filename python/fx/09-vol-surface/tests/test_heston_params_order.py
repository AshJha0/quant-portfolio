"""HestonParams: one keyword API and field order across eq_surface and fx_surface."""

import dataclasses

import pytest
from fx_surface import HestonParams

KW = dict(v0=0.04, kappa=1.5, theta=0.045, xi=0.4, rho=-0.6)


def test_field_order_is_shared():
    assert [f.name for f in dataclasses.fields(HestonParams)] == ["v0", "kappa", "theta", "xi", "rho"]


def test_keyword_construction_has_no_warning(recwarn):
    p = HestonParams(**KW)
    assert (p.xi, p.rho) == (0.4, -0.6)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_positional_keeps_historical_order_with_warning():
    with pytest.warns(DeprecationWarning, match="positional HestonParams"):
        p = HestonParams(0.04, 1.5, 0.045, 0.4, -0.6)
    assert p == HestonParams(**KW)


def test_bad_arguments():
    with pytest.raises(TypeError, match="missing"):
        HestonParams(v0=0.04, kappa=1.5, theta=0.045, xi=0.4)
    with pytest.raises(TypeError, match="unexpected"):
        HestonParams(**KW, sigma=0.1)
    with pytest.raises(ValueError, match="rho"):
        HestonParams(**{**KW, "rho": 1.5})


def test_replace_and_frozen():
    p = HestonParams(**KW)
    assert dataclasses.replace(p, rho=0.0).rho == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.v0 = 0.05  # type: ignore[misc]
