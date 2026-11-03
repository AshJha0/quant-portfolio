"""Names and call forms deprecated in v2.0.0 or earlier are gone in v3.0.0.

docs/VERSIONING.md: a deprecation that warned in a released version is
removed at the next major release. Each case below warned in v2.0.0 and
now fails loudly instead of being accepted.
"""

from __future__ import annotations

import eq_var
import numpy as np
import pytest
from eq_var import (
    acerbi_szekely_z2,
    basel_traffic_light,
    ewma_volatility,
    kupiec_pof,
)
from eq_var.data.synthetic import demo_covariance, simulate_garch_returns


@pytest.mark.parametrize("name", ["safe_cholesky", "cornish_fisher_domain_ok"])
def test_removed_functions(name: str) -> None:
    assert not hasattr(eq_var, name)
    assert name not in eq_var.__all__


def test_kupiec_positional_form_removed() -> None:
    with pytest.raises(TypeError):
        kupiec_pof(250, 5, 0.01)  # type: ignore[misc]


def test_basel_positional_form_removed() -> None:
    with pytest.raises(TypeError):
        basel_traffic_light(7)  # type: ignore[misc]


def test_acerbi_positional_alpha_removed() -> None:
    with pytest.raises(TypeError):
        acerbi_szekely_z2(np.zeros(10), 5.0, 6.0, 0.025)  # type: ignore[misc]


def test_ewma_init_removed() -> None:
    with pytest.raises(TypeError):
        ewma_volatility(np.zeros(10), 0.94, "sample")  # type: ignore[call-arg]


def test_garch_omega_frac_removed() -> None:
    with pytest.raises(TypeError):
        simulate_garch_returns(10, demo_covariance(), omega_frac=0.1)  # type: ignore[call-arg]
