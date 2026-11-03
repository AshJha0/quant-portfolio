"""Names and call forms deprecated in v2.0.0 or earlier are gone in v3.0.0.

docs/VERSIONING.md: a deprecation that warned in a released version is
removed at the next major release. Each case below warned in v2.0.0 and
now fails loudly instead of being accepted.
"""

from __future__ import annotations

import fx_var
import numpy as np
import pytest
from fx_var import basel_traffic_light, empirical_es, empirical_var, kupiec_pof, normal_es, robust_cholesky

PNL = np.random.default_rng(1).normal(0.0, 1.0, 500)


@pytest.mark.parametrize("name", ["es_backtest_acerbi_szekely", "cornish_fisher_domain_ok"])
def test_removed_functions(name: str) -> None:
    assert not hasattr(fx_var, name)
    assert name not in fx_var.__all__


@pytest.mark.parametrize("call", [
    lambda: empirical_var(PNL, confidence=0.99),
    lambda: empirical_es(PNL, confidence=0.99),
    lambda: normal_es(2.0, confidence=0.975),
])
def test_confidence_keyword_removed(call) -> None:
    with pytest.raises(TypeError):
        call()


def test_positional_backtest_forms_removed() -> None:
    with pytest.raises(TypeError):
        kupiec_pof(5, 250, 0.99)
    with pytest.raises(TypeError):
        basel_traffic_light(5, 250)


def test_robust_cholesky_max_tries_removed() -> None:
    with pytest.raises(TypeError):
        robust_cholesky(np.eye(2), max_tries=3)  # type: ignore[call-arg]
