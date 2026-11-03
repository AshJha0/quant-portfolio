"""W14b: ``smile_dynamics=`` on the FX option risk calls (flat vol per call).

With one vol per option the smile has no slope, so every dynamic gives the
same Greeks; the default (sticky strike) keeps the numbers from before the
keyword. Smile-consistent Greeks: ``qp_core.risk.smile_delta``.
"""

from __future__ import annotations

import pytest
from fx_options.greeks import analytic_greeks, finite_difference_greeks

ARGS = (1.10, 1.12, 0.75, 0.04, 0.02, 0.09, "call")


@pytest.mark.parametrize("d", ["sticky_strike", "sticky_moneyness", "sticky_delta"])
def test_default_preserves_numbers_and_dynamics_agree_on_flat_vol(d):
    assert analytic_greeks(*ARGS, smile_dynamics=d) == analytic_greeks(*ARGS)
    assert finite_difference_greeks(*ARGS, smile_dynamics=d) == finite_difference_greeks(*ARGS)


def test_default_delta_is_the_garman_kohlhagen_delta():
    # Pinned from the code before the keyword existed.
    assert analytic_greeks(*ARGS).delta_spot == pytest.approx(0.4926517443929328, rel=1e-14)


def test_unknown_dynamic_is_refused():
    with pytest.raises(ValueError, match="smile_dynamics"):
        analytic_greeks(*ARGS, smile_dynamics="sticky_vol")
    with pytest.raises(ValueError, match="smile_dynamics"):
        finite_difference_greeks(*ARGS, smile_dynamics="sticky_vol")
