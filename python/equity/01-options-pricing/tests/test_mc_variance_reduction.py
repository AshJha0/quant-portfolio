"""Antithetic plus control variate must beat the control variate alone.

The control-variate coefficient is fitted on the antithetic pair averages.
Fitting it on raw draws and pairing afterwards gave SE 0.0246-0.0248 for
this contract at 100k paths (seeds 1-5), worse than the control variate
alone (0.0177).
"""

import pytest

from eq_options import bs_price, mc_price

SEEDS = [1, 2, 3, 4, 5, 6, 7]
OLD_COMBINED_SE = 0.0246


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_antithetic_plus_cv_beats_cv_alone(seed: int, otype: str) -> None:
    args = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, otype)
    both = mc_price(*args, n_paths=100_000, antithetic=True, control_variate=True, seed=seed)
    cv = mc_price(*args, n_paths=100_000, antithetic=False, control_variate=True, seed=seed)
    assert both.std_error <= cv.std_error
    assert abs(both.value - bs_price(*args)) < 4.0 * both.std_error


@pytest.mark.parametrize("seed", SEEDS)
def test_combined_se_well_below_old_value(seed: int) -> None:
    both = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call", n_paths=100_000,
                    antithetic=True, control_variate=True, seed=seed)
    # Measured 0.0086-0.0088 over seeds 1-5. The bound 0.6 x old is about
    # 19 standard errors of the SE estimate above that.
    assert both.std_error < 0.6 * OLD_COMBINED_SE
