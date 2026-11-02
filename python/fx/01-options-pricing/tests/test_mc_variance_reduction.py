"""Antithetic plus control variate must beat the control variate alone.

Beta is fitted on the antithetic pair averages. Fitting it on raw draws
and pairing afterwards gave SE 1.39e-4 for this contract at 100k paths
(seeds 1-5), worse than the control variate alone (9.9e-5).
"""

import pytest
from fx_options import gk_price, mc_price

SEEDS = [1, 2, 3, 4, 5, 6, 7]
OLD_COMBINED_SE = 1.39e-4
ARGS = (1.10, 1.10, 1.0, 0.04, 0.02, 0.10)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_antithetic_plus_cv_beats_cv_alone(seed: int, otype: str) -> None:
    both = mc_price(*ARGS, otype, n_paths=100_000, rng=seed,
                    antithetic=True, control_variate=True)
    cv = mc_price(*ARGS, otype, n_paths=100_000, rng=seed,
                  antithetic=False, control_variate=True)
    assert both.std_error <= cv.std_error
    assert abs(both.price - gk_price(*ARGS, otype)) < 4.0 * both.std_error


@pytest.mark.parametrize("seed", SEEDS)
def test_combined_se_well_below_old_value(seed: int) -> None:
    both = mc_price(*ARGS, "call", n_paths=100_000, rng=seed)
    # Measured 4.71e-5 to 4.82e-5 over seeds 1-5.
    assert both.std_error < 0.6 * OLD_COMBINED_SE
