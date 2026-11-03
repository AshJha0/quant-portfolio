"""W22: ``eq_var.engine.monte_carlo_var`` on every installed engine.

The engines use different generators, so they agree statistically, not by
value. Each is checked against the closed-form linear-book VaR within four
standard errors of the sample quantile, SE = sqrt(a (1 - a) / n) / f(q)
(asymptotic order-statistic variance; Stuart and Ord, Kendall's Advanced
Theory of Statistics, vol. 1, 10.10).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from eq_var.engine import monte_carlo_var
from qp_core.engines import engine_available
from scipy.stats import norm, t

ENGINES = [e for e in ("python", "cpp", "rust") if engine_available(e)]

W = np.array([1_000_000.0, -400_000.0, 250_000.0])
VOL = np.array([0.01, 0.015, 0.02])
CORR = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
COV = CORR * np.outer(VOL, VOL)
SIGMA_P = math.sqrt(W @ COV @ W)
N = 200_000


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_normal_matches_closed_form(engine, alpha):
    q = norm.ppf(alpha)
    exact = -q * SIGMA_P
    se = math.sqrt(alpha * (1 - alpha) / N) / norm.pdf(q) * SIGMA_P
    for seed in (1, 2):
        got = monte_carlo_var(W, COV, alpha, N, "normal", seed=seed, engine=engine)
        assert abs(got - exact) <= 4 * se, (engine, seed, got, exact, se)


@pytest.mark.parametrize("engine", ENGINES)
def test_t_matches_closed_form(engine):
    df, alpha = 5.0, 0.01
    scale = SIGMA_P * math.sqrt((df - 2) / df)
    q = t.ppf(alpha, df)
    exact = -q * scale
    se = math.sqrt(alpha * (1 - alpha) / N) / (t.pdf(q, df) / scale)
    got = monte_carlo_var(W, COV, alpha, N, "t", df=df, seed=3, engine=engine)
    assert abs(got - exact) <= 4 * se, (engine, got, exact, se)


def test_python_engine_is_reproducible_and_validates():
    a = monte_carlo_var(W, COV, 0.01, 10_000, seed=7)
    assert a == monte_carlo_var(W, COV, 0.01, 10_000, seed=7)
    with pytest.raises(ValueError, match="exposures has 2"):
        monte_carlo_var(W[:2], COV, 0.01, 1_000)
    with pytest.raises(ValueError):
        monte_carlo_var(W, COV, 0.7, 1_000)
