"""Shared fit-result container for the GARCH-family models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VolatilityFitResult:
    """Result of a maximum-likelihood fit of a conditional-variance model.

    Attributes
    ----------
    model : str
        Model name ("GARCH", "EGARCH", "GJR-GARCH").
    dist : str
        Innovation distribution ("normal" or "t").
    params : dict[str, float]
        Estimated parameters in natural units (decimal-return scale).
    std_errors : dict[str, float]
        Asymptotic standard errors from the inverse numerical Hessian of the
        negative log-likelihood. NaN where the Hessian is not positive
        definite (surfaced, never hidden).
    loglik : float
        Maximised log-likelihood.
    n_obs : int
        Number of observations used.
    sigma2 : numpy.ndarray
        Fitted conditional variance path (daily variance).
    returns : numpy.ndarray
        The return series the model was fitted on.
    converged : bool
        Optimiser convergence flag.
    message : str
        Optimiser diagnostic message.
    init_var : float
        Pre-sample variance used to start the recursion.
    """

    model: str
    dist: str
    params: dict[str, float]
    std_errors: dict[str, float]
    loglik: float
    n_obs: int
    sigma2: np.ndarray
    returns: np.ndarray
    converged: bool
    message: str
    init_var: float
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        """Akaike information criterion: 2k - 2 lnL (lower is better)."""
        return 2.0 * self.n_params - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        """Bayesian information criterion: k ln n - 2 lnL (lower is better)."""
        return self.n_params * np.log(self.n_obs) - 2.0 * self.loglik

    @property
    def std_residuals(self) -> np.ndarray:
        """Standardised residuals z_t = r_t / sigma_t (unit variance if the
        model is correctly specified)."""
        return self.returns / np.sqrt(self.sigma2)

    def summary(self) -> str:
        """Plain-text parameter table with standard errors and t-stats."""
        lines = [
            f"{self.model}(1,1) [{self.dist}]  n={self.n_obs}  "
            f"logL={self.loglik:.2f}  AIC={self.aic:.2f}  BIC={self.bic:.2f}",
            f"{'param':>10} {'estimate':>14} {'std err':>12} {'t-stat':>10}",
        ]
        for k, v in self.params.items():
            se = self.std_errors.get(k, np.nan)
            t = v / se if se and np.isfinite(se) and se > 0 else np.nan
            lines.append(f"{k:>10} {v:>14.6g} {se:>12.4g} {t:>10.2f}")
        for k, v in self.extra.items():
            lines.append(f"{k:>10} {v:>14.6g}")
        return "\n".join(lines)
