"""Regime inference pipeline: fitting, ONLINE filtering, labelling, stability.

FILTERED vs SMOOTHED — the central honesty point of this project
=================================================================

* **Filtered** probabilities ``P(s_t | x_{1..t})`` use only data available
  at time ``t``.  They are causal and therefore TRADEABLE.
* **Smoothed** probabilities ``P(s_t | x_{1..T})`` condition on the FULL
  sample, including the future.  They look far cleaner in plots — regime
  edges snap into place — precisely because they peek ahead.  Backtesting a
  strategy on smoothed probabilities is lookahead bias and will overstate
  performance, typically dramatically around regime turns.

Everything in this module that feeds trading uses filtered probabilities.
The test suite enforces this with a mutation test: perturbing FUTURE
observations must leave the filtered probability at ``t`` bit-identical,
while the smoothed probability at ``t`` must change (sanity contrast).

Also provided:

* economic labelling of states (``bull`` / ``bear`` / ``transition``) by
  sorting on state emission means — the highest-vol state is always the
  bear, the lowest-vol (and highest-trend) state the bull;
* stability diagnostics: regime flip-flop rate and a minimum-duration
  filter that removes one/two-day flickers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hmm import HMMFit, fit_hmm
from .pca import fit_pca, project

__all__ = [
    "RegimeLabels",
    "label_states",
    "labels_from_vol_means",
    "filtered_probabilities",
    "smoothed_probabilities",
    "flip_flop_rate",
    "min_duration_filter",
    "expanding_fit_detect",
]

BULL, TRANSITION, BEAR = "bull", "transition", "bear"


def _fb(fit: HMMFit, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward on a fitted model (training-window diagnostics)."""
    from .hmm import forward_backward

    return forward_backward(x, fit.startprob, fit.transmat, fit.means, fit.covariances)


@dataclass(frozen=True)
class RegimeLabels:
    """Mapping between HMM state indices and economic labels.

    Attributes
    ----------
    state_to_label : dict[int, str]
        e.g. ``{0: 'bull', 1: 'transition', 2: 'bear'}``.
    bear_state, bull_state : int
        Convenience indices.
    """

    state_to_label: dict[int, str]
    bear_state: int
    bull_state: int

    def label_path(self, states: np.ndarray) -> np.ndarray:
        """Map an integer state path to label strings."""
        return np.array([self.state_to_label[int(s)] for s in states])


def label_states(fit: HMMFit, vol_feature_index: int = 0) -> RegimeLabels:
    """Label HMM states economically by sorting on state emission means.

    The feature at ``vol_feature_index`` must be a volatility-like feature
    (higher = more stressed); by convention the FIRST column of the feature
    table built by :func:`eq_regime.features.build_features` is a realized
    vol.  States are sorted by their emission mean on that feature:

    * highest vol mean  -> ``bear``
    * lowest vol mean   -> ``bull``
    * anything between  -> ``transition``

    This mapping is deterministic given the fit, which makes labels
    comparable across walk-forward refits even though raw state indices
    permute arbitrarily (label switching).

    Parameters
    ----------
    fit : HMMFit
    vol_feature_index : int
        Column of the observation vector holding the vol-like feature.

    Returns
    -------
    RegimeLabels
    """
    k = fit.n_states
    if not 0 <= vol_feature_index < fit.means.shape[1]:
        raise ValueError(
            f"vol_feature_index {vol_feature_index} out of range for "
            f"{fit.means.shape[1]} features"
        )
    vol_means = fit.means[:, vol_feature_index]
    return labels_from_vol_means(vol_means)


def labels_from_vol_means(vol_means: np.ndarray) -> RegimeLabels:
    """Build a :class:`RegimeLabels` from per-state vol-feature means.

    Lowest vol mean -> ``bull``, highest -> ``bear`` (when K > 1),
    everything between -> ``transition``.
    """
    vol_means = np.asarray(vol_means, dtype=float)
    k = len(vol_means)
    if k < 1:
        raise ValueError("need at least one state")
    order = np.argsort(vol_means)  # ascending vol
    mapping: dict[int, str] = {}
    for rank, state in enumerate(order):
        if rank == 0:
            mapping[int(state)] = BULL
        elif rank == k - 1 and k > 1:
            mapping[int(state)] = BEAR
        else:
            mapping[int(state)] = TRANSITION
    bear_state = int(order[-1]) if k > 1 else int(order[0])
    return RegimeLabels(
        state_to_label=mapping, bear_state=bear_state, bull_state=int(order[0])
    )


def filtered_probabilities(fit: HMMFit, x: np.ndarray) -> np.ndarray:
    """ONLINE filtered probabilities ``P(s_t | x_{1..t})`` — tradeable.

    Row ``t`` depends only on observations up to and including ``t``
    (mutation-test-enforced).  Use THIS for signals.
    """
    probs, _ = fit.filter(np.asarray(x, dtype=float))
    return probs


def smoothed_probabilities(fit: HMMFit, x: np.ndarray) -> np.ndarray:
    """Smoothed posteriors ``P(s_t | x_{1..T})`` — NOT tradeable.

    Row ``t`` depends on the whole sample including the future.  Use only
    for historical diagnostics and reporting, never for signals.
    """
    return fit.smooth(np.asarray(x, dtype=float))


def flip_flop_rate(states: np.ndarray) -> float:
    """Fraction of days on which the regime changes (0 = never, 1 = daily).

    A healthy daily regime model flips a few times a year; a rate above a
    few percent signals an unstable fit that needs a duration filter or
    hysteresis.
    """
    states = np.asarray(states)
    if len(states) < 2:
        return 0.0
    return float(np.mean(states[1:] != states[:-1]))


def min_duration_filter(states: np.ndarray, min_duration: int) -> np.ndarray:
    """Remove regime runs shorter than ``min_duration`` days.

    Scans runs left to right; any run shorter than ``min_duration`` is
    absorbed into the PREVIOUS surviving regime (causal choice — the
    alternative, merging into the following regime, would need future
    knowledge of when the run ends beyond the filter horizon).  The first
    run is never removed.

    Parameters
    ----------
    states : (T,) integer regime path.
    min_duration : int
        Minimum run length in days, >= 1 (1 = no-op).

    Returns
    -------
    (T,) filtered path.
    """
    if min_duration < 1:
        raise ValueError(f"min_duration must be >= 1, got {min_duration}")
    states = np.asarray(states).copy()
    if min_duration == 1 or len(states) == 0:
        return states
    out = states.copy()
    current = out[0]
    run_start = 0
    for t in range(1, len(out) + 1):
        if t == len(out) or out[t] != current:
            run_len = t - run_start
            if run_start > 0 and run_len < min_duration:
                out[run_start:t] = out[run_start - 1]
                # merged into previous: continue the previous run
                current = out[run_start - 1]
                # recompute run_start backwards to the start of merged run
                rs = run_start
                while rs > 0 and out[rs - 1] == current:
                    rs -= 1
                run_start = rs
                if t < len(out):
                    if out[t] == current:
                        continue
                    current = out[t]
                    run_start = t
            else:
                if t < len(out):
                    current = out[t]
                    run_start = t
    return out


def expanding_fit_detect(
    features: pd.DataFrame,
    n_states: int = 3,
    min_train: int = 252,
    refit_every: int = 63,
    seed: int = 0,
    vol_feature_index: int = 0,
    n_init: int = 2,
    max_iter: int = 100,
    n_pca: int | None = None,
) -> pd.DataFrame:
    """Walk-forward regime detection on an expanding window.

    At each refit date ``t0`` (every ``refit_every`` days, first at
    ``min_train``), an HMM is fitted on features ``[0, t0)``.  For each
    subsequent day ``t`` until the next refit, the FILTERED probability
    ``P(s_t | x_{1..t})`` is computed with that (frozen) model over data up
    to ``t`` only.  No future observation ever enters the value at ``t``
    — neither through the fit window nor through smoothing.

    When ``n_pca`` is set, a PCA (from scratch, :mod:`eq_regime.pca`) is
    fitted on the SAME training window and the HMM operates on the leading
    ``n_pca`` principal-component scores — still fully causal, since both
    the PCA and the HMM only ever see data up to the refit date, and the
    projection of row ``t`` uses only row ``t``.  State labelling then uses
    posterior-weighted means of the ORIGINAL vol feature (highest = bear),
    which is invariant to the PC rotation.

    Labels are re-derived at every refit via :func:`label_states`, so the
    output columns are economic (``p_bull``, ``p_transition``, ``p_bear``)
    and immune to label switching across refits.

    Parameters
    ----------
    features : pd.DataFrame
        Point-in-time feature table (NaN-free).
    n_states : int
        Number of hidden states.
    min_train : int
        First fit uses this many observations; detection starts there.
    refit_every : int
        Refit cadence in days.
    seed : int
        Master seed for all fits.
    vol_feature_index : int
        Passed to :func:`label_states`.
    n_init, max_iter : HMM fit effort per refit.

    Returns
    -------
    pd.DataFrame
        Indexed by detection dates, columns ``p_bull``, ``p_transition``,
        ``p_bear``, ``regime`` (argmax label) — all filtered / causal.
    """
    if len(features) <= min_train:
        raise ValueError(
            f"need more than min_train={min_train} rows, got {len(features)}"
        )
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every}")
    x_all = features.to_numpy(dtype=float)
    t_len = len(x_all)
    rows: list[np.ndarray] = []
    dates: list = []
    regimes: list[str] = []
    for t0 in range(min_train, t_len, refit_every):
        if n_pca is not None:
            pca = fit_pca(x_all[:t0], n_pca)
            obs = project(x_all, pca)  # row t uses only row t (causal)
        else:
            obs = x_all
        fit = fit_hmm(
            obs[:t0], n_states, seed=seed, n_init=n_init, max_iter=max_iter
        )
        if n_pca is not None:
            # Label via posterior-weighted means of the ORIGINAL vol feature
            # on the training window (rotation-invariant, still causal).
            gamma, _, _ = _fb(fit, obs[:t0])
            nk = gamma.sum(axis=0)
            vol_means = (gamma.T @ x_all[:t0, vol_feature_index]) / np.maximum(nk, 1e-300)
            labels = labels_from_vol_means(vol_means)
        else:
            labels = label_states(fit, vol_feature_index)
        t1 = min(t0 + refit_every, t_len)
        # The forward recursion is causal: row t of a forward pass over
        # x[0:t1] is IDENTICAL to the last row of a pass over x[0:t+1].
        # One pass per block therefore yields the same filtered values as
        # re-running the filter day by day (mutation-test-enforced).
        probs_block = filtered_probabilities(fit, obs[:t1])[t0:t1]
        for t, probs_t in zip(range(t0, t1), probs_block):
            agg = {BULL: 0.0, TRANSITION: 0.0, BEAR: 0.0}
            for s in range(n_states):
                agg[labels.state_to_label[s]] += float(probs_t[s])
            rows.append(np.array([agg[BULL], agg[TRANSITION], agg[BEAR]]))
            dates.append(features.index[t])
            regimes.append(max(agg, key=agg.get))
    out = pd.DataFrame(rows, index=pd.Index(dates), columns=["p_bull", "p_transition", "p_bear"])
    out["regime"] = regimes
    return out
