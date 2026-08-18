"""Point-in-time regime detection: expanding refit, FILTERED probabilities.

Design rules (each enforced by tests):

1. **Filtered only.**  The probability used at date t is
   P(s_t | x_{1..t}) from a model fitted on data up to the last refit
   date <= t.  Nothing downstream of this module may see the future
   (mutation-tested: perturbing x_{t+1:} leaves output at <= t
   unchanged).
2. **Economic labeling.**  HMM state indices are arbitrary; each refit
   maps them to economic labels from the state means in feature space:
   risk_on = lowest ``avg_vol`` mean; with k=3 the remaining two split by
   ``haven_rs`` mean — risk_off = havens rallying (high haven_rs),
   usd_squeeze = havens falling too (low haven_rs).  Risk-off is thus,
   by construction, the high-vol / high-correlation / haven-bid state.
3. **Hysteresis + confirmation.**  Regime switches require the new
   regime's filtered probability to clear ``enter_threshold`` (or the
   incumbent's to fall below ``exit_threshold``) for
   ``min_duration`` consecutive days.  This kills flicker and cuts
   turnover at the cost of detection lag — the lag is measured and
   priced in :mod:`fx_regime.risk`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hmm import HMMModel, filtered_probabilities, fit_hmm

LABELS_2 = ("risk_on", "risk_off")
LABELS_3 = ("risk_on", "risk_off", "usd_squeeze")


@dataclass(frozen=True)
class DetectionConfig:
    """Controls for the expanding-window detector.

    Attributes
    ----------
    n_states : {2, 3}
    min_train : int
        First fit uses this many observations; no output before it.
    refit_every : int
        Refit cadence in days (warm-started from the previous fit).
    enter_threshold : float
        Filtered probability a challenger regime must reach (>=) to
        start the confirmation clock.
    exit_threshold : float
        If the incumbent's probability drops below (<) this, the argmax
        challenger starts the clock even without clearing enter.
    min_duration : int
        Consecutive qualifying days required before a switch commits.
    max_iter, n_init : int
        EM controls for the underlying HMM fits.
    detect_columns : tuple of str
        Feature columns the HMM is fitted on.  Defaults to the
        mean-reverting, regime-informative subset; ``fwd_ts`` is
        EXCLUDED by default because its expanding z-score is
        near-unit-root and lets the HMM carve the sample into spurious
        'early/late' epochs instead of RORO states (see
        docs/VALIDATION.md failure modes).
    """

    n_states: int = 3
    min_train: int = 252
    refit_every: int = 21
    enter_threshold: float = 0.70
    exit_threshold: float = 0.30
    min_duration: int = 2
    max_iter: int = 50
    n_init: int = 2
    detect_columns: tuple[str, ...] = (
        "avg_vol", "carry_ret", "haven_rs", "usd_corr", "em_g10", "usd_str"
    )

    def __post_init__(self) -> None:
        if self.n_states not in (2, 3):
            raise ValueError("n_states must be 2 or 3")
        if not 0.0 < self.exit_threshold < self.enter_threshold <= 1.0:
            raise ValueError("need 0 < exit_threshold < enter_threshold <= 1")
        if self.min_duration < 1:
            raise ValueError("min_duration must be >= 1")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")


def label_states(
    means: np.ndarray, feature_columns: list[str], n_states: int
) -> dict[int, str]:
    """Map HMM state indices to economic labels from feature-space means.

    Parameters
    ----------
    means : (k, p) state means in feature space.
    feature_columns : names aligned to the p columns; must contain
        ``avg_vol`` and, for k=3, ``haven_rs``.
    n_states : {2, 3}

    Returns
    -------
    dict state_index -> label.
    """
    if n_states not in (2, 3):
        raise ValueError("n_states must be 2 or 3")
    if means.shape[0] != n_states:
        raise ValueError("means row count must equal n_states")
    cols = list(feature_columns)
    if "avg_vol" not in cols:
        raise ValueError("feature_columns must contain avg_vol")
    i_vol = cols.index("avg_vol")
    order_by_vol = np.argsort(means[:, i_vol])
    risk_on = int(order_by_vol[0])
    labels = {risk_on: "risk_on"}
    rest = [s for s in range(n_states) if s != risk_on]
    if n_states == 2:
        labels[rest[0]] = "risk_off"
    else:
        if "haven_rs" not in cols:
            raise ValueError("feature_columns must contain haven_rs for k=3")
        i_h = cols.index("haven_rs")
        a, b = rest
        # risk_off = havens rallying; usd_squeeze = havens falling too
        if means[a, i_h] >= means[b, i_h]:
            labels[a], labels[b] = "risk_off", "usd_squeeze"
        else:
            labels[a], labels[b] = "usd_squeeze", "risk_off"
    return labels


def apply_hysteresis(
    probs: pd.DataFrame,
    enter_threshold: float = 0.70,
    exit_threshold: float = 0.30,
    min_duration: int = 2,
    initial: str | None = None,
) -> pd.Series:
    """Turn labeled filtered probabilities into a committed regime path.

    Causal by construction: the regime at t depends only on rows <= t.

    Switch rule: a challenger label c != incumbent starts/extends the
    confirmation clock at t if ``probs[c][t] >= enter_threshold`` or
    ``probs[incumbent][t] < exit_threshold`` (with c the argmax).  After
    ``min_duration`` consecutive qualifying days with the same c, the
    switch commits (dated at the confirming day).  Any non-qualifying
    day, or a change of challenger, resets the clock.

    Parameters
    ----------
    probs : DataFrame (T x labels), rows ~ sum to 1.
    initial : label to start from (default: argmax of the first row).

    Returns
    -------
    Series of labels aligned to ``probs.index``.
    """
    if not 0.0 < exit_threshold < enter_threshold <= 1.0:
        raise ValueError("need 0 < exit_threshold < enter_threshold <= 1")
    if min_duration < 1:
        raise ValueError("min_duration must be >= 1")
    labels = list(probs.columns)
    arr = probs.to_numpy()
    current = initial if initial is not None else labels[int(np.argmax(arr[0]))]
    if current not in labels:
        raise ValueError(f"initial label {current!r} not in columns")
    out: list[str] = []
    pending: str | None = None
    streak = 0
    cur_idx = labels.index(current)
    for t in range(len(arr)):
        row = arr[t]
        cand_idx = int(np.argmax(row))
        cand = labels[cand_idx]
        qualifies = cand != current and (
            row[cand_idx] >= enter_threshold or row[cur_idx] < exit_threshold
        )
        if qualifies:
            if pending == cand:
                streak += 1
            else:
                pending, streak = cand, 1
            if streak >= min_duration:
                current = cand
                cur_idx = labels.index(current)
                pending, streak = None, 0
        else:
            pending, streak = None, 0
        out.append(current)
    return pd.Series(out, index=probs.index, name="regime")


@dataclass
class DetectionResult:
    """Output of :func:`run_detection`.

    Attributes
    ----------
    probs : DataFrame (dates x labels)
        Filtered probabilities under economic labels.
    regimes : Series of committed labels (post hysteresis/confirmation).
    raw_regimes : Series of argmax labels (no hysteresis), for
        turnover comparisons.
    refit_dates : list of dates at which the HMM was refitted.
    models : dict date -> fitted HMMModel (last fit per refit date).
    """

    probs: pd.DataFrame
    regimes: pd.Series
    raw_regimes: pd.Series
    refit_dates: list
    models: dict


def run_detection(
    features: pd.DataFrame,
    config: DetectionConfig | None = None,
    seed: int = 0,
) -> DetectionResult:
    """Expanding-window regime detection with filtered probabilities.

    At each refit date t0 (every ``refit_every`` days from
    ``min_train``), an HMM is fitted to ``features[:t0]`` (warm-started
    from the previous fit).  For each t in the following block the
    filtered probability P(s_t | x_{1..t}) is computed with that frozen
    model via a single forward pass over ``features[:t+1]`` — causal for
    every t because the forward recursion at t never touches rows > t.

    Parameters
    ----------
    features : DataFrame (T x p)
        PIT-standardised feature block; must contain ``avg_vol`` (and
        ``haven_rs`` for the 3-state config).
    config : DetectionConfig
    seed : RNG seed for the first (cold) fit.

    Returns
    -------
    DetectionResult

    Raises
    ------
    ValueError
        If there are fewer than ``min_train + 1`` rows.
    """
    cfg = config or DetectionConfig()
    T = len(features)
    if T < cfg.min_train + 1:
        raise ValueError(
            f"need more than min_train={cfg.min_train} rows, got {T}"
        )
    use_cols = [c for c in cfg.detect_columns if c in features.columns]
    if "avg_vol" not in use_cols:
        raise ValueError("detection features must include avg_vol")
    features = features[use_cols]
    X = features.to_numpy(dtype=float)
    cols = list(features.columns)
    label_names = LABELS_2 if cfg.n_states == 2 else LABELS_3

    all_probs = np.full((T, cfg.n_states), np.nan)
    refit_dates = []
    models: dict = {}
    model: HMMModel | None = None

    t0 = cfg.min_train
    while t0 < T:
        train = X[:t0]
        model = fit_hmm(
            train,
            cfg.n_states,
            seed=seed,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter,
            init_model=model,
        )
        labels = label_states(model.means, cols, cfg.n_states)
        refit_dates.append(features.index[t0])
        models[features.index[t0]] = model
        block_end = min(t0 + cfg.refit_every, T)
        # one forward pass over data through the end of the block:
        # filtered row t uses only x_{1..t} -> causal for every t
        filt = filtered_probabilities(model, X[:block_end])
        for t in range(t0, block_end):
            for s in range(cfg.n_states):
                j = label_names.index(labels[s])
                all_probs[t, j] = filt[t, s]
        t0 = block_end

    probs = pd.DataFrame(
        all_probs[cfg.min_train:],
        index=features.index[cfg.min_train:],
        columns=list(label_names),
    )
    regimes = apply_hysteresis(
        probs,
        enter_threshold=cfg.enter_threshold,
        exit_threshold=cfg.exit_threshold,
        min_duration=cfg.min_duration,
    )
    raw = probs.idxmax(axis=1).rename("regime")
    return DetectionResult(
        probs=probs,
        regimes=regimes,
        raw_regimes=raw,
        refit_dates=refit_dates,
        models=models,
    )
