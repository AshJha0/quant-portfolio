"""Risk analysis: per-regime stats, transition attribution, detection-lag cost.

The centrepiece is the ORACLE COMPARISON — an honesty metric.  The
oracle strategy knows the true hidden state (zero detection lag, zero
classification error) but pays the same costs and the same one-day
execution delay.  The gap between oracle and filtered performance is
therefore exactly what imperfect, lagged detection costs; the gap
between filtered and static carry is what detection is worth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def perf_stats(net: pd.Series) -> dict[str, float]:
    """Annualised performance statistics of a daily return series.

    Returns
    -------
    dict with ``ann_return`` (mean * 252), ``ann_vol`` (std * sqrt 252,
    ddof=1), ``sharpe``, ``max_drawdown`` (most negative peak-to-trough
    of the cumulative sum, <= 0), ``calmar``, ``hit_rate``, ``n_days``.
    """
    r = net.dropna()
    n = len(r)
    if n == 0:
        raise ValueError("empty return series")
    mu = float(r.mean()) * TRADING_DAYS
    sd = float(r.std(ddof=1)) * np.sqrt(TRADING_DAYS) if n > 1 else 0.0
    equity = r.cumsum()
    peak = equity.cummax()
    dd = float((equity - peak).min())
    return {
        "ann_return": mu,
        "ann_vol": sd,
        "sharpe": mu / sd if sd > 0 else 0.0,
        "max_drawdown": dd,
        "calmar": mu / abs(dd) if dd < 0 else np.inf,
        "hit_rate": float((r > 0).mean()),
        "n_days": n,
    }


def per_regime_stats(net: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Performance statistics partitioned by regime label.

    The partition is exact: the ``total_pnl`` column sums to the total
    P&L of ``net`` over the common index (tested to 1e-12).

    Parameters
    ----------
    net : daily net returns.
    regimes : labels aligned (or alignable) to ``net.index``.

    Returns
    -------
    DataFrame indexed by regime label with perf columns + ``total_pnl``.
    """
    reg = regimes.reindex(net.index)
    if reg.isna().any():
        raise ValueError("regimes must cover the net return index")
    rows = {}
    for label, grp in net.groupby(reg):
        s = perf_stats(grp)
        s["total_pnl"] = float(grp.sum())
        rows[label] = s
    return pd.DataFrame(rows).T.sort_index()


def regime_spells(regimes: pd.Series) -> pd.DataFrame:
    """Contiguous same-label spells.

    Returns
    -------
    DataFrame with columns ``label``, ``start`` (position), ``end``
    (exclusive position), ``length``.
    """
    labels = regimes.to_numpy()
    if len(labels) == 0:
        raise ValueError("empty regime series")
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return pd.DataFrame(
        {
            "label": labels[starts],
            "start": starts,
            "end": ends,
            "length": ends - starts,
        }
    )


def transition_attribution(
    net: pd.Series, regimes: pd.Series, window: int = 5
) -> pd.DataFrame:
    """Split each regime spell's P&L into entry window vs remainder.

    For every spell, the first ``window`` days are the "transition"
    bucket, the rest is "steady".  Identity (tested):
    transition + steady per label sums to that label's total P&L, and
    the grand total equals ``net.sum()``.

    Returns
    -------
    DataFrame indexed by label with columns ``transition_pnl``,
    ``steady_pnl``, ``total_pnl``, ``n_spells``.
    """
    reg = regimes.reindex(net.index)
    if reg.isna().any():
        raise ValueError("regimes must cover the net return index")
    spells = regime_spells(reg)
    vals = net.to_numpy()
    out: dict[str, dict[str, float]] = {}
    for _, sp in spells.iterrows():
        lab = sp["label"]
        s, e = int(sp["start"]), int(sp["end"])
        cut = min(s + window, e)
        d = out.setdefault(
            lab,
            {"transition_pnl": 0.0, "steady_pnl": 0.0, "total_pnl": 0.0,
             "n_spells": 0.0},
        )
        d["transition_pnl"] += float(vals[s:cut].sum())
        d["steady_pnl"] += float(vals[cut:e].sum())
        d["total_pnl"] += float(vals[s:e].sum())
        d["n_spells"] += 1.0
    return pd.DataFrame(out).T.sort_index()


def detection_lag(
    true_regimes: pd.Series,
    detected_regimes: pd.Series,
    target_labels: tuple[str, ...] = ("risk_off", "usd_squeeze"),
) -> pd.DataFrame:
    """Days between each true flip into a risk state and its detection.

    For every true-regime spell whose label is in ``target_labels``, the
    lag is the number of days from the spell's start until the detector
    first reports ANY label in ``target_labels`` (searching to the end
    of the spell plus a grace of one spell length).  Detection is
    defined DEFENSIVELY: flagging usd_squeeze during a true risk_off
    spell counts as detected, because both books cut carry — what is
    being priced is the days the filter left the book exposed, not the
    label spelling.  Only FRESH flips are counted: spells entered from a
    state outside ``target_labels`` (a risk_off spell straight after a
    usd_squeeze spell is a label change, not a new flip).  Undetected
    spells are censored at the search horizon and flagged.

    Returns
    -------
    DataFrame with one row per true flip: ``label``, ``start_date``,
    ``lag_days`` (>= 0), ``detected`` (bool), ``spell_length``.
    """
    common = detected_regimes.index.intersection(true_regimes.index)
    if len(common) == 0:
        raise ValueError("no overlapping dates")
    t_reg = true_regimes.reindex(common)
    d_reg = detected_regimes.reindex(common)
    spells = regime_spells(t_reg)
    d_np = d_reg.to_numpy()
    rows = []
    for _, sp in spells.iterrows():
        lab = sp["label"]
        if lab not in target_labels:
            continue
        s, e, length = int(sp["start"]), int(sp["end"]), int(sp["length"])
        if s == 0:
            continue  # not a flip, sample starts here
        if t_reg.iloc[s - 1] in target_labels:
            continue  # label change within a risk episode, not a fresh flip
        horizon = min(e + length, len(d_np))
        hit = np.flatnonzero(np.isin(d_np[s:horizon], target_labels))
        detected = hit.size > 0
        lag = int(hit[0]) if detected else horizon - s
        rows.append(
            {
                "label": lab,
                "start_date": common[s],
                "lag_days": lag,
                "detected": detected,
                "spell_length": length,
            }
        )
    return pd.DataFrame(rows)


def detection_lag_report(
    true_regimes: pd.Series,
    detected_regimes: pd.Series,
    oracle_net: pd.Series,
    filtered_net: pd.Series,
    target_labels: tuple[str, ...] = ("risk_off", "usd_squeeze"),
) -> dict[str, float | pd.DataFrame]:
    """Detection-lag statistics and the cost of the lag per flip.

    The lag cost of a flip is the oracle-minus-filtered net P&L summed
    over the lag window (flip, flip + lag] — the money lost while the
    filter had not yet flagged the true state.  The window is shifted
    one day past the flip because BOTH books carry the old position over
    the flip day itself (one-day execution delay), and a label detected
    at the close of day flip+lag only repositions the filtered book for
    day flip+lag+1 (position drift outside the window is not charged to
    the lag).

    Returns
    -------
    dict with ``flips`` (per-flip DataFrame incl. ``lag_cost``),
    ``mean_lag_days``, ``median_lag_days``, ``mean_cost_per_flip``,
    ``total_lag_cost``, ``n_flips``, ``detection_rate``.
    """
    flips = detection_lag(true_regimes, detected_regimes, target_labels)
    diff = (oracle_net - filtered_net).dropna()
    costs = []
    for _, fl in flips.iterrows():
        start = fl["start_date"]
        lag = int(fl["lag_days"])
        pos = diff.index.searchsorted(start)
        costs.append(float(diff.iloc[pos + 1 : pos + 1 + lag].sum()))
    flips = flips.assign(lag_cost=costs)
    if len(flips) == 0:
        return {
            "flips": flips,
            "mean_lag_days": np.nan,
            "median_lag_days": np.nan,
            "mean_cost_per_flip": np.nan,
            "total_lag_cost": 0.0,
            "n_flips": 0,
            "detection_rate": np.nan,
        }
    return {
        "flips": flips,
        "mean_lag_days": float(flips["lag_days"].mean()),
        "median_lag_days": float(flips["lag_days"].median()),
        "mean_cost_per_flip": float(flips["lag_cost"].mean()),
        "total_lag_cost": float(flips["lag_cost"].sum()),
        "n_flips": int(len(flips)),
        "detection_rate": float(flips["detected"].mean()),
    }


def oracle_gap_decomposition(
    oracle_net: pd.Series,
    strategy_net: pd.Series,
    true_regimes: pd.Series,
    risk_labels: tuple[str, ...] = ("risk_off", "usd_squeeze"),
) -> dict[str, float]:
    """Split the oracle-vs-strategy P&L gap by true regime.

    ``gap_risk_days`` is the part of the oracle's edge earned while the
    market was truly in a risk state (detection lag / wrong defensive
    book); ``gap_calm_days`` is the part earned in true risk_on
    (false alarms: the strategy was defensive while carry paid).
    Identity (tested): gap_risk_days + gap_calm_days = gap_total.

    Returns
    -------
    dict with ``gap_total``, ``gap_risk_days``, ``gap_calm_days``
    (cumulative return units).
    """
    common = oracle_net.index.intersection(strategy_net.index)
    if len(common) == 0:
        raise ValueError("no overlapping dates")
    diff = oracle_net.reindex(common) - strategy_net.reindex(common)
    reg = true_regimes.reindex(common)
    if reg.isna().any():
        raise ValueError("true_regimes must cover the overlapping dates")
    risk_mask = reg.isin(risk_labels)
    return {
        "gap_total": float(diff.sum()),
        "gap_risk_days": float(diff[risk_mask].sum()),
        "gap_calm_days": float(diff[~risk_mask].sum()),
    }


def carry_drawdown_decomposition(
    carry_net: pd.Series,
    true_regimes: pd.Series,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    risk_labels: tuple[str, ...] = ("risk_off", "usd_squeeze"),
) -> pd.DataFrame:
    """How much of the carry book's risk-state losses land in the first K days.

    For each K in ``horizons``: the static carry book's P&L summed over
    the first K days of every risk-state spell, as a share of its total
    P&L in risk states.  A share near 1 at small K is the signature of
    the carry crash: the damage is front-loaded, which is precisely why
    detection lag is expensive.

    Returns
    -------
    DataFrame indexed by K with ``pnl_first_k``, ``share_of_risk_pnl``.
    """
    reg = true_regimes.reindex(carry_net.index)
    if reg.isna().any():
        raise ValueError("regimes must cover the carry return index")
    spells = regime_spells(reg)
    vals = carry_net.to_numpy()
    risk_spells = spells[spells["label"].isin(risk_labels)]
    total_risk = float(
        sum(vals[int(s.start) : int(s.end)].sum() for s in risk_spells.itertuples())
    )
    rows = {}
    for K in horizons:
        first_k = float(
            sum(
                vals[int(s.start) : min(int(s.start) + K, int(s.end))].sum()
                for s in risk_spells.itertuples()
            )
        )
        share = first_k / total_risk if total_risk != 0 else np.nan
        rows[K] = {"pnl_first_k": first_k, "share_of_risk_pnl": share}
    return pd.DataFrame(rows).T.rename_axis("K")


def comparison_table(nets: dict[str, pd.Series]) -> pd.DataFrame:
    """Side-by-side performance table (e.g. oracle / filtered / static).

    All series are aligned to their common index before stats are
    computed so the comparison is apples-to-apples.
    """
    if not nets:
        raise ValueError("no return series supplied")
    common = None
    for s in nets.values():
        common = s.index if common is None else common.intersection(s.index)
    return pd.DataFrame(
        {name: perf_stats(s.reindex(common)) for name, s in nets.items()}
    ).T
