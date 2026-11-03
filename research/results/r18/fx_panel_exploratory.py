"""EXPLORATORY real-data FX carry / value / momentum panel, 2000-2015 only.

Plain idea: run the W23 panel code (``research/panel/``) on real free data
for the training years, with the hold-out lock closed, so the draft
combination registration can replace its assumed IC and N_eff with
in-sample numbers. Nothing on or after 2016-01-01 is read: every frame goes
through ``HoldoutLock.training`` and then ``run_panel``'s own guard.

Data (all from the local ``qp_data`` store, filled by ``fetch_store.py``):

* spot: ECB reference rates, inverted to USD per unit of each currency;
* short rates: OECD MEI 3-month interbank rates on FRED, used only from
  their release date (bounded at the 1st of month m+2);
* CPI: OECD MEI CPI all items (EUR uses Germany, stated proxy), same lag;
* universe and live dates: ``research/universe/fx.csv``.

Labelled exploratory: the data were chosen after the code existed and the
results are not a test. The store holds today's vintage, so revisions to
pre-2016 rates and CPI are not removed (``revision_risk``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qp_data

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for sub in ("panel", "lib", "combination"):
    if str(REPO / "research" / sub) not in sys.path:
        sys.path.insert(0, str(REPO / "research" / sub))
sys.path.insert(0, str(HERE))

import combine  # type: ignore[import-not-found]  # noqa: E402
import signals as sg  # type: ignore[import-not-found]  # noqa: E402
import study  # type: ignore[import-not-found]  # noqa: E402
from fetch_store import OECD  # type: ignore[import-not-found]  # noqa: E402
from holdout_lock import HoldoutLock  # type: ignore[import-not-found]  # noqa: E402

END = pd.Timestamp("2015-12-31")


def _get(key: str) -> pd.DataFrame | None:
    try:
        return qp_data.get(key)
    except qp_data.NoVintage:
        return None


def known_at(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Latest value whose release date is on or before each date."""
    f = frame.dropna(subset=["value"]).sort_values("release_date")
    s = pd.Series(f["value"].to_numpy(float), index=pd.DatetimeIndex(f["release_date"]))
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    px, used = {}, {}
    for c in OECD:
        if c == "EUR":
            continue
        f = _get(f"ecb:EXR:{c}")
        if f is not None:
            px[c] = f[c]
            used[f"ecb:EXR:{c}"] = f.attrs["sha256"]
    daily = pd.DataFrame(px).sort_index()
    daily = daily[daily.index <= END]
    spot = sg.month_ends(sg.xr.invert_ecb(daily, "USD"))
    spot.index = spot.index.to_period("M").to_timestamp("M")
    idx = pd.DatetimeIndex(spot.index)
    rates, cpi = {}, {}
    for c, (iso2, iso3) in OECD.items():
        r = _get(f"fred:IR3TIB01{iso2}M156N")
        if r is not None:
            rates[c] = known_at(r, idx) / 100.0
            used[f"fred:IR3TIB01{iso2}M156N"] = r.attrs["sha256"]
        key = "fred:DEUCPIALLMINMEI" if c == "EUR" else f"fred:{iso3}CPIALLMINMEI"
        p = _get(key)
        if p is not None:
            cpi[c] = known_at(p, idx)
            used[key] = p.attrs["sha256"]
    uni = pd.read_csv(REPO / "research" / "universe" / "fx.csv")
    live = pd.DataFrame(False, index=idx, columns=spot.columns)
    for _, row in uni.iterrows():
        c = row["ccy"]
        if c not in live.columns:
            continue
        first = pd.Timestamp("1999-01-01") if row["first"] == "auto" else pd.Timestamp(row["first"])
        last = pd.Timestamp(row["last"]) if isinstance(row["last"], str) and row["last"] else END
        live[c] = (idx >= first) & (idx <= last) & spot[c].notna().to_numpy()
    return spot, pd.DataFrame(rates, index=idx), pd.DataFrame(cpi, index=idx), live, used


def implied_ic(fm_slopes: pd.Series, rx: pd.DataFrame, live: pd.DataFrame) -> float:
    """Mean slope over the mean cross-sectional sd of next-month returns: the
    slope on a z-scored signal equals IC times that sd."""
    sd = rx.shift(-1).where(live).std(axis=1, ddof=1).reindex(fm_slopes.index)
    return float(fm_slopes.mean() / sd.mean())


def main() -> dict:
    spot, rates, cpi, live, used = load_inputs()
    rx = sg.monthly_excess_returns(spot, rates, "USD")
    sigs = {
        "carry": sg.carry(rates, "USD"),
        "value": sg.value(spot, cpi, "USD"),
        "momentum": sg.momentum(rx),
    }
    lock = HoldoutLock()
    tr = HoldoutLock.training
    lv = tr(live.reindex(columns=rx.columns).fillna(False))
    # The draft registration's score: equal-weight z-scores, clip 3, at least 2 present.
    sigs["combined"] = combine.combine_signals(
        {k: tr(v).reindex(columns=rx.columns) for k, v in sigs.items()}, lv
    ).reindex(index=lv.index, columns=rx.columns)
    res = study.run_panel(tr(rx), {k: tr(v) for k, v in sigs.items()}, lv, lock=lock)
    import panel  # type: ignore[import-not-found]

    for name, sig in sigs.items():
        cols = [c for c in sig.columns if c in rx.columns]
        fm = panel.fama_macbeth(tr(sig)[cols], tr(rx).shift(-1)[cols], live=lv[cols], nw_lag=3)
        res["signals"][name]["implied_ic"] = implied_ic(fm["slopes"], tr(rx)[cols], lv[cols])
        res["signals"][name]["slope_sd"] = float(fm["slopes"].std(ddof=1))
    res.update(
        {
            "label": "exploratory",
            "window": [str(rx.dropna(how="all").index.min().date()), str(END.date())],
            "holdout_lock": "closed; no row on or after 2016-01-01 was loaded",
            "currencies_with_returns": int(rx.notna().any().sum()),
            "missing_rates": sorted(set(spot.columns) - set(rates.columns)),
            "eur_cpi_proxy": "Germany (OECD MEI); no euro-area MEI CPI on FRED",
            "inputs_sha256": used,
            "caveats": [
                "today's vintage of OECD MEI rates and CPI: revisions not removed",
                "3-month interbank rates, not forward-implied rates: carry ignores CIP deviations",
                "pre-2000 data cannot be stored (qp_core.holidays covers 2000-2099), so the window starts 2000",
            ],
        }
    )
    return res


def _clean(o: object) -> object:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


if __name__ == "__main__":
    out = _clean(main())
    (HERE / "fx_panel_exploratory.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for k, v in out["signals"].items():  # type: ignore[index]
        print(
            k,
            {
                x: v[x]
                for x in ("months", "hml_sharpe_annual", "hml_t_nw", "fm_t", "median_live", "implied_ic")
            },
        )
