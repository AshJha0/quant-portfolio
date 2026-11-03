"""Simulated power for the three draft registrations (W6 power gate, Round 18).

Plain idea: before a registration is signed we want to know how often its
own decision rule, run by its own code on data the size of its own sample,
would detect an effect of the size the literature reports. We plant an
effect of known size in synthetic data, run the registered rule many times,
and count rejections. With nothing planted the count is the false-positive
rate, which should sit near 5 %.

Formal statement. For a design ``D`` (sample shape, estimator, decision rule)
and a planted effect ``theta``, power is ``P_theta(reject H0 | D)``,
estimated by the share of ``reps`` independent simulations that reject; its
Monte Carlo standard error is ``sqrt(p (1 - p) / reps)``.

The three designs (one function each):

* ``combination_panel``: ``research/combination/PREREG_DRAFT.md``. A
  Fama-MacBeth slope on a cross-sectional score, NW lag 3, one-sided 5 %,
  over the 130 hold-out months 2016-01 to 2026-10. Effect = monthly
  information coefficient (IC). The cross-section is grouped in regional
  clusters with correlated signals and correlated return noise, which is
  what makes ``N_eff`` smaller than ``N``.
* ``ust_descriptive``: ``research/ust/PREREG_DRAFT.md``. One monthly series
  (slope or fly) from a sign-of-carry rule over 442 months. Descriptive
  under W6 (single series, ``monitoring``); the rate reported is how often
  a NW t-statistic would exceed 1.645, so readers know what the
  description could resolve.
* ``tsmom_cross_asset``: ``research/prereg/2026-11-03-tsmom-cross-asset.md``.
  The registered ``research/tsmom/tsmom.py`` rule on 31 assets (30 FX
  excess returns sharing a dollar factor, one equity series), 132 hold-out
  months after a 24-month warm-up, NW lag 3, t > 1.645. Effect = the sd of
  a persistent (AR(1), phi 0.9) monthly drift, as in
  ``research/power/momentum-fx-g10.json``; each cell also reports the mean
  realised portfolio Sharpe so the cell can be read against the literature.

Worked example (``tests/test_simulate.py``): with nothing planted each
design rejects about 5 % of the time; with IC 0.10 on 30 independent
currencies the panel test rejects almost always (analytic power
``Phi(0.10 sqrt(30) sqrt(12) sqrt(10.83) - 1.645) ~ 1``).

Synthetic data only. Nothing here reads market data or the hold-out.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_R = Path(__file__).resolve().parents[1]
for sub in ("lib", "panel", "tsmom"):
    if str(_R / sub) not in sys.path:
        sys.path.insert(0, str(_R / sub))

import panel  # type: ignore[import-not-found]  # noqa: E402
import tsmom as ts  # type: ignore[import-not-found]  # noqa: E402
from study import nw_tstat  # type: ignore[import-not-found]  # noqa: E402

Z = 1.645
HOLDOUT_MONTHS = 130  # 2016-01 to 2026-10 inclusive
UST_MONTHS = 442  # 1990-01 to 2026-10 inclusive
TSMOM_MONTHS = 132  # 2016-01 to 2026-12 inclusive
_SYNTH_START = "1960-01-31"  # synthetic dates sit before the hold-out lock


def _months(n: int) -> pd.DatetimeIndex:
    return pd.date_range(_SYNTH_START, periods=n, freq="ME")


def _clustered(rng: np.random.Generator, t: int, n: int, clusters: int, rho: float) -> np.ndarray:
    """``t x n`` unit-variance normals, correlation ``rho`` inside each cluster."""
    lab = np.arange(n) % clusters
    common = rng.standard_normal((t, clusters))[:, lab]
    return math.sqrt(rho) * common + math.sqrt(1.0 - rho) * rng.standard_normal((t, n))


def combination_panel_once(
    rng: np.random.Generator,
    ic: float,
    n: int = 30,
    t: int = HOLDOUT_MONTHS,
    clusters: int = 5,
    rho_signal: float = 0.3,
    rho_noise: float = 0.3,
    vol_m: float = 0.03,
) -> float | None:
    """One draw: FM t-statistic (NW lag 3) of next-month returns on the score."""
    s = _clustered(rng, t, n, clusters, rho_signal)
    zs = (s - s.mean(axis=1, keepdims=True)) / s.std(axis=1, ddof=1, keepdims=True)
    noise = _clustered(rng, t, n, clusters, rho_noise)
    dollar = rng.standard_normal((t, 1))  # common move, removed by the FM demeaning
    r = vol_m * (ic * zs + math.sqrt(1.0 - ic * ic) * noise + dollar)
    idx = _months(t)
    cols = [f"C{i}" for i in range(n)]
    fm = panel.fama_macbeth(pd.DataFrame(zs, idx, cols), pd.DataFrame(r, idx, cols), nw_lag=3)
    return fm["t"]


def ust_once(rng: np.random.Generator, beta: float, t: int = UST_MONTHS, phi: float = 0.95) -> float | None:
    """One draw: NW t of a sign-of-carry rule; return = beta * carry + noise."""
    c = np.empty(t + 1)
    c[0] = rng.standard_normal()
    e = rng.standard_normal(t + 1)
    for k in range(1, t + 1):
        c[k] = phi * c[k - 1] + math.sqrt(1 - phi * phi) * e[k]
    r = beta * c[:-1] + rng.standard_normal(t)  # realised next-month excess return
    strat = np.sign(c[:-1]) * r
    return nw_tstat(pd.Series(strat), lag=3)


def tsmom_once(
    rng: np.random.Generator,
    drift_sd_annual: float,
    n_fx: int = 30,
    t: int = TSMOM_MONTHS,
    warmup: int = 24,
    phi: float = 0.9,
    dollar_load: float = 0.6,
) -> tuple[float | None, float | None]:
    """One draw: (NW t, annual Sharpe) of the registered TSMOM portfolio."""
    total = t + warmup
    vols = np.r_[np.full(n_fx, 0.10), 0.16] / math.sqrt(12)  # FX 10 %, equity 16 % a year
    n = n_fx + 1
    dollar = rng.standard_normal((total, 1))
    shocks = rng.standard_normal((total, n))
    shocks[:, :n_fx] = dollar_load * dollar + math.sqrt(1 - dollar_load**2) * shocks[:, :n_fx]
    sd_m = drift_sd_annual / 12.0
    mu = np.zeros((total, n))
    if sd_m > 0:
        mu[0] = rng.standard_normal(n) * sd_m
        inn = rng.standard_normal((total, n)) * sd_m * math.sqrt(1 - phi * phi)
        for k in range(1, total):
            mu[k] = phi * mu[k - 1] + inn[k]
    rx = pd.DataFrame(mu + vols * shocks, _months(total), [f"A{i}" for i in range(n)])
    port = ts.tsmom(rx)["portfolio"].iloc[-t:]
    sd = float(port.std(ddof=1))
    return nw_tstat(port, lag=3), (float(port.mean()) / sd * math.sqrt(12) if sd > 0 else None)


def _rate(ts_: list[float | None]) -> float:
    return float(np.mean([x is not None and x > Z for x in ts_]))


def _cell(name: str, value: float, truth: str, reps: int, rate: float, **extra: object) -> dict:
    return {
        name: value,
        "truth": truth,
        "reps": reps,
        "rejection_rate": rate,
        "mc_se": math.sqrt(rate * (1 - rate) / reps),
        **extra,
    }


def run_combination(reps: int = 400, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    for ic in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        t = [combination_panel_once(rng, ic) for _ in range(reps)]
        cells.append(_cell("ic_monthly", ic, "null" if ic == 0 else "planted IC", reps, _rate(t)))
    return {
        "what": "simulated power for research/combination/PREREG_DRAFT.md; synthetic data only",
        "design": "panel_fama_macbeth",
        "n_currencies": 30,
        "months": HOLDOUT_MONTHS,
        "clusters": 5,
        "rho_signal": 0.3,
        "rho_noise": 0.3,
        "decision_rule": "Fama-MacBeth mean slope, Newey-West lag 3, t > 1.645",
        "estimator": "research/lib/panel.py::fama_macbeth",
        "seed": seed,
        "cells": cells,
    }


def run_ust(reps: int = 1000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    for beta in (0.0, 0.05, 0.10, 0.15, 0.20):
        t = [ust_once(rng, beta) for _ in range(reps)]
        # Sharpe of sign(c) * (beta c + e) is about beta * E|c| / 1 per month
        sr = beta * math.sqrt(2 / math.pi) * math.sqrt(12)
        cells.append(
            _cell(
                "beta",
                beta,
                "null" if beta == 0 else "planted carry predictability",
                reps,
                _rate(t),
                approx_sharpe_annual=sr,
            )
        )
    return {
        "what": "simulated detection rate for research/ust/PREREG_DRAFT.md (descriptive); synthetic data only",
        "design": "single_series",
        "months": UST_MONTHS,
        "carry_ar1_phi": 0.95,
        "decision_rule": "descriptive; reported rate = share with Newey-West lag 3 t > 1.645",
        "costs": "not simulated (0.5 bp of DV01 per unit turnover in the registration)",
        "seed": seed,
        "cells": cells,
    }


def run_tsmom(reps: int = 300, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    for d in (0.0, 0.02, 0.04, 0.06, 0.08):
        out = [tsmom_once(rng, d) for _ in range(reps)]
        srs = [s for _, s in out if s is not None]
        cells.append(
            _cell(
                "drift_sd_annual",
                d,
                "null" if d == 0 else "planted momentum",
                reps,
                _rate([t for t, _ in out]),
                mean_sharpe=float(np.mean(srs)),
            )
        )
    return {
        "what": "simulated power for research/prereg/2026-11-03-tsmom-cross-asset.md; synthetic data only",
        "design": "single_series (one portfolio mean)",
        "assets": "30 FX (dollar loading 0.6, 10 % vol) + 1 equity (16 %)",
        "months": TSMOM_MONTHS,
        "warmup_months": 24,
        "monthly_drift_phi": 0.9,
        "rule": "research/tsmom/tsmom.py defaults (12-month sign, 40 % target, EWMA com 12)",
        "decision_rule": "portfolio mean, Newey-West lag 3, t > 1.645",
        "seed": seed,
        "cells": cells,
    }


def main(out_dir: Path | None = None) -> None:
    out_dir = out_dir or Path(__file__).resolve().parent
    for name, fn in (
        ("combination-panel.json", run_combination),
        ("ust-slope-fly.json", run_ust),
        ("tsmom-cross-asset.json", run_tsmom),
    ):
        doc = fn()
        (out_dir / name).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(name, [(c["truth"], round(c["rejection_rate"], 3)) for c in doc["cells"]])


if __name__ == "__main__":
    main()
