"""Seeded synthetic data: sovereign country-year panel, FX trade book, counterparties.

Everything here is deterministic given the ``seed`` and runs fully offline.
The sovereign panel has a *known ground truth*: a latent logistic
crisis/default model with

* a **nonlinear reserve-cover effect** — import cover only matters below a
  threshold (hinge ``max(0, THRESHOLD - cover)``), mirroring the empirical
  early-warning literature (reserves adequacy is a tail protection, not a
  linear driver);
* **panel structure** — country fixed effects and AR(1) feature dynamics, so
  rows within a country are serially correlated (which is exactly why a
  random row split leaks information; see ``cleaning.time_split``);
* **contagion years** — regional crisis flags in fixed calendar years
  (Asia 1997-98, LatAm 2001-02, EMEA 1998/2008) plus a *planted global
  contagion year in 2020* that lands in the out-of-time window and stresses
  calibration;
* **deliberately leaky post-crisis fields** (``imf_program_next_year``,
  ``devaluation_next_year_pct``) that are functions of the outcome — used to
  demonstrate the leakage guards in ``cleaning`` and the IV red-flag in
  ``woe``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..settlement import FXTrade

__all__ = [
    "RESERVE_COVER_THRESHOLD",
    "CRISIS_YEARS",
    "LEAKY_FIELDS",
    "FEATURES",
    "SCORECARD_FEATURES",
    "TRUE_BETA",
    "generate_sovereign_panel",
    "generate_logistic_data",
    "generate_fx_trade_book",
    "generate_counterparty_set",
]

#: Months of import cover below which reserves start to matter (true model).
RESERVE_COVER_THRESHOLD: float = 3.0

#: Regional crisis (contagion) years.  2020 is a planted *global* contagion
#: year that falls in the out-of-time test window of the example pipeline.
CRISIS_YEARS: dict[str, tuple[int, ...]] = {
    "Asia": (1997, 1998, 2020),
    "LatAm": (2001, 2002, 2020),
    "EMEA": (1998, 2008, 2020),
    "Africa": (2009, 2020),
}

#: Post-outcome fields that must never enter a PD model (target leakage).
LEAKY_FIELDS: tuple[str, ...] = ("imf_program_next_year", "devaluation_next_year_pct")

#: Legitimate model features, in canonical order.
FEATURES: tuple[str, ...] = (
    "reserves_import_cover",
    "ext_debt_gdp",
    "st_debt_reserves",
    "ca_gdp",
    "fiscal_gdp",
    "inflation",
    "fx_regime_peg",
    "commodity_dependence",
    "political_stability",
    "contagion",
)

#: Features used by the scorecard.  ``contagion`` is deliberately EXCLUDED:
#: it is flagged contemporaneously with the regional crisis (not observable
#: ex ante at the annual horizon), so using it would be borderline outcome
#: leakage; the desk applies contagion as a stress overlay instead.  Its
#: absence is what makes the planted 2020 contagion year a genuine
#: out-of-time calibration stress (see VALIDATION.md).
SCORECARD_FEATURES: tuple[str, ...] = tuple(f for f in FEATURES if f != "contagion")

#: Ground-truth coefficients of the latent crisis model (on natural units;
#: reserve cover enters through the hinge ``max(0, 3 - cover)``).
TRUE_BETA: dict[str, float] = {
    "intercept": -6.50,
    "reserve_hinge": 0.90,       # per month of shortfall below threshold
    "ext_debt_gdp": 0.012,       # per pct-of-GDP
    "st_debt_reserves": 0.85,    # Guidotti ratio
    "ca_gdp": -0.09,             # per pct-of-GDP
    "fiscal_gdp": -0.07,         # per pct-of-GDP
    "inflation": 0.014,          # per pct
    "fx_regime_peg": 0.45,       # pegs mask risk, then break
    "commodity_dependence": 0.007,
    "political_stability": -0.40,
    "contagion": 1.30,
}


def generate_sovereign_panel(
    n_countries: int = 60,
    start_year: int = 1995,
    end_year: int = 2023,
    seed: int = 42,
    missing_frac: float = 0.07,
) -> pd.DataFrame:
    """Generate a seeded sovereign country-year panel with known ground truth.

    Parameters
    ----------
    n_countries : int
        Number of synthetic sovereigns, spread over 4 regions.
    start_year, end_year : int
        Inclusive year range (annual macro observations).
    seed : int
        Seed for the ``numpy.random.Generator``; identical seeds reproduce
        the panel exactly.
    missing_frac : float
        Fraction of ``political_stability`` observations set missing (the
        WGI-style index has genuine gaps in real data).

    Returns
    -------
    pandas.DataFrame
        Columns: ``country, region, year``, the features in ``FEATURES``,
        the leaky fields in ``LEAKY_FIELDS``, ``true_pd`` (ground-truth
        1-year crisis probability) and the binary outcome ``default``
        (crisis/default event in the following year).  Base rate ~3-6%.
    """
    if end_year <= start_year:
        raise ValueError("end_year must be > start_year")
    rng = np.random.default_rng(seed)
    regions = ("Asia", "LatAm", "EMEA", "Africa")
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)

    rows: list[dict] = []
    for i in range(n_countries):
        region = regions[i % len(regions)]
        country = f"{region[:2].upper()}{i:03d}"
        # Country-level structural draws
        c_effect = rng.normal(0.0, 0.35)                    # latent fixed effect
        cover_mean = rng.uniform(1.5, 9.0)                  # months of imports
        extdebt_mean = rng.uniform(15.0, 110.0)             # % GDP
        guidotti_mean = rng.uniform(0.2, 1.8)               # ST debt / reserves
        ca_mean = rng.uniform(-7.0, 5.0)                    # % GDP
        fiscal_mean = rng.uniform(-8.0, 2.0)                # % GDP
        infl_mean = rng.uniform(1.0, 25.0)                  # % p.a.
        commodity = rng.uniform(5.0, 90.0)                  # % of exports
        polstab_mean = rng.normal(0.0, 1.0)                 # WGI-style index
        peg = int(rng.random() < 0.4)                       # FX regime dummy

        # AR(1) feature paths (persistence rho=0.8) around country means
        def ar1(mean: float, sd: float, lo: float, hi: float) -> np.ndarray:
            x = np.empty(n_years)
            x[0] = mean + rng.normal(0.0, sd)
            for t in range(1, n_years):
                x[t] = mean + 0.8 * (x[t - 1] - mean) + rng.normal(0.0, sd * 0.6)
            return np.clip(x, lo, hi)

        cover = ar1(cover_mean, 1.2, 0.1, 18.0)
        extdebt = ar1(extdebt_mean, 12.0, 2.0, 250.0)
        guidotti = ar1(guidotti_mean, 0.30, 0.02, 5.0)
        ca = ar1(ca_mean, 2.0, -20.0, 15.0)
        fiscal = ar1(fiscal_mean, 1.8, -20.0, 8.0)
        infl = np.exp(ar1(np.log(infl_mean), 0.45, -2.0, 5.0))
        polstab = ar1(polstab_mean, 0.25, -2.5, 2.5)

        contagion = np.array(
            [1.0 if y in CRISIS_YEARS[region] else 0.0 for y in years]
        )

        b = TRUE_BETA
        eta = (
            b["intercept"]
            + c_effect
            + b["reserve_hinge"] * np.maximum(0.0, RESERVE_COVER_THRESHOLD - cover)
            + b["ext_debt_gdp"] * extdebt
            + b["st_debt_reserves"] * guidotti
            + b["ca_gdp"] * ca
            + b["fiscal_gdp"] * fiscal
            + b["inflation"] * infl
            + b["fx_regime_peg"] * peg
            + b["commodity_dependence"] * commodity
            + b["political_stability"] * polstab
            + b["contagion"] * contagion
        )
        true_pd = 1.0 / (1.0 + np.exp(-eta))
        default = (rng.random(n_years) < true_pd).astype(int)

        # Leaky post-crisis fields: functions of the OUTCOME (forbidden).
        imf = np.where(default == 1, (rng.random(n_years) < 0.85), (rng.random(n_years) < 0.03)).astype(int)
        deval = np.where(
            default == 1,
            rng.uniform(20.0, 70.0, n_years),
            np.abs(rng.normal(0.0, 2.0, n_years)),
        )

        for t, y in enumerate(years):
            rows.append(
                {
                    "country": country,
                    "region": region,
                    "year": int(y),
                    "reserves_import_cover": cover[t],
                    "ext_debt_gdp": extdebt[t],
                    "st_debt_reserves": guidotti[t],
                    "ca_gdp": ca[t],
                    "fiscal_gdp": fiscal[t],
                    "inflation": infl[t],
                    "fx_regime_peg": peg,
                    "commodity_dependence": commodity,
                    "political_stability": polstab[t],
                    "contagion": contagion[t],
                    "imf_program_next_year": imf[t],
                    "devaluation_next_year_pct": deval[t],
                    "true_pd": true_pd[t],
                    "default": int(default[t]),
                }
            )

    df = pd.DataFrame(rows)
    # Genuine data gaps in the governance indicator
    mask = rng.random(len(df)) < missing_frac
    df.loc[mask, "political_stability"] = np.nan
    return df


def generate_logistic_data(
    beta: np.ndarray,
    intercept: float,
    n: int = 20_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean i.i.d. logistic data with known coefficients (for IRLS recovery tests).

    Features are standard normal; ``P(y=1|x) = sigmoid(intercept + x @ beta)``.

    Returns
    -------
    (X, y) : tuple of ndarray
        Design matrix (n, k) and binary outcomes (n,).
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    X = rng.standard_normal((n, beta.size))
    p = 1.0 / (1.0 + np.exp(-(intercept + X @ beta)))
    y = (rng.random(n) < p).astype(float)
    return X, y


def generate_fx_trade_book(seed: int = 7) -> list[FXTrade]:
    """Deterministic 6-trade FX settlement book across JPY/EUR/GBP/USD time zones.

    The book mixes CLS and non-CLS counterparties and includes the classic
    Herstatt direction (pay JPY/EUR early, receive USD late).  ``seed`` is
    accepted for interface uniformity; the book is fixed.
    """
    _ = seed
    return [
        FXTrade("T1", "TokyoBank", "USDJPY", 50_000_000, 148.0, we_buy_base=True, cls_settled=False),
        FXTrade("T2", "EuroBank", "EURUSD", 40_000_000, 1.08, we_buy_base=True, cls_settled=False),
        FXTrade("T3", "EuroBank", "EURUSD", 25_000_000, 1.09, we_buy_base=False, cls_settled=False),
        FXTrade("T4", "LondonBank", "GBPUSD", 30_000_000, 1.27, we_buy_base=True, cls_settled=True),
        FXTrade("T5", "TokyoBank", "EURJPY", 20_000_000, 159.8, we_buy_base=False, cls_settled=False),
        FXTrade("T6", "CLSBank", "USDJPY", 60_000_000, 147.5, we_buy_base=False, cls_settled=True),
    ]


def generate_counterparty_set(seed: int = 11) -> pd.DataFrame:
    """Deterministic counterparty reference set for CVA / limits demos.

    Returns
    -------
    pandas.DataFrame
        Columns: ``counterparty, rating, pd_1y, lgd, cls_member``.
        PDs are the rating-band midpoints used by the scorecard mapping.
    """
    _ = seed
    data = [
        ("TokyoBank", "AA", 0.0005, 0.40, False),
        ("EuroBank", "A", 0.0020, 0.40, False),
        ("LondonBank", "A", 0.0020, 0.40, True),
        ("CLSBank", "AA", 0.0005, 0.40, True),
        ("EMSovereignX", "BB", 0.0200, 0.55, False),
        ("FrontierY", "B", 0.0600, 0.60, False),
    ]
    return pd.DataFrame(
        data, columns=["counterparty", "rating", "pd_1y", "lgd", "cls_member"]
    )
