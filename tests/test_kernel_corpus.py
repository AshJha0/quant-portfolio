"""Characterisation corpus for the equity/FX twin kernels (W20, Round 8).

The plain idea. Before two copies of a kernel are merged into one, record
what each copy returns today on a fixed grid of inputs. The merge is then
judged against that record, not against the new code (Feathers, *Working
Effectively with Legacy Code*, ch. 13, "characterization tests").

The formal statement. ``tests/data/kernel_corpus.json`` holds, for each
twin family (CRR tree, SSVI total variance, Dupire local vol, Dupire local
vol Monte Carlo, Heston Fourier, Heston Monte Carlo), the inputs and the
outputs of both the equity and the FX implementation as captured on
``milestone/v2.1`` at ad91c80, before any W20 merge. Every value must
reproduce to ``|now - then| <= 1e-12 * max(1, |then|)``, the W20
acceptance bound. The FX rows use ``r_f`` where the equity rows use ``q``
(Garman-Kohlhagen is Black-Scholes with ``q = r_f``), and the corpus pairs
them on the same numbers, so the file also shows where the twins differ
from each other today (see ``test_twin_gap_is_recorded``).

Worked example. The first CRR row is S = K = 100, T = 1, r = 5 %,
q = r_f = 2 %, sigma = 20 %, European call, 1 step: the equity tree
returned 11.073540703840278 and the FX tree 11.073540703840242, 3.6e-14
apart (the FX tree built node spots as ``S u**m``, the equity tree as
``exp(ln S + m ln u)``). After W20 both wrappers call ``qp_core.trees``
and each must still match its own recorded number within 1e-12.

Regenerate (only before a deliberate, MIGRATION-recorded break)::

    python tests/test_kernel_corpus.py --write
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

CORPUS = Path(__file__).with_name("data") / "kernel_corpus.json"
TOL = 1e-12

eq_options = pytest.importorskip("eq_options")
fx_options = pytest.importorskip("fx_options")
eq_surface = pytest.importorskip("eq_surface")
fx_surface = pytest.importorskip("fx_surface")

from eq_options.binomial import crr_price  # noqa: E402
from eq_surface import dupire as eq_dupire  # noqa: E402
from eq_surface import heston as eq_heston  # noqa: E402
from eq_surface import heston_mc as eq_hmc  # noqa: E402
from eq_surface import ssvi as eq_ssvi  # noqa: E402
from fx_options.binomial import binomial_price  # noqa: E402
from fx_surface import dupire as fx_dupire  # noqa: E402
from fx_surface import heston as fx_heston  # noqa: E402
from fx_surface import heston_mc as fx_hmc  # noqa: E402
from fx_surface import ssvi as fx_ssvi  # noqa: E402

Row = dict[str, Any]


def _crr_rows() -> list[Row]:
    rows: list[Row] = []
    for S, K in ((100.0, 100.0), (100.0, 80.0), (100.0, 125.0), (1.3, 1.25)):
        for T, r, q, sig in ((1.0, 0.05, 0.02, 0.2), (0.25, -0.01, 0.03, 0.45),
                             (2.0, 0.03, 0.08, 0.1)):
            for ot in ("call", "put"):
                for ex in ("european", "american"):
                    for n in (1, 2, 7, 50, 201):
                        a = dict(S=S, K=K, T=T, r=r, q=q, sigma=sig, option_type=ot,
                                 exercise=ex, n=n)
                        rows.append({
                            "args": a,
                            "eq": crr_price(S, K, T, r, sig, q, ot, ex, n),
                            "fx": binomial_price(S, K, T, r, q, sig, ot, n, ex),
                        })
    return rows


SSVI_P = dict(rho=-0.4, eta=1.2, gamma=0.4)
K_GRID = [-1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0]
THETAS = [0.004, 0.02, 0.04, 0.09, 0.2]


def _theta_of_t(T: float) -> float:
    return 0.04 * T + 0.002 * math.sqrt(T)


def _w(mod: Any) -> Callable[[np.ndarray, float], np.ndarray]:
    p = mod.SSVIParams(**SSVI_P)
    return lambda k, T: p.total_variance(np.asarray(k, dtype=float), _theta_of_t(T))


def _floats(v: np.ndarray | float) -> list[float]:
    return [float(x) for x in np.atleast_1d(v)]


def _ssvi_rows() -> list[Row]:
    pe, pf = eq_ssvi.SSVIParams(**SSVI_P), fx_ssvi.SSVIParams(**SSVI_P)
    return [{"args": {"theta": th, "k": K_GRID},
             "eq": _floats(pe.total_variance(np.array(K_GRID), th)),
             "fx": _floats(pf.total_variance(np.array(K_GRID), th))}
            for th in THETAS]


def _dupire_rows() -> list[Row]:
    we, wf = _w(eq_ssvi), _w(fx_ssvi)
    return [{"args": {"T": T, "k": K_GRID},
             "eq": _floats(eq_dupire.local_vol_from_total_variance(we, np.array(K_GRID), T)),
             "fx": _floats(fx_dupire.local_vol_from_total_variance(wf, np.array(K_GRID), T))}
            for T in (0.1, 0.5, 1.0, 3.0)]


def _dupire_mc_rows() -> list[Row]:
    we, wf = _w(eq_ssvi), _w(fx_ssvi)
    K = np.array([80.0, 100.0, 120.0])
    pe, se = eq_dupire.local_vol_mc_price(we, 100.0, K, 1.0, 0.03, 0.01, 2_000, 20, 11)
    pf, sf = fx_dupire.local_vol_mc_price(wf, 100.0, K, 1.0, 0.03, 0.01, 2_000, 20, 11)
    return [{"args": {"S0": 100.0, "K": K.tolist(), "T": 1.0, "r": 0.03, "q": 0.01,
                      "n_paths": 2_000, "n_steps": 20, "seed": 11},
             "eq": [*map(float, pe), *map(float, se)],
             "fx": [*map(float, pf), *map(float, sf)]}]


HESTON_P = dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.6)


def _eq_heston_params() -> Any:
    names = ("v0", "kappa", "theta", "xi", "rho")
    fields = getattr(eq_heston.HestonParams, "__dataclass_fields__", {})
    kw = {n: HESTON_P[n] for n in names if n in fields}
    if "sigma" in fields:
        kw["sigma"] = HESTON_P["xi"]
    return eq_heston.HestonParams(**kw)


def _heston_rows() -> list[Row]:
    pe = _eq_heston_params()
    pf = fx_heston.HestonParams(**HESTON_P)
    rows: list[Row] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for K in (80.0, 100.0, 120.0):
            for T in (0.25, 1.0, 3.0):
                rows.append({
                    "args": {"S": 100.0, "K": K, "T": T, "r": 0.03, "q": 0.01},
                    "eq": [eq_heston.heston_call(100.0, K, T, 0.03, 0.01, pe, m)
                           for m in ("gl", "damped", "p1p2")],
                    "fx": [float(fx_heston.heston_price(100.0, K, T, 0.03, 0.01, pf, 1, m))
                           for m in ("cos", "gil_pelaez")],
                })
    return rows


def _heston_mc_rows() -> list[Row]:
    pe = _eq_heston_params()
    pf = fx_heston.HestonParams(**HESTON_P)
    rows: list[Row] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for scheme in ("qe", "euler_ft"):
            e = eq_hmc.heston_mc_price(100.0, 100.0, 1.0, 0.03, 0.01, pe, n_paths=4_000,
                                       n_steps=16, scheme=scheme, seed=5)
            f = fx_hmc.mc_price(100.0, 100.0, 1.0, 0.03, 0.01, pf, 1, n_paths=4_000,
                                n_steps=16, scheme=scheme, seed=5)
            rows.append({"args": {"scheme": scheme, "n_paths": 4_000, "n_steps": 16,
                                  "seed": 5},
                         "eq": _mc_numbers(e), "fx": _mc_numbers(f)})
    return rows


def _mc_numbers(res: Any) -> list[float]:
    if isinstance(res, tuple):
        return [float(v) for v in res]
    for a, b in (("price", "std_error"), ("price", "stderr"), ("price", "se")):
        if hasattr(res, a) and hasattr(res, b):
            return [float(getattr(res, a)), float(getattr(res, b))]
    return [float(res)]


FAMILIES: dict[str, Callable[[], list[Row]]] = {
    "crr": _crr_rows,
    "ssvi_total_variance": _ssvi_rows,
    "dupire_local_vol": _dupire_rows,
    "dupire_local_vol_mc": _dupire_mc_rows,
    "heston_fourier": _heston_rows,
    "heston_mc": _heston_mc_rows,
}


def _flat(v: Any) -> list[float]:
    return [float(x) for x in (v if isinstance(v, list) else [v])]


@pytest.fixture(scope="module")
def corpus() -> dict[str, list[Row]]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["families"]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_corpus_reproduces(corpus: dict[str, list[Row]], family: str) -> None:
    now = FAMILIES[family]()
    then = corpus[family]
    assert [r["args"] for r in now] == [r["args"] for r in then]
    worst = 0.0
    for a, b in zip(now, then, strict=True):
        for side in ("eq", "fx"):
            for x, y in zip(_flat(a[side]), _flat(b[side]), strict=True):
                worst = max(worst, abs(x - y) / max(1.0, abs(y)))
    assert worst <= TOL, f"{family}: worst scaled difference {worst:.3e}"


def test_twin_gap_is_recorded(corpus: dict[str, list[Row]]) -> None:
    """The deterministic twins agree with each other on the same numbers.

    CRR, SSVI and Dupire are the same maths in both packages, so the two
    recorded columns must already agree within the W20 bound; that is what
    makes a single shared kernel possible without a numerical break.
    """
    for family in ("crr", "ssvi_total_variance", "dupire_local_vol"):
        for row in corpus[family]:
            for x, y in zip(_flat(row["eq"]), _flat(row["fx"]), strict=True):
                assert abs(x - y) <= TOL * max(1.0, abs(y)), (family, row["args"])


def _write() -> None:
    CORPUS.parent.mkdir(exist_ok=True)
    data = {
        "description": "W20 characterisation corpus; see tests/test_kernel_corpus.py.",
        "captured_on": "milestone/v2.1 at ad91c80 (2026-11-03), before the W20 merge",
        "tolerance": "abs <= 1e-12 * max(1, |recorded|)",
        "families": {name: fn() for name, fn in FAMILIES.items()},
    }
    CORPUS.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {CORPUS}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write()
