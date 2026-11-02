"""Regenerate the numeric tables in docs/VALIDATION.md (synthetic data).

Sections:
  mc     Monte Carlo bias vs Fourier, ATM 1y call, 400k paths (VALIDATION 3)
  svi    Heston calibrated to the SVI-generated chain, residuals by expiry (5.1)
  ridge  three recalibrations under fresh 0.3 vol-point noise (5.2)

Run:  python examples/validation_tables.py [mc|svi|ridge ...]   (offline, seeded)
Nothing is written to disk.
"""

from __future__ import annotations

import sys
import time
import warnings

import eq_surface as es
import numpy as np
import pandas as pd
from eq_surface.data import generate_chain

pd.set_option("display.width", 120)


def _market(chain):
    S, r, q = chain.spot, chain.rate, chain.div_yield
    Ts, Ks, IVs = [], [], []
    for T in sorted(chain.df.expiry.unique()):
        sl = chain.slice(T)
        iv = es.implied_vol_vector(sl.call_mid.values, S, sl.strike.values, float(T), r, q)
        ok = np.isfinite(iv)
        Ts.append(float(T))
        Ks.append(sl.strike.values[ok])
        IVs.append(iv[ok])
    return S, r, q, np.array(Ts), Ks, IVs


def mc_table() -> None:
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    sets = {
        "mild (Feller satisfied)": es.HestonParams(v0=0.04, kappa=2.0, theta=0.04, rho=-0.5, xi=0.3),
        "extreme (Feller ratio 0.08)": es.HestonParams(v0=0.04, kappa=1.0, theta=0.04, rho=-0.9, xi=1.0),
    }
    for name, p in sets.items():
        ref = float(es.heston_call_gl(S, K, T, r, q, p))
        rows = []
        for steps in (4, 8, 16, 32, 64):
            row = {"steps/yr": steps}
            for scheme in ("euler_ft", "qe"):
                res = es.heston_mc_price(S, K, T, r, q, p, n_paths=400_000, n_steps=steps,
                                         scheme=scheme, seed=1)
                err = res.price - ref
                row[scheme] = f"{err:+.4f} ({err / res.stderr:+.1f} SE)"
            rows.append(row)
        print(f"\n{name}: Fourier reference {ref:.4f}")
        print(pd.DataFrame(rows).to_string(index=False))


def svi_failure_table() -> None:
    chain = generate_chain(mode="svi", seed=7)
    S, r, q, Ts, Ks, IVs = _market(chain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = es.calibrate_heston(S, r, q, Ts, Ks, IVs, n_starts=3, seed=0)
    model = es.heston_model_ivs(S, r, q, Ts, Ks, res.params)
    rows = []
    for T, K, iv, ivm in zip(Ts, Ks, IVs, model, strict=False):
        d = (ivm - iv) * 100.0
        F = S * np.exp((r - q) * T)
        j = int(np.argmin(np.abs(np.log(K / F))))
        rows.append({"T": round(T, 4), "rmse_vp": np.sqrt(np.nanmean(d * d)),
                     "min_vp": np.nanmin(d), "atm_vp": d[j], "max_vp": np.nanmax(d)})
    print(f"\nHeston fitted to the SVI chain (seed 7): {res.params}")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def ridge_table() -> None:
    chain = generate_chain(mode="heston", seed=42)
    S, r, q, Ts, Ks, IVs = _market(chain)
    rows = []
    for day, seed in enumerate((11, 22, 33), start=1):
        rng = np.random.default_rng(seed)
        noisy = [iv + 0.003 * rng.standard_normal(iv.size) for iv in IVs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = es.calibrate_heston(S, r, q, Ts, Ks, noisy, n_starts=3, seed=0)
        p = res.params
        rows.append({"day": day, "v0": p.v0, "kappa": p.kappa, "theta": p.theta,
                     "rho": p.rho, "xi": p.xi, "rmse_vp": res.rmse_vol_points,
                     "cond_J": f"{res.condition_number:.1e}"})
    print(f"\nTruth: {chain.true_heston}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1:] or ["mc", "svi", "ridge"]
    t0 = time.time()
    for name in which:
        {"mc": mc_table, "svi": svi_failure_table, "ridge": ridge_table}[name]()
    print(f"\ntotal {time.time() - t0:.0f}s")
