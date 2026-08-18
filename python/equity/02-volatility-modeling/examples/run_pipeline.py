"""End-to-end pipeline: data -> models -> forecasts -> evaluation -> decision.

Simulates a realistic equity index return series WITH a leverage effect
(GJR-GARCH is the true data-generating process), fits all five volatility
models, then runs a 500-day out-of-sample 1-step forecast race with QLIKE/MSE
losses and Diebold-Mariano tests. Reproduces the numbers quoted in README.md
and docs/VALIDATION.md. Runtime target: well under 120s.

Run from the project root:  python examples/run_pipeline.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from eq_vol.data import synthetic as syn  # noqa: E402
from eq_vol import egarch as eg  # noqa: E402
from eq_vol import gjr as gj  # noqa: E402
from eq_vol.evaluation import (  # noqa: E402
    arch_lm_test,
    forecast_race_table,
    ljung_box_squared,
    mincer_zarnowitz,
    sign_bias_test,
)
from eq_vol.ewma import ewma_variance, lambda_to_halflife  # noqa: E402
from eq_vol.forecasting import rolling_one_step_forecasts, term_structure  # noqa: E402
from eq_vol.garch import fit_garch  # noqa: E402
from eq_vol.historical import realized_vol, window_sensitivity  # noqa: E402

pd.set_option("display.width", 110)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# true DGP: GJR-GARCH with a strong equity-style leverage effect
TRUE = {"omega": 3e-6, "alpha": 0.04, "gamma": 0.12, "beta": 0.87}  # P = 0.97
N_TRAIN, N_TEST = 3000, 500
SEED = 20260818


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    t_start = time.time()
    sim = syn.simulate_gjr(N_TRAIN + N_TEST, seed=SEED, **TRUE)
    r = sim.returns
    train = r[:N_TRAIN]

    hr("1. DATA: simulated equity returns, GJR(1,1) true model with leverage")
    print(f"true params: {TRUE}  (persistence {TRUE['alpha'] + TRUE['gamma']/2 + TRUE['beta']:.3f})")
    print(f"n_train={N_TRAIN}, n_test={N_TEST}, "
          f"ann. vol={np.sqrt(np.mean(train**2)*252):.2%}, "
          f"skew={pd.Series(train).skew():.2f}, kurt={pd.Series(train).kurt():.2f}")

    hr("2. HISTORICAL / EWMA baselines")
    print(f"21d realized vol (last): {realized_vol(train, 21)[-1]:.2%} | "
          f"63d: {realized_vol(train, 63)[-1]:.2%}")
    print("\nwindow sensitivity of rolling realized vol:")
    print(window_sensitivity(train, windows=(10, 21, 63, 126)))
    lam = 0.94
    ew = ewma_variance(train, lam=lam)
    print(f"\nEWMA(lambda={lam}, half-life {lambda_to_halflife(lam):.1f}d) "
          f"latest ann. vol: {np.sqrt(ew[-1]*252):.2%}")

    hr("3. MAXIMUM-LIKELIHOOD FITS (parameter tables with standard errors)")
    fits = {
        "GARCH": fit_garch(train),
        "EGARCH": eg.fit_egarch(train),
        "GJR": gj.fit_gjr(train),
    }
    for name, res in fits.items():
        print()
        print(res.summary())

    hr("4. IN-SAMPLE DIAGNOSTICS (does the symmetric model miss the leverage?)")
    rows = []
    for name, res in fits.items():
        lb_p = float(ljung_box_squared(res.std_residuals, 10)["lb_pvalue"].iloc[0])
        lm_p = arch_lm_test(res.std_residuals)["lm_pvalue"]
        sb = sign_bias_test(res)
        rows.append({"model": name, "aic": res.aic, "bic": res.bic,
                     "ljung_box_p": lb_p, "arch_lm_p": lm_p,
                     "sign_bias_joint_p": sb.loc["joint_F", "pvalue"]})
    diag = pd.DataFrame(rows).set_index("model")
    print(diag)
    d_aic = diag.loc["GARCH", "aic"] - diag.loc["GJR", "aic"]
    print(f"\n-> AIC/BIC strongly prefer the asymmetric models (GJR beats GARCH "
          f"by {d_aic:.1f} AIC points).")
    print("   Note: on 3,000 obs the sign-bias test has limited power against "
          "this leverage size;\n   with 10,000 obs it rejects the symmetric "
          "GARCH decisively (unit-tested in tests/test_evaluation.py).")

    hr("5. NEWS IMPACT CURVES (annualised next-day vol after a z-sigma shock)")
    z = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    pg = fits["GJR"].params
    pe = fits["EGARCH"].params
    _, v_gjr = gj.news_impact_curve(pg["omega"], pg["alpha"], pg["gamma"], pg["beta"], z_grid=z)
    _, v_eg = eg.news_impact_curve(pe["omega"], pe["alpha"], pe["gamma"], pe["beta"], z_grid=z)
    nic = pd.DataFrame({"z_shock": z,
                        "GJR_vol": np.sqrt(v_gjr * 252),
                        "EGARCH_vol": np.sqrt(v_eg * 252)}).set_index("z_shock")
    print(nic.map(lambda x: f"{x:.2%}"))
    print("\n-> a -2 sigma shock raises next-day vol far more than +2 sigma:")
    print(f"   GJR   : {np.sqrt(v_gjr[1]*252):.2%} vs {np.sqrt(v_gjr[5]*252):.2%}")
    print(f"   EGARCH: {np.sqrt(v_eg[1]*252):.2%} vs {np.sqrt(v_eg[5]*252):.2%}")

    hr("6. VOL TERM STRUCTURE (GJR forecast from the last training date)")
    ts = term_structure(fits["GJR"], horizon=252)
    print(ts.loc[[1, 5, 21, 63, 126, 252]].map(lambda x: f"{x:.2%}"))
    uncond = np.sqrt(fits["GJR"].extra["unconditional_variance"] * 252)
    print(f"-> converges toward the unconditional level {uncond:.2%}")

    hr(f"7. OUT-OF-SAMPLE FORECAST RACE ({N_TEST} days, 1-step, refit every 25d)")
    t0 = time.time()
    race_specs = {
        "historical_21d": dict(model="historical", hist_window=21),
        "ewma_0.94": dict(model="ewma", lam=0.94),
        "garch": dict(model="garch", refit_every=25),
        "egarch": dict(model="egarch", refit_every=25),
        "gjr": dict(model="gjr", refit_every=25),
    }
    forecasts = {}
    for name, spec in race_specs.items():
        model = spec.pop("model")
        res = rolling_one_step_forecasts(r, model, min_train=N_TRAIN, **spec)
        forecasts[name] = res.forecasts
    proxy = r[N_TRAIN:] ** 2
    table = forecast_race_table(forecasts, proxy, benchmark="garch")
    print(f"(race computed in {time.time() - t0:.1f}s; proxy = squared returns; "
          f"DM vs GARCH benchmark, negative = better than GARCH)\n")
    show = table.rename(columns={"mse": "mse_x1e8"})
    show["mse_x1e8"] = show["mse_x1e8"] * 1e8
    print(show)

    mz = mincer_zarnowitz(forecasts["gjr"], proxy)
    print(f"\nMincer-Zarnowitz (GJR): intercept={mz.intercept:.2e} "
          f"slope={mz.slope:.3f} (se {mz.slope_se:.3f}) R2={mz.r2:.3f} "
          f"joint p={mz.joint_pvalue:.3f}")

    names = list(table.index)
    print("\nQLIKE ranking (best -> worst): " + "  >  ".join(names))
    ok = (set(names[:2]) == {"gjr", "egarch"}
          and names[2:] == ["garch", "ewma_0.94", "historical_21d"])
    print("expected ordering (asymmetric > symmetric > EWMA > rolling window):",
          "CONFIRMED" if ok else "NOT confirmed on this seed")

    # oracle reference: the true conditional variance of the DGP
    q_oracle = float(np.mean(np.log(sim.sigma2[N_TRAIN:]) + proxy / sim.sigma2[N_TRAIN:]))
    print(f"oracle QLIKE (true conditional variance): {q_oracle:.4f}")

    print(f"\nTotal pipeline runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
