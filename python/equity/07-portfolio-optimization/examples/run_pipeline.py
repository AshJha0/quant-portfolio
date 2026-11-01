"""End-to-end pipeline: data -> estimation -> covariance -> MVO/frontier
-> Black-Litterman -> risk parity -> walk-forward race -> crisis analysis.

Reproduces every number quoted in README.md and docs/VALIDATION.md.
Deterministic (seeded); runs offline. Prints only and writes no files.
All data are synthetic.

Usage:  python examples/run_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eq_port as ep  # noqa: E402
from eq_port.data import generate_panel  # noqa: E402

PPY = 252.0
SEED = 1  # main panel seed
STUDY_SEEDS = list(range(7, 17))  # estimation-error study panels
RACE_SEEDS = list(range(1, 21))  # walk-forward race panels


def hr(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def ci_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Column means with a 95% t-interval across rows (independent panels)."""
    n = len(df)
    se = df.std(ddof=1) / np.sqrt(n)
    q = float(stats.t.ppf(0.975, n - 1))
    return pd.DataFrame({label: df.mean(), "CI low": df.mean() - q * se,
                         "CI high": df.mean() + q * se})


def avg_corr(c: np.ndarray) -> float:
    d = np.sqrt(np.diag(c))
    corr = c / np.outer(d, d)
    n = len(c)
    return float((corr.sum() - n) / (n * (n - 1)))


def main() -> None:
    t0 = time.time()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")

    # ------------------------------------------------------------------ data
    hr("1. SYNTHETIC 8-ASSET PANEL (market+sector factors, crisis regime)")
    panel = generate_panel(
        n_assets=8, n_periods=2400, seed=SEED, regimes=True,
        crisis_start_frac=0.75, crisis_len_frac=0.05,
        crisis_vol_mult=3.0, crisis_mkt_drift=-1.2,
    )
    r = panel.returns
    names = panel.asset_names
    print(f"T = {len(r)} days, N = {r.shape[1]} assets, "
          f"crisis days = {panel.crisis_mask.sum()}")
    print(f"true annual means : {np.round(panel.true_mean * PPY, 4)}")
    print(f"true annual vols  : "
          f"{np.round(np.sqrt(np.diag(panel.true_cov) * PPY), 4)}")
    print(f"avg TRUE pairwise correlation: calm {avg_corr(panel.true_cov):.3f}"
          f"  crisis {avg_corr(panel.crisis_cov):.3f}")

    # -------------------------------------------- estimation error, in numbers
    hr("2. ESTIMATION ERROR (20 disjoint 252-day windows x 10 synthetic panels)")
    print("Estimate on each window, then evaluate the portfolio under the TRUE\n"
          "moments of that synthetic panel (no evaluation noise). Seeds 7-16.")
    cols = ["Tan raw (unconstrained)", "Tan raw (long-only)", "Tan JS (long-only)",
            "MinVar LW (long-only)", "ERC LW", "EqualWeight"]
    seed_means, all_recs = [], []
    n_fail = n_win = fb_raw = fb_js = 0
    gross = []
    for st_seed in STUDY_SEEDS:
        study = generate_panel(n_assets=8, n_periods=252 * 21, seed=st_seed, regimes=False)
        sr_data = study.returns.to_numpy()
        mu_t, cov_t = study.true_mean, study.true_cov

        def true_sr(w: np.ndarray, mu_t=mu_t, cov_t=cov_t) -> float:
            return ep.portfolio_return(w, mu_t) / ep.portfolio_vol(w, cov_t) * np.sqrt(PPY)

        recs = []
        for k in range(20):
            est = sr_data[k * 252 : (k + 1) * 252]
            mu_raw = est.mean(axis=0)
            js = ep.james_stein_mean(est)
            lw = ep.ledoit_wolf_cc(est)
            sig = ep.psd_repair(lw.cov)
            sig_raw = ep.psd_repair(ep.sample_cov(est))
            t_raw = ep.max_sharpe_qp(mu_raw, sig, bounds=(0, 1))
            t_js = ep.max_sharpe_qp(js.mean, sig, bounds=(0, 1))
            fb_raw += t_raw.fallback
            fb_js += t_js.fallback
            rec = {
                "Tan raw (long-only)": true_sr(t_raw.weights),
                "Tan JS (long-only)": true_sr(t_js.weights),
                "MinVar LW (long-only)": true_sr(ep.min_variance_constrained(sig, bounds=(0, 1))),
                "ERC LW": true_sr(ep.erc_weights(sig)),
                "EqualWeight": true_sr(np.full(8, 1 / 8)),
                "phi_JS": js.intensity,
                "delta_LW": lw.intensity,
                "achievable": true_sr(ep.tangency_weights(mu_t, cov_t)),
            }
            try:
                w_u = ep.tangency_weights(mu_raw, sig_raw)
                rec["Tan raw (unconstrained)"] = true_sr(w_u)
                gross.append(np.abs(w_u).sum())
            except ValueError:
                n_fail += 1  # 1'S^{-1}mu <= 0: closed form refuses
            n_win += 1
            recs.append(rec)
        seed_means.append(pd.DataFrame(recs).mean())
        all_recs.extend(recs)
    sm = pd.DataFrame(seed_means)
    print(f"mean achievable (true tangency) Sharpe: {sm['achievable'].mean():.3f}")
    print(ci_table(sm[cols], "mean true Sharpe"))
    print("\npaired difference vs MinVar LW (per-panel means, 95% t-interval):")
    print(ci_table(sm[cols].sub(sm["MinVar LW (long-only)"], axis=0)
                   .drop(columns="MinVar LW (long-only)"), "diff vs MinVar"))
    print(f"\nmin-variance fallback (no estimated mean > rf): Tan raw {fb_raw}/{n_win}, "
          f"Tan JS {fb_js}/{n_win} windows")
    print(f"unconstrained raw tangency refused (1'S^-1 mu <= 0) in {n_fail}/{n_win} "
          f"windows; gross leverage mean {np.mean(gross):.1f}x, max {np.max(gross):.1f}x")
    dfa = pd.DataFrame(all_recs)
    print(f"avg shrinkage intensities: James-Stein phi = {dfa['phi_JS'].mean():.2f}, "
          f"Ledoit-Wolf delta = {dfa['delta_LW'].mean():.2f}")

    study0 = generate_panel(n_assets=8, n_periods=252, seed=STUDY_SEEDS[0], regimes=False)
    lw_full = ep.ledoit_wolf_cc(study0.returns.to_numpy())
    print(f"LW conditioning (one 252d window): cond(sample) "
          f"{ep.condition_number(lw_full.sample):.1f} -> cond(shrunk) "
          f"{ep.condition_number(lw_full.cov):.1f}")

    # ------------------------------------------------------------- frontier
    hr("3. EFFICIENT FRONTIER (true moments; annualised)")
    fr = ep.efficient_frontier(panel.true_mean, panel.true_cov, n_points=8)
    # long-only minimum vol at exactly each target return; NaN where the
    # target is outside (min(mu), max(mu)) and so unattainable long-only
    long_only_vols = np.array([
        ep.portfolio_vol(ep.target_return_portfolio(
            panel.true_mean, panel.true_cov, t, bounds=(0.0, 1.0)), panel.true_cov)
        if panel.true_mean.min() < t < panel.true_mean.max() else np.nan
        for t in fr.returns
    ])
    frontier_tbl = pd.DataFrame({
        "AnnRet": fr.returns * PPY,
        "AnnVol (unconstr)": fr.vols * np.sqrt(PPY),
        "AnnVol (long-only)": long_only_vols * np.sqrt(PPY),
        "MaxWeight (unconstr)": fr.weights.max(axis=1),
    })
    print(frontier_tbl.to_string(index=False))
    print(f"\ntrue tangency Sharpe (annualised): "
          f"{ep.portfolio_return(ep.tangency_weights(panel.true_mean, panel.true_cov), panel.true_mean) / ep.portfolio_vol(ep.tangency_weights(panel.true_mean, panel.true_cov), panel.true_cov) * np.sqrt(PPY):.3f}")

    # ------------------------------------------------------- Black-Litterman
    hr("4. BLACK-LITTERMAN (reverse-optimized prior + one view)")
    pi = ep.implied_equilibrium_returns(panel.true_cov, panel.market_weights,
                                        risk_aversion=2.5)
    w_back = ep.tangency_weights(pi, panel.true_cov)
    print(f"round-trip: max|tangency(pi) - market weights| = "
          f"{np.abs(w_back - panel.market_weights).max():.2e}")
    p_view = np.zeros((1, 8))
    p_view[0, 0], p_view[0, 1] = 1.0, -1.0
    view_ann = 0.03
    bl = ep.black_litterman(pi, panel.true_cov, p_view, [view_ann / PPY], tau=0.05)
    bl_tbl = pd.DataFrame({
        "prior (ann)": pi * PPY,
        "posterior (ann)": bl.mean * PPY,
        "shift (ann)": (bl.mean - pi) * PPY,
    }, index=names)
    print(f"view: {names[0]} beats {names[1]} by {view_ann:.0%}/yr "
          f"(prior spread {float((p_view @ pi)[0]) * PPY: .2%})")
    print(bl_tbl)
    w_bl = ep.max_sharpe_constrained(bl.mean, bl.cov, bounds=(0, 1))
    w_prior = ep.max_sharpe_constrained(pi, panel.true_cov * 1.05, bounds=(0, 1))
    print(f"long-only tangency weight on {names[0]}: prior {w_prior[0]:.3f} -> "
          f"posterior {w_bl[0]:.3f}; on {names[1]}: {w_prior[1]:.3f} -> "
          f"{w_bl[1]:.3f}")

    # ------------------------------------------------------------ risk parity
    hr("5. RISK PARITY (ERC): risk contributions and vol targeting")
    w_erc = ep.erc_weights(panel.true_cov)
    w_ew = np.full(8, 1 / 8)
    rc_tbl = pd.DataFrame({
        "ERC weight": w_erc,
        "ERC risk contrib %": ep.risk_contributions(w_erc, panel.true_cov)
        / (w_erc @ panel.true_cov @ w_erc) * 100,
        "EW risk contrib %": ep.risk_contributions(w_ew, panel.true_cov)
        / (w_ew @ panel.true_cov @ w_ew) * 100,
    }, index=names)
    print(rc_tbl)
    erc_vol = ep.portfolio_vol(w_erc, panel.true_cov) * np.sqrt(PPY)
    print(f"ERC ex-ante annual vol (unlevered): {erc_vol:.2%}")
    w_lev = ep.vol_target_overlay(w_erc, panel.true_cov, target_vol=0.10)
    print(f"vol-targeted to 10%: leverage = {w_lev.sum():.2f}x, ex-ante vol = "
          f"{np.sqrt(w_lev @ panel.true_cov @ w_lev * PPY):.2%}")
    print(f"diversification ratio: ERC "
          f"{ep.diversification_ratio(w_erc, panel.true_cov):.3f}, EW "
          f"{ep.diversification_ratio(w_ew, panel.true_cov):.3f}")

    # ------------------------------------------------------ walk-forward race
    hr("6. WALK-FORWARD RACE (252d window, monthly rebalance, 10bp costs)")
    static_tilt = np.array([0.15] * 4 + [0.10] * 4)

    def make_strategies() -> dict:
        return {
            "EqualWeight": ep.strategy_equal_weight,
            "MinVar (LW)": ep.make_min_variance_strategy(use_lw=True),
            "Tangency raw": ep.make_tangency_strategy(shrink_mean=False, use_lw=True),
            "Tangency JS": ep.make_tangency_strategy(shrink_mean=True, use_lw=True),
            "ERC (LW)": ep.make_erc_strategy(use_lw=True),
            "Static tilt": ep.make_static_strategy(static_tilt),
        }

    strategies = make_strategies()
    race = ep.run_race(r, strategies, window=252, rebalance_every=21, cost_bps=10.0)

    metrics = ep.summary_table({k: v.net_returns for k, v in race.items()})
    metrics["AnnTurnover"] = [
        race[k].turnover.sum() / (len(race[k].net_returns) / PPY)
        for k in metrics.index
    ]
    metrics["TotalCost"] = [race[k].total_cost for k in metrics.index]
    metrics["AvgEffN"] = [
        np.mean([ep.effective_n(w) for w in race[k].weights.to_numpy()])
        for k in metrics.index
    ]
    print(f"main panel (seed {SEED}):")
    print(metrics)
    print("\nSharpe with Lo (2002) autocorrelation-adjusted SE (seed "
          f"{SEED} panel):")
    for k, res in race.items():
        lo = ep.sharpe_lo(res.net_returns)
        print(f"  {k:14s} SR = {lo.sharpe: .3f}  (Lo-adj {lo.sharpe_lo: .3f}, "
              f"SE {lo.se:.3f})")

    print("\nJobson-Korkie/Memmel test vs MinVar (LW), main panel:")
    for k, res in race.items():
        if k == "MinVar (LW)":
            continue
        t = ep.sharpe_difference_test(res.net_returns, race["MinVar (LW)"].net_returns)
        print(f"  {k:14s} dSR = {t.diff: .3f} (SE {t.se:.3f}, p = {t.p_value:.3f})")

    print(f"\nnet Sharpe across {len(RACE_SEEDS)} independent synthetic panels "
          f"(seeds {RACE_SEEDS[0]}-{RACE_SEEDS[-1]}):")
    multi, turn, fb = {}, {}, {}
    for seed in RACE_SEEDS:
        strategies_s = strategies if seed == SEED else make_strategies()
        p_s = panel if seed == SEED else generate_panel(
            n_assets=8, n_periods=2400, seed=seed, regimes=True,
            crisis_start_frac=0.75, crisis_len_frac=0.05,
            crisis_vol_mult=3.0, crisis_mkt_drift=-1.2,
        )
        race_s = race if seed == SEED else ep.run_race(
            p_s.returns, strategies_s, window=252, rebalance_every=21, cost_bps=10.0)
        multi[seed] = {k: ep.sharpe_ratio(v.net_returns) for k, v in race_s.items()}
        turn[seed] = {k: v.turnover.sum() / (len(v.net_returns) / PPY)
                      for k, v in race_s.items()}
        fb[seed] = {k: strategies_s[k].stats["fallbacks"] / strategies_s[k].stats["calls"]
                    for k in ("Tangency raw", "Tangency JS")}
    multi_df = pd.DataFrame(multi).T
    print(ci_table(multi_df, "mean net Sharpe"))
    print("\npaired difference vs MinVar (LW) across panels (95% t-interval):")
    diff = multi_df.sub(multi_df["MinVar (LW)"], axis=0).drop(columns="MinVar (LW)")
    tbl = ci_table(diff, "diff vs MinVar")
    tbl["panels above"] = (diff > 0).sum()
    print(tbl)
    print("\nmean annual turnover across panels:")
    print(pd.DataFrame(turn).T.mean().to_string())
    fbm = pd.DataFrame(fb).T.mean()
    print(f"\nshare of rebalances in min-variance fallback: Tangency raw "
          f"{fbm['Tangency raw']:.1%}, Tangency JS {fbm['Tangency JS']:.1%}")

    # --------------------------------------------------------- crisis analysis
    hr("7. CRISIS-REGIME SUBPERIOD (correlations jump toward 1)")
    span_mask = pd.Series(panel.crisis_mask, index=r.index).iloc[252:].to_numpy()
    calm_r = r.iloc[252:][~span_mask].to_numpy()
    crisis_r = r.iloc[252:][span_mask].to_numpy()
    print(f"realized avg pairwise correlation: calm "
          f"{avg_corr(np.cov(calm_r.T)):.3f} -> crisis "
          f"{avg_corr(np.cov(crisis_r.T)):.3f}")
    crisis_rows = {}
    for k, res in race.items():
        cr = res.net_returns[span_mask]
        calm = res.net_returns[~span_mask]
        crisis_rows[k] = {
            "Crisis AnnVol": ep.annualized_vol(cr),
            "Crisis MaxDD": ep.max_drawdown(cr),
            "Crisis TotRet": float(np.prod(1 + cr) - 1),
            "Calm Sharpe": ep.sharpe_ratio(calm),
        }
    print(pd.DataFrame(crisis_rows).T)
    print(f"\npipeline wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
