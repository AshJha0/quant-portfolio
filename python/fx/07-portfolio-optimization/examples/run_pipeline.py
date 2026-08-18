"""End-to-end FX portfolio optimization pipeline.

data -> total returns -> style portfolios -> covariance -> allocation
(MVO frontier / ERC / CVaR-constrained) -> optimal hedging -> walk-forward
race with costs and carry -> GBP-base reporting.

Runs offline on the seeded synthetic panel in well under 120 s and prints the
numbers quoted in README.md / docs/VALIDATION.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fx_port as fp  # noqa: E402
from fx_port.data import make_equity_portfolio, make_panel  # noqa: E402

SEED = 123
N_DAYS = 3024  # ~12 years of business days
ALPHA = 0.95


def fmt(x: float, pct: bool = False, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%" if pct else f"{x:.{digits}f}"


def stat_line(name: str, s: dict[str, float]) -> str:
    return (
        f"  {name:<18} ann.ret {fmt(s['ann_return'], True):>8}  "
        f"vol {fmt(s['ann_vol'], True):>7}  SR {fmt(s['sharpe']):>5} "
        f"(SE {fmt(s['sharpe_se'])})  skew {fmt(s['skew']):>6}  "
        f"MDD {fmt(s['max_drawdown'], True):>7}  "
        f"CVaR95 {fmt(s['cvar'], True, 2):>6}"
    )


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("FX PORTFOLIO OPTIMIZATION & CURRENCY RISK ALLOCATION  (fx_port)")
    print(f"seed={SEED}, {N_DAYS} business days, universe = G10 ex-USD + MXN/BRL/TRY")
    print("=" * 78)

    # ------------------------------------------------------------------ 1
    panel = make_panel(seed=SEED, n_days=N_DAYS)
    ccys = list(panel.spots.columns)
    dec = fp.total_log_returns(panel.spots, panel.rates)
    print("\n[1] Currency total returns = spot + carry (exact decomposition)")
    ident = (dec.total - (dec.spot + dec.carry)).abs().to_numpy().max()
    print(f"  max |total - (spot + carry)| = {ident:.2e}")
    avg_carry = (dec.carry.mean() * 252).sort_values()
    print(
        "  ann. carry accrual: "
        + ", ".join(f"{c} {fmt(avg_carry[c], True, 1)}" for c in ["JPY", "CHF", "EUR", "AUD", "MXN", "BRL", "TRY"])
    )

    # ------------------------------------------------------------------ 2
    print("\n[2] Style portfolios (dollar-neutral, gross = 2.0, 1-day lag)")
    gross = 2.0
    idx = dec.total.index
    sig_carry = fp.carry_signal(panel.rates, ccys).reindex(idx)
    sig_mom = fp.momentum_signal(panel.spots).reindex(idx)
    sig_val = fp.value_signal(panel.spots, panel.ppp).reindex(idx)
    styles = {}
    for name, sig in (("carry", sig_carry), ("momentum", sig_mom), ("value", sig_val)):
        ret, w = fp.style_returns(dec.total, sig, gross=gross)
        live = ret[(w.abs().sum(axis=1) > 0)]
        styles[name] = live
        print(stat_line(name.upper(), fp.summary(live, ALPHA)))
    style_panel = pd.DataFrame(styles).dropna()
    print(
        "  -> carry: positive premium, NEGATIVE skew (crash risk); "
        "momentum/value: milder tails"
    )

    # ------------------------------------------------------------------ 3
    print("\n[3] Multi-style covariance and efficient frontier (sum(w)=1)")
    _, lam_ccy = fp.shrunk_means(dec.total)
    print(f"  currency-level James-Stein shrinkage intensity (12 ccys): "
          f"{lam_ccy:.2f}")
    style_mu, lam = fp.shrunk_means(style_panel)
    style_cov = style_panel.cov()
    print(f"  style means (lam={lam:.2f}: JS needs K>3, no shrink across 3 styles):",
          {k: f"{v * 252:.1%}" for k, v in style_mu.items()})
    print("  style correlations:")
    corr = style_panel.corr()
    for a in corr.index:
        print(f"    {a:<9}", " ".join(f"{corr.loc[a, b]:+.2f}" for b in corr.columns))
    front = fp.efficient_frontier(
        style_mu, style_cov, n_points=7, sum_to=1.0, gross_limit=1.5
    )
    print("  frontier (target ann.ret -> ann.vol):")
    for _, row in front.iterrows():
        print(
            f"    {fmt(row['target_return'] * 252, True):>7} -> "
            f"{fmt(row['volatility'] * np.sqrt(252), True):>6}"
        )

    # ------------------------------------------------------------------ 4
    print("\n[4] ERC across styles + 5% vol target")
    w_erc = fp.erc_weights(style_cov)
    rc = fp.risk_contributions(w_erc, style_cov)
    print("  ERC weights:", {k: fmt(float(v)) for k, v in w_erc.items()},
          "| risk contribs:", {k: fmt(float(v)) for k, v in rc.items()})
    w_tgt = fp.vol_target(w_erc, style_cov, target_vol=0.05)
    print(f"  vol-targeted scale {w_tgt.sum() / w_erc.sum():.3f}x -> ex-ante vol "
          f"{fmt(fp.portfolio_vol(w_tgt, style_cov), True)}")

    # ------------------------------------------------------------------ 5
    print("\n[5] Skew-aware carry sizing: CVaR budget vs mean-variance")
    carry_r = style_panel["carry"]
    carry_stats = fp.summary(carry_r, ALPHA)
    vol_size = 0.05 / carry_stats["ann_vol"]  # naive vol-target sizing to 5%
    cvar_budget = 0.005  # 0.5% daily CVaR95 budget
    cvar_size, cvar_at = fp.carry_sizing(
        carry_r, alpha=ALPHA, cvar_limit=cvar_budget, max_leverage=vol_size
    )
    print(f"  carry sleeve: SR {fmt(carry_stats['sharpe'])}, "
          f"skew {fmt(carry_stats['skew'])}, CVaR95 {fmt(carry_stats['cvar'], True)}/day")
    print(f"  vol-target sizing (5% ann):    size = {fmt(vol_size)}x  -> "
          f"CVaR95 {fmt(vol_size * carry_stats['cvar'], True)}/day")
    print(f"  CVaR-budget sizing (<= {fmt(cvar_budget, True)}): size = "
          f"{fmt(cvar_size)}x  -> CVaR95 {fmt(cvar_at, True)}/day "
          f"({fmt(1 - cvar_size / vol_size, True, 0)} smaller book)")
    # multi-style LP: max return s.t. CVaR budget
    lp = fp.max_return_cvar_constrained(
        style_panel, alpha=ALPHA, cvar_limit=0.004, sum_to=None, gross_limit=1.5
    )
    lp_free = fp.max_return_cvar_constrained(
        style_panel, alpha=ALPHA, cvar_limit=10.0, sum_to=None, gross_limit=1.5
    )
    tail_free = fp.empirical_cvar(style_panel @ lp_free.weights, ALPHA)
    wf = {k: round(float(v), 2) for k, v in lp_free.weights.items()}
    wc = {k: round(float(v), 2) for k, v in lp.weights.items()}
    print("  multi-style RU-LP, gross<=1.5:")
    print(f"    unconstrained mean-chaser: {wf}  CVaR95 {fmt(tail_free, True)}, "
          f"E[ret] {fmt(lp_free.expected_return * 252, True, 1)}/yr")
    print(f"    CVaR95 <= 0.40%:           {wc}  CVaR95 {fmt(lp.cvar, True)}, "
          f"E[ret] {fmt(lp.expected_return * 252, True, 1)}/yr")
    print(f"    -> tail cut {fmt(1 - lp.cvar / tail_free, True, 0)} for a return "
          f"give-up of {fmt((lp_free.expected_return - lp.expected_return) * 252, True, 1)}/yr")

    # ------------------------------------------------------------------ 6
    print("\n[6] Optimal currency hedging (international equity portfolio)")
    eq = make_equity_portfolio(seed=SEED, n_days=N_DAYS)
    rep = fp.variance_decomposition(eq.unhedged_returns, eq.fx_returns, eq.exposures)
    print("  optimal hedge ratios:",
          {k: fmt(float(v)) for k, v in rep.hedge_ratios.items()})
    print(f"  ann.vol: unhedged {fmt(np.sqrt(rep.var_unhedged * 252), True)}, "
          f"full hedge {fmt(np.sqrt(rep.var_full * 252), True)}, "
          f"optimal {fmt(np.sqrt(rep.var_optimal * 252), True)}")
    print(f"  variance reduction: full {fmt(rep.reduction_full, True, 1)} vs "
          f"optimal {fmt(rep.reduction_optimal, True, 1)} -> "
          "full hedging is NOT optimal: safe havens (JPY/CHF) are underhedged"
          f" (h_JPY={fmt(rep.hedge_ratios['JPY'])}, h_CHF={fmt(rep.hedge_ratios['CHF'])})")

    # ------------------------------------------------------------------ 7
    print("\n[7] Walk-forward race across styles (est=504d, monthly rebalance,")
    print("    5 bps style-level costs, carry accrued daily, no lookahead)")
    # style sleeves decomposed into spot and carry legs for the ledger
    spot_leg, carry_leg = {}, {}
    for name, sig in (("carry", sig_carry), ("momentum", sig_mom), ("value", sig_val)):
        _, w_applied = fp.style_returns(dec.total, sig, gross=gross)
        spot_leg[name] = (w_applied * dec.spot).sum(axis=1)
        carry_leg[name] = (w_applied * dec.carry).sum(axis=1)
    spot_leg = pd.DataFrame(spot_leg).loc[style_panel.index]
    carry_leg = pd.DataFrame(carry_leg).loc[style_panel.index]

    def alloc_ew(hist: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0 / 3, index=hist.columns)

    def alloc_mvo(hist: pd.DataFrame) -> pd.Series:
        h = hist.tail(504)
        mu, _ = fp.shrunk_means(h)
        res = fp.max_utility(mu, h.cov(), gamma=40.0, sum_to=1.0,
                             gross_limit=1.5)
        return res.weights

    def alloc_erc(hist: pd.DataFrame) -> pd.Series:
        return fp.erc_weights(hist.tail(504).cov())

    def alloc_cvar(hist: pd.DataFrame) -> pd.Series:
        h = hist.tail(504)
        try:
            return fp.max_return_cvar_constrained(
                h, alpha=ALPHA, cvar_limit=0.004, sum_to=1.0, gross_limit=1.5
            ).weights
        except ValueError:  # infeasible window: stand down to equal weight
            return pd.Series(1.0 / 3, index=h.columns)

    racers = {"EW": alloc_ew, "MVO": alloc_mvo, "ERC": alloc_erc,
              "CVaR-constr": alloc_cvar}
    race_nets = {}
    for name, fn in racers.items():
        bt = fp.run_backtest(spot_leg, carry_leg, fn, est_window=504,
                             rebalance_every=21, cost_bps=5.0)
        race_nets[name] = bt.ledger["net"]
        s = fp.summary(bt.ledger["net"], ALPHA)
        carry_share = bt.ledger["carry_pnl"].sum() / max(bt.ledger["net"].sum(), 1e-12)
        print(stat_line(name, s)
              + f"  carry P&L share {fmt(carry_share, True, 0):>5}"
              + f"  cost drag {fmt(bt.ledger['cost'].sum() / len(bt.ledger) * 252, True)}/yr")

    # ------------------------------------------------------------------ 8
    print("\n[8] GBP-base reporting (London desk)")
    conv = fp.base_conversion_returns(panel.spots["GBP"], panel.rates,
                                      base="GBP").reindex(dec.total.index)
    total_gbp = fp.convert_base(dec.total, conv)
    ret_usd, w_c = fp.style_returns(dec.total, sig_carry, gross=gross)
    ret_gbp = (w_c * total_gbp).sum(axis=1)
    diff = float((ret_usd - ret_gbp).abs().max())
    print(f"  dollar-neutral carry sleeve: |USD-base - GBP-base| log-return "
          f"diff = {diff:.2e} (exact invariance)")
    ew_net = race_nets["EW"]
    ew_gbp = ew_net + conv.reindex(ew_net.index)  # net-long book shifts by conv
    print(f"  EW multi-style (net-long): SR {fmt(fp.sharpe_ratio(ew_net))} (USD base) "
          f"vs {fmt(fp.sharpe_ratio(ew_gbp))} (GBP base) — net books are NOT "
          "base-invariant; hedge the base or report both")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
