"""Write the Python reference values the Rust and C++ risk-kernel ports must reproduce.

Run from the repository root (root .venv, PYTHONPATH with qp_core, fi_rates and fx_var):

    python python/equity/04-fixed-income/tools/gen_risk_parity.py

Output: ``rust/core/tests/data/risk_parity.txt``, shared by
``rust/core/tests/risk_parity.rs`` and ``cpp/core/tests/test_risk.cpp``.
Floats are written with ``repr`` so they round-trip exactly.

What is frozen and what is recomputed
-------------------------------------
The curve case uses the real ``fi_rates`` objects: seven quotes (deposit,
FRA, five par swaps) bootstrapped with ``loglinear_df`` and a cash-flow
book with flows between pillars (the book of
``tests/test_risk_par_explain.py``). The finite-difference inputs (quote
Jacobian ``J``, node risk ``g_z``, node Hessian ``Gamma_z`` and the Hessians
``H_i`` of each model quote) are frozen in the file; the ports recompute
from them the outputs that are pure algebra: quote risk ``J^-T g_z``, par
DV01 per bp, and the quote gamma ``J^-T (Gamma_z - sum_i g_q,i H_i) J^-1``.
Freezing the finite differences is deliberate: a central difference with a
1e-6 step divides rounding noise by 1e-6, so two languages that differ in
the last bit of ``exp`` cannot agree to 1e-12 on a re-derived Jacobian.

Lines (tag first):
  ``N n``                    curve size
  ``J`` / ``GZ`` / ``GAMZ``  row-major J, node risk, node Hessian
  ``H i``                    row-major Hessian of model quote i
  ``GQ`` / ``DV01`` / ``GAMQ`` expected quote risk, DV01 per bp, quote gamma
  ``GRID d`` + ``AX`` lines + ``V``  a d-axis vol grid (row-major vols)
  ``PT c... n``              a query point and its notional in the test PV
  ``W`` / ``VOL``            weights and vol at the preceding point
  ``VEGA``                   bucket vegas of PV = sum notional * vol(pt)^2
  ``TA`` + ``D``/``M``/``G``/``A`` Taylor attribution case
  ``EX hpl new_trades`` + ``T`` time items + ``R rtpl unexplained total``
  ``SM s0 T r q k0 a b c``   a smile ``a + b x + c x x`` with ``x = K - k0``
  ``SD K S call dyn price delta gamma up dn`` one quote under the preceding
                             smile (dyn 0 strike, 1 moneyness, 2 delta); ``up``
                             and ``dn`` are the deltas ``smile_gamma`` differences
  ``RK K call`` + ``S``/``P``/``TT``/``RR``  a realised path and its RMS
                             unexplained P&L per dynamic, preceding smile
  ``PA S K T r_d r_f sigma call want``  premium-adjusted delta
  ``RC conv v0 x0`` + ``V1``/``X1``/``PNL``  reporting-currency conversion
                             (conv 0 scenario, 1 spot) of a real ``fx_var``
                             book, values frozen, P&L from ``reporting_pnl``
  ``TB`` + ``TW expiry w...``  tenor buckets and the split of expiries over them

The smile is a polynomial written with ``+`` and ``*`` only, so every language
evaluates it bit for bit. ``smile_gamma`` is a difference of a difference, and
a smile that went through ``exp`` or ``log`` would put last-bit noise through
both.

``smile_gamma`` cannot meet 1e-12 relative to itself across languages, for the
reason above: its step is ``1e-4 S``, so one last bit of delta (1e-16) moves a
gamma of 0.004 by about 1e-12 of itself. Python's ``erfc`` and the ports' differ
in that last bit. So ``up`` and ``dn`` are frozen, the ports match them to 1e-12,
and they check gamma to the error those two deltas allow.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from fi_rates.bootstrap import FRA, Deposit, ParSwap, bootstrap_curve
from fi_rates.risk import CashflowBook, par_rate_gamma, quote_jacobian
from fi_rates.risk.par import _ordered, _vector_hessian, model_quotes, node_risk
from fx_var.book import Book, Cash, Forward, Market, Option, Spot
from fx_var.reporting import _fx_level, reporting_pnl, tenor_weights
from qp_core.risk import (
    Smile,
    SmileDynamics,
    VolGrid,
    explain,
    node_to_quote_risk,
    premium_adjusted_delta,
    rank_smile_dynamics,
    smile_delta,
    smile_gamma,
    smile_price,
    taylor_attribution,
    vega_grid,
)

OUT = pathlib.Path("rust/core/tests/data/risk_parity.txt")


def _row(tag: str, values) -> str:
    return " ".join([tag, *(repr(float(v)) for v in np.ravel(values))])


def _curve_case(lines: list[str]) -> None:
    quotes = [
        Deposit(0.5, 0.040), FRA(0.5, 1.0, 0.041), ParSwap(2.0, 0.042),
        ParSwap(3.0, 0.0415), ParSwap(5.0, 0.040), ParSwap(7.0, 0.0405),
        ParSwap(10.0, 0.041),
    ]
    t = [1, 2, 3, 4, 5, 6, 0.75, 1.75, 2.75, 3.5, 9.25]
    a = [4.5] * 5 + [104.5] + [-30.0] * 4 + [80.0]
    book = CashflowBook(tuple(float(x) for x in t), tuple(a))
    curve = bootstrap_curve(quotes, "loglinear_df")

    def pv(c):
        return book.pv(c, 0.0)

    ordered = _ordered(quotes, curve)
    jac = quote_jacobian(ordered, curve)
    g_z = node_risk(pv, curve)
    g_q = node_to_quote_risk(g_z, jac)
    hess = _vector_hessian(
        lambda c: np.concatenate(([pv(c)], model_quotes(ordered, c))), curve, 1e-4
    )
    gam_q = par_rate_gamma(pv, quotes, curve, jacobian=jac, quote_risk=g_q)
    inv = np.linalg.inv(jac)
    check = inv.T @ (hess[0] - np.einsum("i,ijk->jk", g_q, hess[1:])) @ inv
    assert np.array_equal(check, gam_q), "frozen Hessians differ from par_rate_gamma's"
    n = jac.shape[0]
    lines.append(f"N {n}")
    lines.append(_row("J", jac))
    lines.append(_row("GZ", g_z))
    lines.append(_row("GAMZ", hess[0]))
    for i in range(n):
        lines.append(_row(f"H {i}", hess[1 + i]).replace(f"H {i}", "H", 1))
    lines.append(_row("GQ", g_q))
    lines.append(_row("DV01", g_q * 1e-4))
    lines.append(_row("GAMQ", gam_q))


def _grid_case(lines: list[str], axes, points) -> None:
    rng = np.random.default_rng(len(axes) * 7 + len(points))
    shape = tuple(len(x) for x in axes)
    grid = VolGrid(tuple(np.asarray(x, float) for x in axes), 0.1 + 0.2 * rng.random(shape))
    lines.append(f"GRID {len(axes)}")
    for x in axes:
        lines.append(_row("AX", x))
    lines.append(_row("V", grid.vols))
    for pt, notional in points:
        lines.append(_row("PT", [*pt, notional]))
        lines.append(_row("W", grid.weights(*pt)))
        lines.append(_row("VOL", [grid.vol(*pt)]))

    def pv(g: VolGrid) -> float:
        return float(sum(n * g.vol(*p) ** 2 for p, n in points))

    lines.append(_row("VEGA", vega_grid(pv, grid)))


DYNAMICS = (SmileDynamics.STICKY_STRIKE, SmileDynamics.STICKY_MONEYNESS,
            SmileDynamics.STICKY_DELTA)


def _smile_case(lines: list[str], s0, expiry, r, q, k0, a, b, c, strikes, spots, path) -> None:
    def vol0(k: float) -> float:
        x = k - k0
        return a + b * x + c * x * x

    smile = Smile(s0, expiry, r, q, vol0)
    lines.append(_row("SM", [s0, expiry, r, q, k0, a, b, c]))
    for k in strikes:
        for s in spots:
            for call in (True, False):
                for i, d in enumerate(DYNAMICS):
                    h = 1e-4 * s
                    lines.append(_row("SD", [
                        k, s, int(call), i,
                        smile_price(smile, k, s, call, smile_dynamics=d),
                        smile_delta(smile, k, s, call, smile_dynamics=d),
                        smile_gamma(smile, k, s, call, smile_dynamics=d),
                        smile_delta(smile, k, s + h, call, smile_dynamics=d),
                        smile_delta(smile, k, s - h, call, smile_dynamics=d),
                    ]))
    # A short path priced under sticky moneyness plus noise, one day a step.
    k, call, seed = path
    rng = np.random.default_rng(seed)
    n = 12
    t = np.arange(n) / 252.0
    sp = s0 * np.exp(np.cumsum(rng.normal(0.0, 0.006, n)))
    p = []
    for si, ti in zip(sp, t, strict=True):
        local = Smile(s0, expiry - float(ti), r, q, vol0)
        p.append(smile_price(local, k, float(si), call,
                             smile_dynamics=SmileDynamics.STICKY_MONEYNESS))
    p = np.array(p) + rng.normal(0.0, 1e-3 * s0 / 100.0, n)
    ranked = rank_smile_dynamics(sp, p, t, smile, k, call)
    lines.append(_row("RK", [k, int(call)]))
    lines.append(_row("S", sp))
    lines.append(_row("P", p))
    lines.append(_row("TT", t))
    lines.append(_row("RR", [ranked[d] for d in DYNAMICS]))


def _premium_cases(lines: list[str]) -> None:
    for s, k, t, rd, rf, v in [(1.10, 1.12, 0.5, 0.05, 0.03, 0.08),
                               (1.10, 1.0, 2.0, 0.01, 0.04, 0.12),
                               (150.0, 155.0, 0.25, 0.001, 0.05, 0.11)]:
        for call in (True, False):
            lines.append(_row("PA", [s, k, t, rd, rf, v, int(call),
                                     premium_adjusted_delta(s, k, t, rd, rf, v, call)]))


def _reporting_cases(lines: list[str]) -> None:
    market = Market(
        spot_usd={"EUR": 1.10, "GBP": 1.27, "JPY": 1 / 150.0},
        rates={"USD": 0.05, "EUR": 0.03, "GBP": 0.045, "JPY": 0.001},
        vols={"EURUSD": 0.08, "GBPJPY": 0.11},
    )
    book = Book([
        Cash("EUR", 2e6), Spot("GBPUSD", -1.5e6), Spot("EURJPY", 1e6),
        Forward("GBPJPY", 5e5, 0.75), Option("EURUSD", 3e6, 1.12, 0.4, "call"),
        Option("GBPJPY", -1e6, 190.0, 1.5, "put"),
    ])
    rng = np.random.default_rng(20261104)
    cols = ["FX:EUR", "FX:GBP", "FX:JPY", "IR:USD", "VOL:EURUSD"]
    scale = np.array([0.006, 0.007, 0.008, 5e-4, 0.004])
    sc = pd.DataFrame(rng.standard_t(5, size=(8, len(cols))) * scale, columns=cols)
    sh = Book._normalise_shocks(sc)
    v0, v1 = book.value_usd(market, None), book.value_usd(market, sh)
    for ccy, conv in [("EUR", "scenario"), ("EUR", "spot"), ("JPY", "scenario"),
                      ("USD", "scenario")]:
        x1 = np.broadcast_to(_fx_level(market, ccy, sh), v1.shape)
        lines.append(_row("RC", [int(conv == "spot"), v0, _fx_level(market, ccy, None)]))
        lines.append(_row("V1", v1))
        lines.append(_row("X1", x1))
        lines.append(_row("PNL", reporting_pnl(book, market, sc, reporting_ccy=ccy,
                                               conversion=conv)))


def _tenor_cases(lines: list[str]) -> None:
    buckets = (0.25, 0.5, 1.0, 2.0, 5.0)
    lines.append(_row("TB", buckets))
    for e in (0.1, 0.25, 0.3, 0.5, 0.75, 1.9, 5.0, 7.0):
        lines.append(_row("TW", [e, *tenor_weights(e, buckets)]))


def main() -> None:
    lines = ["# generated by python/equity/04-fixed-income/tools/gen_risk_parity.py"]
    _curve_case(lines)
    _grid_case(
        lines,
        [[0.25, 0.5, 1.0, 2.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        [((0.1, 70.0), 3.0), ((0.75, 95.0), -2.0), ((1.0, 100.0), 5.0),
         ((1.5, 117.5), 1.5), ((3.0, 130.0), -4.0), ((0.3, 120.0), 2.5)],
    )
    _grid_case(
        lines,
        [[0.5, 1.0, 5.0], [1.0, 2.0, 10.0], [-0.01, 0.0, 0.01]],
        [((0.7, 1.5, 0.003), 4.0), ((5.0, 10.0, 0.02), -1.0), ((0.1, 3.0, -0.005), 2.0),
         ((2.0, 2.0, 0.0), 3.0)],
    )
    rng = np.random.default_rng(20261103)
    for n, with_gamma in [(5, True), (3, False), (8, True)]:
        d = rng.normal(0, 100, n)
        m = rng.normal(0, 0.01, n)
        g = rng.normal(0, 50, (n, n)) if with_gamma else None
        lines.append(f"TA {n} {int(with_gamma)}")
        lines.append(_row("D", d))
        lines.append(_row("M", m))
        if g is not None:
            lines.append(_row("G", g))
        lines.append(_row("A", taylor_attribution(d, m, g)))
        hpl = float(np.sum(taylor_attribution(d, m, g)) + rng.normal(0, 0.1))
        time = {"carry": float(rng.normal(0, 1)), "roll": float(rng.normal(0, 1))}
        new = float(rng.normal(0, 1))
        ex = explain(hpl=hpl, time=time, delta=d, move=m, labels=[f"f{i}" for i in range(n)],
                     gamma=g, new_trades=new)
        lines.append(_row("EX", [hpl, new]))
        lines.append(_row("T", list(time.values())))
        lines.append(_row("R", [ex.rtpl, ex.unexplained, ex.total]))
    _smile_case(lines, 100.0, 1.0, 0.0, 0.0, 100.0, 0.20, -0.001, 0.0,
                [100.0], [100.0], (100.0, True, 11))
    _smile_case(lines, 100.0, 1.5, 0.04, 0.015, 100.0, 0.22, -0.0012, 2e-5,
                [80.0, 100.0, 125.0], [100.0, 97.0, 104.0], (105.0, False, 12))
    _smile_case(lines, 1.10, 0.75, 0.05, 0.03, 1.10, 0.08, -0.05, 0.4,
                [1.0, 1.12, 1.25], [1.10, 1.14], (1.12, True, 13))
    _premium_cases(lines)
    _reporting_cases(lines)
    _tenor_cases(lines)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
