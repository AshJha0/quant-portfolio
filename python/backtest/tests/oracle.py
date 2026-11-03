"""Exact rational oracle for the hand fixtures.

Re-computes a backtest in shares and cash with ``fractions.Fraction``, from
the rules stated in ``qp_bt.core`` and nothing else: no weights, no numpy, no
code shared with the engine. Linear costs only, so the trade equation
``N = m - c sum |w_i N - H_i|`` is solved exactly (guess the trade signs at
``N = m``, solve the linear equation, confirm the signs).
"""

from __future__ import annotations

from fractions import Fraction as F


def solve_linear(m, held, w, c):
    signs = [1 if wi * m - hi > 0 else (-1 if wi * m - hi < 0 else 0) for wi, hi in zip(w, held, strict=True)]
    for _ in range(len(w) + 2):
        n = (m + c * sum(si * hi for si, hi in zip(signs, held, strict=True))) / (
            1 + c * sum(si * wi for si, wi in zip(signs, w, strict=True)))
        new = [1 if wi * n - hi > 0 else (-1 if wi * n - hi < 0 else 0) for wi, hi in zip(w, held, strict=True)]
        if new == signs:
            return n
        signs = new
    raise AssertionError("signs did not settle")


def run(prices, targets, phi, lag, cost_bp, extra_bp, rate, tau, borrow_per_period):
    """prices: list of lists of Fraction; targets: dict decision_bar -> list of Fraction.

    Returns per-bar dicts with gross, cost, borrow, fin, net, nav, turnover.
    """
    n_assets = len(prices[0])
    q = [F(0)] * n_assets
    cash = F(1)
    nav_prev = F(1)
    c = F(cost_bp + extra_bp) / 10_000
    out = []
    for s in range(len(prices)):
        pp = prices[s - 1] if s > 0 else prices[0]
        p = prices[s]
        pf = [a + phi * (b - a) for a, b in zip(pp, p, strict=True)]
        # overnight carry on the opening book
        cash_w = cash / nav_prev
        rf = rate * tau if s > 0 else F(0)
        fin = cash_w * rf * nav_prev if s > 0 else F(0)
        borrow = borrow_per_period * sum(-qi * pi for qi, pi in zip(q, pf, strict=True) if qi < 0) if s > 0 else F(0)
        cost = F(0)
        traded = F(0)
        dec = s - lag
        if dec >= 0 and dec in targets:
            w = targets[dec]
            held = [qi * pi for qi, pi in zip(q, pf, strict=True)]
            m = cash + sum(held)
            n = solve_linear(m, held, w, c)
            q_new = [wi * n / pi for wi, pi in zip(w, pf, strict=True)]
            traded = sum(abs(a - b) * pi for a, b, pi in zip(q_new, q, pf, strict=True))
            cost = c * traded
            cash = cash - sum((a - b) * pi for a, b, pi in zip(q_new, q, pf, strict=True)) - cost
            q = q_new
            if phi == 0 and s > 0:
                # filled at close s-1: the new book is what is held overnight
                fin = cash / nav_prev * rf * nav_prev
                borrow = borrow_per_period * sum(-qi * pi for qi, pi in zip(q, pf, strict=True)
                                                 if qi < 0)
        cash = cash + fin - borrow
        nav = cash + sum(qi * pi for qi, pi in zip(q, p, strict=True))
        net = nav / nav_prev - 1
        out.append({"net": net, "cost": cost / nav_prev, "borrow": borrow / nav_prev,
                    "fin": fin / nav_prev, "gross": net + (cost + borrow - fin) / nav_prev,
                    "turnover": traded / nav_prev, "nav": nav, "q": list(q)})
        nav_prev = nav
    return out
