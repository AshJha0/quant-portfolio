"""Independent FRTB SBM oracle for GIRR and FX delta and vega (test-only).

Written from the text of BCBS d457 / the consolidated Basel Framework
chapter MAR21 alone, without reference to ``qp_core.capital``. It is a
plain, spreadsheet-style second implementation: explicit loops, no numpy,
every number traceable to a paragraph. Its only purpose is to be compared
with ``qp_core.capital.frtb_sbm`` (plan W17, "a second independent
implementation written by a different lane").

Conventions (MAR21.19, MAR21.25): a GIRR delta sensitivity is the PV change
for a 1 bp shift divided by 0.0001, so the risk weights below are rate moves
(0.017 = 1.7 %). Vega sensitivity is implied vol times dV/dvol (MAR21.25).

Paragraph map:
  MAR21.4   weighted sensitivity, bucket aggregation K_b, across-bucket
            aggregation with the S_b fallback when the sum is negative.
  MAR21.6   three correlation scenarios (medium, high, low).
  MAR21.42  GIRR delta risk weights by tenor.
  MAR21.44  sqrt(2) relief for the specified currencies.
  MAR21.46  GIRR same-curve tenor correlation, theta = 3 %, floor 40 %.
  MAR21.47  different curves, same currency: 99.9 % times MAR21.46.
  MAR21.50  GIRR across-currency gamma = 50 %.
  MAR21.87  FX delta risk weight 15 %.
  MAR21.88  sqrt(2) relief for the specified currency pairs.
  MAR21.89  FX across-pair gamma = 60 %.
  MAR21.92  vega risk weight min(55 % * sqrt(LH / 10), 100 %);
            LH = 60 days (GIRR), 40 days (FX); both cap at 100 %.
  MAR21.93  FX vega correlation exp(-1 % |Tk - Tl| / min(Tk, Tl)).
  MAR21.94  GIRR vega correlation: option-maturity factor times
            underlying-maturity factor, each of the MAR21.93 form.
  MAR21.95  vega across-bucket gamma = the delta gamma of the class.
"""

from __future__ import annotations

import math

SCENARIOS = ("medium", "high", "low")

# MAR21.42
GIRR_TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
GIRR_RW = (0.017, 0.017, 0.016, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011)
# MAR21.44 (domestic reporting currency is outside this oracle's scope)
SPECIFIED_CCY = frozenset({"EUR", "USD", "GBP", "AUD", "JPY", "SEK", "CAD"})
# MAR21.88: USD against each of these, plus first-order crosses among them.
SPECIFIED_FX = frozenset(
    "USD EUR JPY GBP AUD CAD CHF MXN CNY NZD RUB HKD SGD TRY KRW SEK ZAR INR NOK BRL".split()
)
FX_RW = 0.15


def scen(rho: float, s: str) -> float:
    """MAR21.6: rescale one correlation parameter for a scenario."""
    if s == "medium":
        return rho
    if s == "high":
        return min(1.25 * rho, 1.0)
    if s == "low":
        return max(2.0 * rho - 1.0, 0.75 * rho)
    raise ValueError(s)


def k_bucket(ws: list[float], rho: list[list[float]]) -> float:
    """MAR21.4(4): K_b = sqrt(max(0, sum_k sum_l rho_kl WS_k WS_l)), rho_kk = 1."""
    tot = 0.0
    n = len(ws)
    for i in range(n):
        tot += ws[i] * ws[i]
        for j in range(n):
            if i != j:
                tot += rho[i][j] * ws[i] * ws[j]
    return math.sqrt(max(0.0, tot))


def across(kb: list[float], sb: list[float], gamma: float) -> float:
    """MAR21.4(5) with a single gamma for every bucket pair."""

    def total(s: list[float]) -> float:
        t = sum(k * k for k in kb)
        for b in range(len(kb)):
            for c in range(len(kb)):
                if b != c:
                    t += gamma * s[b] * s[c]
        return t

    t = total(sb)
    if t < 0.0:
        alt = [max(min(s, k), -k) for s, k in zip(sb, kb, strict=True)]
        t = total(alt)
    return math.sqrt(max(0.0, t))


def _charge(buckets: dict[str, tuple[list[float], list[list[float]]]], gamma: float, s: str) -> float:
    kb, sb = [], []
    for name in sorted(buckets):
        ws, rho = buckets[name]
        kb.append(k_bucket(ws, [[scen(r, s) for r in row] for row in rho]))
        sb.append(sum(ws))
    return across(kb, sb, scen(gamma, s))


def _all(buckets: dict, gamma: float) -> dict[str, float]:
    out = {s: _charge(buckets, gamma, s) for s in SCENARIOS}
    out["charge"] = max(out[s] for s in SCENARIOS)
    return out


def _tenor_corr(a: float, b: float, theta: float) -> float:
    return math.exp(-theta * abs(a - b) / min(a, b))


def _net(rows: list[tuple]) -> dict[tuple, float]:
    """Sum sensitivities that share a risk factor (MAR21.4(1)-(2))."""
    acc: dict[tuple, float] = {}
    for *key, v in rows:
        acc[tuple(key)] = acc.get(tuple(key), 0.0) + v
    return acc


def girr_delta(rows: list[tuple[str, str, float, float]], relief: bool = True) -> dict[str, float]:
    """rows: (currency, curve, tenor, sensitivity)."""
    net = _net(rows)
    buckets: dict[str, tuple[list, list]] = {}
    for ccy in {k[0] for k in net}:
        keys = sorted(k for k in net if k[0] == ccy)
        ws = []
        for k in keys:
            rw = GIRR_RW[GIRR_TENORS.index(k[2])]
            if relief and ccy in SPECIFIED_CCY:
                rw /= math.sqrt(2.0)  # MAR21.44
            ws.append(rw * net[k])  # MAR21.4(3)
        rho = []
        for a in keys:
            row = []
            for b in keys:
                r = max(_tenor_corr(a[2], b[2], 0.03), 0.40)  # MAR21.46
                if a[1] != b[1]:
                    r *= 0.999  # MAR21.47
                row.append(r)
            rho.append(row)
        buckets[ccy] = (ws, rho)
    return _all(buckets, 0.50)  # MAR21.50


def girr_vega(rows: list[tuple[str, float, float, float]]) -> dict[str, float]:
    """rows: (currency, option_tenor, underlying_tenor, sensitivity)."""
    rw = min(0.55 * math.sqrt(60 / 10), 1.0)  # MAR21.92
    net = _net(rows)
    buckets = {}
    for ccy in {k[0] for k in net}:
        keys = sorted(k for k in net if k[0] == ccy)
        ws = [rw * net[k] for k in keys]
        rho = [
            [
                min(_tenor_corr(a[1], b[1], 0.01) * _tenor_corr(a[2], b[2], 0.01), 1.0)  # MAR21.94
                for b in keys
            ]
            for a in keys
        ]
        buckets[ccy] = (ws, rho)
    return _all(buckets, 0.50)  # MAR21.95 -> MAR21.50


def _fx_pair_specified(pair: str) -> bool:
    p = pair.replace("/", "").upper()
    return len(p) == 6 and p[:3] in SPECIFIED_FX and p[3:] in SPECIFIED_FX


def fx_delta(rows: list[tuple[str, float]], relief: bool = True) -> dict[str, float]:
    """rows: (pair, sensitivity). One risk factor per bucket, so K_b = |WS|."""
    net = _net(rows)
    buckets = {}
    for (pair,), v in net.items():
        rw = FX_RW / math.sqrt(2.0) if relief and _fx_pair_specified(pair) else FX_RW
        buckets[pair] = ([rw * v], [[1.0]])
    return _all(buckets, 0.60)  # MAR21.89


def fx_vega(rows: list[tuple[str, float, float]]) -> dict[str, float]:
    """rows: (pair, option_tenor, sensitivity)."""
    rw = min(0.55 * math.sqrt(40 / 10), 1.0)  # MAR21.92
    net = _net(rows)
    buckets = {}
    for pair in {k[0] for k in net}:
        keys = sorted(k for k in net if k[0] == pair)
        ws = [rw * net[k] for k in keys]
        rho = [[_tenor_corr(a[1], b[1], 0.01) for b in keys] for a in keys]  # MAR21.93
        buckets[pair] = (ws, rho)
    return _all(buckets, 0.60)  # MAR21.95 -> MAR21.89
