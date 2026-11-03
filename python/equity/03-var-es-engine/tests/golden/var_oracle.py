#!/usr/bin/env python3
"""Independent VaR / ES / backtest oracle (mpmath and exact arithmetic only).

Why this file exists
--------------------
``generate_golden.py`` records what ``eq_var`` / ``fx_var`` return, so the
C++ and Rust engines that match it are checked against the Python engine,
not against the mathematics. This oracle computes the same quantities from
their definitions, with no code shared with any engine:

* it imports only the standard library and mpmath (no ``eq_var``,
  ``fx_var``, numpy or scipy; ``test_oracle_independence`` enforces this);
* closed forms are evaluated at 50 significant digits and rounded to the
  nearest double once, at the end;
* order statistics and the Acerbi-Tasche ES of a sample are exact rational
  arithmetic (``fractions.Fraction``) on the sample's decimal values;
* the Student-t quantile is found by bisection on the t CDF written as a
  regularized incomplete beta function, not by any library ppf.

Conventions (from the methodology spec, restated here from first principles)
-------------------------------------------------------------------------
* P&L is positive for a gain. VaR and ES are reported as positive losses.
* Normal: ``VaR = -(mu + sigma z_a)``, ``ES = sigma phi(z_a) / a - mu``,
  ``z_a = Phi^{-1}(a)``.
* Student-t is variance matched: the standardised t with ``nu`` degrees of
  freedom is scaled by ``s = sqrt((nu - 2) / nu)`` so its variance is 1.
  ``VaR = -(mu + sigma s q)``, ``ES = sigma s f_nu(q) (nu + q^2) /
  ((nu - 1) a) - mu`` with ``q = t_nu^{-1}(a)`` (McNeil, Frey and Embrechts
  2015, ch. 2, example 2.15).
* Cornish-Fisher with skew ``S`` and excess kurtosis ``K``:
  ``z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36``,
  ``VaR = -(mu + sigma z_cf)``. Emitted only where the expansion is
  monotone on ``[-z, z]`` (checked here by exact minimisation of the
  quadratic-in-z^2 derivative).
* Historical, regulatory rule (BCBS MAR33.4 / MAR99: the 3rd worst loss of
  250 at 99 %): losses ``L = -P`` sorted descending, ``k = n a``,
  ``m = ceil(k)`` clamped to ``[1, n]``, ``VaR = L(m)``. The oracle takes
  the ceiling of the exact rational ``k``; the engines use
  ``ceil(k - 1e-9)`` in floating point, which agrees whenever ``n a`` is
  representable to better than 1e-9 (true for every case here).
* Historical ES (Acerbi and Tasche 2002): the integral of the empirical
  quantile over the worst ``a`` of mass,
  ``ES = [sum_{i<m} L(i) + (k - (m - 1)) L(m)] / k``.
* Historical, interpolated rule (for reference; the engines do not expose
  it yet): the linear interpolation of the sorted losses at position
  ``h = (n - 1) a`` from the worst (Hyndman and Fan 1996 type 7 on L).
* Kupiec (1995) POF: ``LR = -2 ln[(1-a)^(n-x) a^x] + 2 ln[(1-p)^(n-x) p^x]``,
  ``p = x / n``, ``0 ln 0 = 0``; p-value ``P(chi2_1 > LR) = erfc(sqrt(LR/2))``.
* Christoffersen (1998): first-order Markov LR_ind, chi2(1); LR_cc =
  LR_uc + LR_ind, chi2(2) with p-value ``exp(-LR/2)``.
* Basel traffic light (BCBS 1996): ``F(x) = P(X <= x)``,
  ``X ~ Binomial(n, a)``, summed exactly in rationals.

Usage::

    python tests/golden/var_oracle.py            # writes both projects' JSON
    python tests/golden/var_oracle.py --check    # regenerate and compare
"""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50

HERE = Path(__file__).resolve().parent
EQ_OUT = HERE / "oracle_vectors.json"
FX_OUT = HERE.parents[3] / "fx" / "03-var-es-engine" / "tests" / "golden" / "oracle_vectors.json"

ALPHAS = ["0.001", "0.01", "0.025", "0.05"]
DFS = ["3", "4", "6", "10", "30"]
# (skew, excess kurtosis, alpha) for Cornish-Fisher; only valid triples are emitted.
CF_CASES = [("-0.5", "1", "0.01"), ("0.122", "-0.427", "0.01"), ("0.2", "0.5", "0.025"),
            ("-0.3", "2", "0.05"), ("0", "0", "0.01")]
HIST_ALPHAS = ["0.004", "0.01", "0.025", "0.05", "0.1"]
KUPIEC_CASES = [(250, x, "0.01") for x in range(0, 13)] + [
    (500, 9, "0.01"), (1000, 15, "0.01"), (250, 12, "0.025"), (250, 5, "0.05"), (1000, 50, "0.05")]
BASEL_CASES = [(250, x, "0.01") for x in range(0, 11)] + [
    (500, 8, "0.01"), (500, 12, "0.01"), (1000, 15, "0.01"), (250, 12, "0.025")]

# Odd sigma and mean so that sign and scale conventions are exercised.
SIGMA_MEAN = ("12345.678", "-37.5")


def f(x: mp.mpf) -> float:
    """Round an mpf to the nearest double (once, at the end)."""
    return float(x)


def tag(s: str) -> str:
    """'0.025' -> 'a0025' style key fragment; '-0.5' -> 'm05'."""
    return s.replace("-", "m").replace(".", "")


# ------------------------------------------------------------ distributions
def norm_ppf(a: mp.mpf) -> mp.mpf:
    return -mp.sqrt(2) * mp.erfinv(1 - 2 * a)


def norm_pdf(z: mp.mpf) -> mp.mpf:
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def t_cdf(x: mp.mpf, nu: mp.mpf) -> mp.mpf:
    """Student-t CDF through the regularized incomplete beta function."""
    tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + x * x), regularized=True) / 2
    return tail if x < 0 else 1 - tail


def t_pdf(x: mp.mpf, nu: mp.mpf) -> mp.mpf:
    return (mp.gamma((nu + 1) / 2) / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / 2))
            * (1 + x * x / nu) ** (-(nu + 1) / 2))


def t_ppf_lower(a: mp.mpf, nu: mp.mpf) -> mp.mpf:
    """Lower-tail t quantile (a < 0.5) by bisection on t_cdf to 1e-45."""
    lo, hi = mp.mpf(-1e6), mp.mpf(0)
    for _ in range(400):
        mid = (lo + hi) / 2
        if t_cdf(mid, nu) < a:
            lo = mid
        else:
            hi = mid
        if hi - lo < mp.mpf("1e-45"):
            break
    return (lo + hi) / 2


# --------------------------------------------------------------- parametric
def normal_var_es(sigma, mean, a):
    z = norm_ppf(a)
    return -(mean + sigma * z), sigma * norm_pdf(z) / a - mean


def t_var_es(sigma, mean, a, nu):
    q = t_ppf_lower(a, nu)
    s = mp.sqrt((nu - 2) / nu)
    var = -(mean + sigma * s * q)
    es = sigma * s * t_pdf(q, nu) * (nu + q * q) / ((nu - 1) * a) - mean
    return var, es


def cf_z(z, skew, ek):
    return (z + (z * z - 1) * skew / 6 + (z ** 3 - 3 * z) * ek / 24
            - (2 * z ** 3 - 5 * z) * skew * skew / 36)


def cf_valid(skew, ek, a) -> bool:
    """dz_cf/dz > 0 on [-z_a, z_a]. The derivative is A z^2 + B z + C."""
    za = -norm_ppf(a)
    aq = ek / 8 - skew * skew / 6
    bq = skew / 3
    cq = 1 - ek / 8 + 5 * skew * skew / 36
    pts = [-za, za]
    if aq != 0:
        v = -bq / (2 * aq)
        if -za < v < za:
            pts.append(v)
    return min(aq * p * p + bq * p + cq for p in pts) > 0


def portfolio_sigma(w, cov):
    n = len(w)
    return mp.sqrt(mp.fsum(w[i] * cov[i][j] * w[j] for i in range(n) for j in range(n)))


# --------------------------------------------------------------- historical
def round2(x: float) -> float:
    return math.floor(x * 100.0 + 0.5) / 100.0


def conv_sample() -> list[float]:
    """Spec S13 sample: P_i = round2(1000 sin(0.37 i + 0.1)(1 + 0.5 cos(0.11 i))).

    The inputs are cents, so libm differences in sin/cos (an ulp or two)
    cannot move a value unless it sits within an ulp of a half cent; the
    test files rebuild the sample in each language and must agree exactly.
    """
    return [round2(1000.0 * math.sin(0.37 * i + 0.1) * (1.0 + 0.5 * math.cos(0.11 * i)))
            for i in range(250)]


def hist_kth_worst(pnl: list[float], a: Fraction) -> tuple[Fraction, Fraction, int]:
    """Regulatory k-th worst VaR and Acerbi-Tasche ES, exact rationals."""
    losses = sorted((-Fraction(x) for x in pnl), reverse=True)
    n = len(losses)
    k = n * a
    m = min(max(math.ceil(k), 1), n)  # k is an exact rational here
    var = losses[m - 1]
    es = (sum(losses[: m - 1], Fraction(0)) + (k - (m - 1)) * losses[m - 1]) / k
    return var, es, m


def hist_interpolated(pnl: list[float], a: Fraction) -> Fraction:
    """Type-7 linear interpolation of the worst-first sorted losses at (n-1) a."""
    losses = sorted((-Fraction(x) for x in pnl), reverse=True)
    h = (len(losses) - 1) * a
    lo = math.floor(h)
    frac = h - lo
    hi = min(lo + 1, len(losses) - 1)
    return losses[lo] + frac * (losses[hi] - losses[lo])


# ---------------------------------------------------------------- backtests
def xlogy(x, y):
    return x * mp.log(y) if x > 0 else mp.mpf(0)


def kupiec(n: int, x: int, a: mp.mpf):
    p = mp.mpf(x) / n
    ll0 = xlogy(n - x, 1 - a) + xlogy(x, a)
    ll1 = xlogy(n - x, 1 - p) + xlogy(x, p)
    lr = max(-2 * (ll0 - ll1), mp.mpf(0))
    return lr, mp.erfc(mp.sqrt(lr / 2))


def christoffersen(ex: list[int], a: mp.mpf):
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(ex[:-1], ex[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0:
            n01 += 1
        elif cur == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = mp.mpf(n01) / (n00 + n01) if n00 + n01 else mp.mpf(0)
    pi11 = mp.mpf(n11) / (n10 + n11) if n10 + n11 else mp.mpf(0)
    pi = mp.mpf(n01 + n11) / (n00 + n01 + n10 + n11)

    def xl(c, y):
        return c * mp.log(y) if c > 0 and y > 0 else mp.mpf(0)

    ll_markov = xl(n00, 1 - pi01) + xl(n01, pi01) + xl(n10, 1 - pi11) + xl(n11, pi11)
    ll_iid = xl(n00 + n10, 1 - pi) + xl(n01 + n11, pi)
    lr_ind = max(-2 * (ll_iid - ll_markov), mp.mpf(0))
    lr_uc, _ = kupiec(len(ex), sum(ex), a)
    lr_cc = lr_uc + lr_ind
    return lr_ind, mp.erfc(mp.sqrt(lr_ind / 2)), lr_cc, mp.exp(-lr_cc / 2)


def binom_cdf(n: int, x: int, a: Fraction) -> Fraction:
    return sum((Fraction(math.comb(n, i)) * a ** i * (1 - a) ** (n - i) for i in range(x + 1)),
               Fraction(0))


def exceptions_eq() -> list[int]:
    """Equity case C pattern: t % 40 == 0 plus days 100 and 101 (9 in 250)."""
    return [1 if (t % 40 == 0 or t in (100, 101)) else 0 for t in range(250)]


def exceptions_fx() -> list[int]:
    """FX case C pattern: t % 37 == 5 plus days 100 and 101 (9 in 250)."""
    return [1 if (t % 37 == 5 or t in (100, 101)) else 0 for t in range(250)]


# Published tables the oracle must reproduce (checked by the tests).
# BCBS (1996), "Supervisory framework for the use of backtesting", table 1:
# cumulative probability of x or fewer exceptions in 250 days at 99 %
# coverage, in percent to two decimals.
BCBS_1996_CUM_PCT = {0: 8.11, 1: 28.58, 2: 54.32, 3: 75.81, 4: 89.22, 5: 95.88, 6: 98.63,
                     7: 99.60, 8: 99.89, 9: 99.97, 10: 99.99}
# Kupiec (1995), table 3 as reproduced by Jorion (Value at Risk, 3rd ed.,
# table 6-3): non-rejection region of the POF test at the 5 % level,
# (lowest accepted N, highest accepted N) for T = 255 / 510 / 1000.
KUPIEC_1995_REGIONS = {
    ("0.01", 255): (0, 6), ("0.01", 510): (2, 10), ("0.01", 1000): (5, 16),
    ("0.025", 255): (3, 11), ("0.025", 510): (7, 20), ("0.025", 1000): (16, 35),
    ("0.05", 255): (7, 19), ("0.05", 510): (17, 35), ("0.05", 1000): (38, 64),
}
CHI2_1_95 = mp.mpf("3.841458820694124032")  # chi2(1) 95 % quantile


# ------------------------------------------------------------------- cases
EQ_CASE_B = {"exposures": ["1000000", "-500000", "200000"], "vols": ["0.01", "0.015", "0.02"],
             "corr": [["1", "0.5", "0.25"], ["0.5", "1", "0.3"], ["0.25", "0.3", "1"]]}
FX_CASE_B = {"exposures": ["11000000", "-4500000", "-2400000"],
             "cov": [["3.6e-5", "1.1e-5", "-2e-6"], ["1.1e-5", "4.9e-5", "-1e-6"],
                     ["-2e-6", "-1e-6", "2.5e-7"]]}


def case_b_cov(project: str):
    if project == "equity":
        v = [mp.mpf(x) for x in EQ_CASE_B["vols"]]
        c = [[mp.mpf(x) for x in row] for row in EQ_CASE_B["corr"]]
        return ([mp.mpf(x) for x in EQ_CASE_B["exposures"]],
                [[v[i] * c[i][j] * v[j] for j in range(3)] for i in range(3)])
    return ([mp.mpf(x) for x in FX_CASE_B["exposures"]],
            [[mp.mpf(x) for x in row] for row in FX_CASE_B["cov"]])


def build(project: str) -> dict[str, object]:
    vals: dict[str, float] = {}
    # 1. Normal and Student-t VaR / ES, unit sigma and an odd sigma / mean.
    sig_odd, mean_odd = (mp.mpf(x) for x in SIGMA_MEAN)
    for a_s in ALPHAS:
        a = mp.mpf(a_s)
        v, e = normal_var_es(mp.mpf(1), mp.mpf(0), a)
        vals[f"normal_var_unit_a{tag(a_s)}"], vals[f"normal_es_unit_a{tag(a_s)}"] = f(v), f(e)
        v, e = normal_var_es(sig_odd, mean_odd, a)
        vals[f"normal_var_odd_a{tag(a_s)}"], vals[f"normal_es_odd_a{tag(a_s)}"] = f(v), f(e)
        for nu_s in DFS:
            nu = mp.mpf(nu_s)
            v, e = t_var_es(mp.mpf(1), mp.mpf(0), a, nu)
            vals[f"t{nu_s}_var_unit_a{tag(a_s)}"], vals[f"t{nu_s}_es_unit_a{tag(a_s)}"] = f(v), f(e)
            v, e = t_var_es(sig_odd, mean_odd, a, nu)
            vals[f"t{nu_s}_var_odd_a{tag(a_s)}"], vals[f"t{nu_s}_es_odd_a{tag(a_s)}"] = f(v), f(e)
    # 2. Cornish-Fisher (odd sigma and mean).
    for s_s, k_s, a_s in CF_CASES:
        s, k, a = mp.mpf(s_s), mp.mpf(k_s), mp.mpf(a_s)
        key = f"cf_s{tag(s_s)}_k{tag(k_s)}_a{tag(a_s)}"
        vals[f"{key}_valid"] = 1.0 if cf_valid(s, k, a) else 0.0
        if vals[f"{key}_valid"]:
            vals[f"{key}_var_odd"] = f(-(mean_odd + sig_odd * cf_z(norm_ppf(a), s, k)))
    # 3. Case B portfolio sigma, parametric VaR and the MC target.
    w, cov = case_b_cov(project)
    sig_b = portfolio_sigma(w, cov)
    vals["b_sigma"] = f(sig_b)
    for a_s in ("0.01", "0.025"):
        a = mp.mpf(a_s)
        vals[f"b_normal_var_a{tag(a_s)}"] = f(normal_var_es(sig_b, mp.mpf(0), a)[0])
        vals[f"b_normal_var_10d_a{tag(a_s)}"] = f(normal_var_es(sig_b * mp.sqrt(10), 0, a)[0])
        vals[f"b_normal_es_a{tag(a_s)}"] = f(normal_var_es(sig_b, mp.mpf(0), a)[1])
        vals[f"b_t5_var_a{tag(a_s)}"] = f(t_var_es(sig_b, mp.mpf(0), a, mp.mpf(5))[0])
        vals[f"b_t5_es_a{tag(a_s)}"] = f(t_var_es(sig_b, mp.mpf(0), a, mp.mpf(5))[1])
    # 4. Historical, both rules, on the spec S13 sample.
    p = conv_sample()
    for a_s in HIST_ALPHAS:
        var, es, m = hist_kth_worst(p, Fraction(a_s))
        vals[f"hist_kth_var_a{tag(a_s)}"] = float(var)
        vals[f"hist_kth_es_a{tag(a_s)}"] = float(es)
        vals[f"hist_kth_rank_a{tag(a_s)}"] = float(m)
        vals[f"hist_interp_var_a{tag(a_s)}"] = float(hist_interpolated(p, Fraction(a_s)))
    # 5. Kupiec, Christoffersen, Basel.
    for n, x, a_s in KUPIEC_CASES:
        lr, pv = kupiec(n, x, mp.mpf(a_s))
        vals[f"kupiec_lr_n{n}_x{x}_a{tag(a_s)}"], vals[f"kupiec_p_n{n}_x{x}_a{tag(a_s)}"] = f(lr), f(pv)
    for name, ex in (("eq", exceptions_eq()), ("fx", exceptions_fx())):
        lr_i, p_i, lr_c, p_c = christoffersen(ex, mp.mpf("0.01"))
        vals[f"chr_{name}_ind_lr"], vals[f"chr_{name}_ind_p"] = f(lr_i), f(p_i)
        vals[f"chr_{name}_cc_lr"], vals[f"chr_{name}_cc_p"] = f(lr_c), f(p_c)
    for n, x, a_s in BASEL_CASES:
        vals[f"basel_cdf_n{n}_x{x}_a{tag(a_s)}"] = float(binom_cdf(n, x, Fraction(a_s)))

    return {
        "description": (
            "Independent oracle for the VaR/ES/backtest engines (W5). mpmath at 50 digits and "
            "exact rationals; imports nothing from eq_var or fx_var. Python, C++ and Rust must "
            "match closed forms to 1e-10 relative and MC within 3 standard errors."
        ),
        "generator": "python/equity/03-var-es-engine/tests/golden/var_oracle.py",
        "project": project,
        "inputs": {
            "alphas": ALPHAS, "dfs": DFS, "sigma_mean_odd": list(SIGMA_MEAN),
            "cornish_fisher": [list(c) for c in CF_CASES],
            "case_b": EQ_CASE_B if project == "equity" else FX_CASE_B,
            "hist_sample": "P_i = round2(1000 sin(0.37 i + 0.1) (1 + 0.5 cos(0.11 i))), i=0..249, "
                           "round2(x) = floor(100 x + 0.5) / 100",
            "hist_alphas": HIST_ALPHAS,
            "kupiec": [list(c) for c in KUPIEC_CASES],
            "christoffersen": {"eq": "t % 40 == 0 or t in (100, 101), t < 250",
                               "fx": "t % 37 == 5 or t in (100, 101), t < 250", "alpha": "0.01"},
            "basel": [list(c) for c in BASEL_CASES],
        },
        "versions": {"python": sys.version.split()[0], "mpmath": mp.__version__, "dps": mp.mp.dps},
        "tolerance": {"closed_form_rel": 1e-10, "mc_se": 3.0},
        "values": vals,
    }


def main(argv: list[str]) -> int:
    check = "--check" in argv
    bad = 0
    for project, path in (("equity", EQ_OUT), ("fx", FX_OUT)):
        data = build(project)
        text = json.dumps(data, indent=2, sort_keys=False) + "\n"
        if check:
            old = json.loads(path.read_text())
            if old["values"] != data["values"]:
                print(f"{path}: values differ from a fresh oracle run")
                bad += 1
        else:
            path.write_text(text)
            print(f"wrote {len(data['values'])} oracle values to {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
