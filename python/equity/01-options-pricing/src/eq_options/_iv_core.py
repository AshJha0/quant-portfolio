"""Shared implied-volatility core: normalised Black, solved on the OTM side.

The same algorithm, constants and branch thresholds are implemented in the
Python, C++ and Rust engines. Inputs to :func:`total_vol` are the normalised
OTM time value ``beta``, its complement ``u = b_max - beta`` and
``x = -|ln(F/K)| <= 0``; the result is the total volatility
``s = sigma sqrt(T)``. One evaluation is one pass of the iteration loop.
"""
import math

from qp_core.normal import norm_cdf, norm_pdf

SQRT2 = 1.4142135623730951
INV_SQRT_2PI = 0.3989422804014327
LN_SQRT_2PI = 0.9189385332046728
SQRT_2PI = 2.5066282746310002
EPS = 2.220446049250313e-16
MAX_ITER = 100




def normalised_otm_call(x, s):
    """b(x,s) = e^{x/2} N(x/s+s/2) - e^{-x/2} N(x/s-s/2) for x <= 0, s > 0,
    written as cosh(x/2) A + sinh(x/2) (N(d1)+N(d2)), A = N(d1) - N(d2)."""
    h = x / s
    t = 0.5 * s
    d1 = h + t
    d2 = h - t
    if -x < 0.1 and s < 0.2:
        # A = 2 n(h) sum_k He_{2k}(h) t^{2k+1} / (2k+1)!
        he_prev, he_cur = 1.0, h          # He_0, He_1
        n = 1
        tt = t * t
        term = t
        acc = t
        small = 0
        for k in range(1, 40):
            he_next = h * he_cur - n * he_prev
            n += 1
            he_prev, he_cur = he_cur, he_next
            he_next = h * he_cur - n * he_prev
            n += 1
            he_prev, he_cur = he_cur, he_next
            # he_prev is now He_{2k}
            term *= tt / ((2 * k) * (2 * k + 1))
            c = he_prev * term
            acc += c
            # two consecutive negligible terms: He_{2k}(h) can vanish at isolated h
            small = small + 1 if abs(c) <= 1e-17 * abs(acc) else 0
            if small == 2:
                break
        a = 2.0 * norm_pdf(h) * acc
    elif abs(d1) < 1.0 and abs(d2) < 1.0:
        a = 0.5 * (math.erf(d1 / SQRT2) - math.erf(d2 / SQRT2))
    else:
        a = norm_cdf(d1) - norm_cdf(d2)
    return math.cosh(0.5 * x) * a + math.sinh(0.5 * x) * (norm_cdf(d1) + norm_cdf(d2))


def normalised_otm_call_complement(x, s):
    """b_max - b = e^{x/2} N(-d1) + e^{-x/2} N(d2): a sum, no cancellation."""
    d1 = x / s + 0.5 * s
    d2 = x / s - 0.5 * s
    return math.exp(0.5 * x) * norm_cdf(-d1) + math.exp(-0.5 * x) * norm_cdf(d2)


def acklam_guess(p):
    """Acklam inverse normal, no refinement: used for the initial guess only."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def total_vol(beta, u, x, counter=None):
    """Solve b(x, s) = beta for s. Requires beta > 0, u > 0, x <= 0."""
    b_max = math.exp(0.5 * x)
    s_c = math.sqrt(-2.0 * x)                     # inflection point of b in s
    b_c = normalised_otm_call(x, s_c) if s_c > 0.0 else 0.0
    if beta < b_c:
        mode = 0                                  # lower: f = 1/ln(beta) - 1/ln b(s)
        ln_target = math.log(beta)
        lo, hi = 0.0, s_c
        # Mills-ratio asymptotic, y = s^2:
        # ln b ~ -x^2/(2y) - y/8 - ln sqrt(2pi) + 1.5 ln y - ln(x^2 - y^2/4)
        y = x * x / (-2.0 * ln_target)
        for _ in range(3):
            if not y < -2.0 * x:
                break
            rhs = (-ln_target - 0.125 * y - LN_SQRT_2PI + 1.5 * math.log(y)
                   - math.log(x * x - 0.25 * y * y))
            if not rhs > 0.0:
                break
            y = x * x / (2.0 * rhs)
        s = min(math.sqrt(y), s_c)
    else:
        lo, hi = s_c, math.inf
        if beta <= 0.5 * b_max:
            mode = 1                              # middle: f = ln b(s) - ln(beta)
            ln_target = math.log(beta)
        else:
            mode = 2                              # upper: f = ln(u) - ln(b_max - b(s))
            ln_target = math.log(u)
        p = u / (b_max + 1.0 / b_max)            # < 0.5 always
        s = max(-2.0 * acklam_guess(p), s_c)
        if not s > 0.0:
            s = SQRT_2PI * beta
    # The bracket ends start as the a-priori bounds (0 or s_c, s_c or inf);
    # an end is "probed" once the objective has been evaluated there.
    lo_probed = hi_probed = False
    for _ in range(MAX_ITER):
        if counter is not None:
            counter[0] += 1
        d1 = x / s + 0.5 * s
        bp = b_max * norm_pdf(d1)                 # db/ds
        r2 = x * x / (s * s * s) - 0.25 * s       # b''/b'
        if mode == 0:
            b = normalised_otm_call(x, s)
            if not b > 0.0:                       # underflow: s is far too small
                lo = s
                s = 0.5 * (lo + hi)
                continue
            L = math.log(b)
            Lp = bp / b
            Lpp = Lp * r2 - Lp * Lp
            f = 1.0 / ln_target - 1.0 / L
            fp = Lp / (L * L)
            fpp = Lpp / (L * L) - 2.0 * Lp * Lp / (L * L * L)
        elif mode == 1:
            b = normalised_otm_call(x, s)
            Lp = bp / b
            f = math.log(b) - ln_target
            fp = Lp
            fpp = Lp * r2 - Lp * Lp
        else:
            w = normalised_otm_call_complement(x, s)
            Gp = bp / w
            f = ln_target - math.log(w)
            fp = Gp
            fpp = Gp * r2 + Gp * Gp
        if f > 0.0:
            hi = min(hi, s)
            hi_probed = hi == s
        elif f < 0.0:
            lo = max(lo, s)
            lo_probed = lo == s
        else:
            return s
        newton = f / fp
        den = 1.0 - 0.5 * newton * fpp / fp
        s_new = s - newton / den if den > 0.5 else math.nan
        if lo < s_new < hi:
            tol = 1e-7                             # Halley: next error ~ step^3
        else:
            s_new = s - newton
            if lo < s_new < hi:
                tol = 1e-10                        # Newton: next error ~ step^2
            elif s_new >= hi and math.isfinite(hi) and not hi_probed:
                s_new = hi                         # root lies next to s_c: probe it
                tol = 0.0
            elif s_new <= lo and lo > 0.0 and not lo_probed:
                s_new = lo
                tol = 0.0
            else:
                s_new = 0.5 * (lo + hi) if math.isfinite(hi) else 2.0 * s
                tol = 0.0
        if abs(s_new - s) <= max(tol, 4.0 * EPS) * s or (hi - lo) <= 4.0 * EPS * s:
            return s_new
        s = s_new
    raise ValueError("implied vol: no convergence in 100 iterations")


def black_implied_vol(price, F, K, T, df, is_call, counter=None):
    """Caller has validated inputs and checked lower < price < upper."""
    lower = df * max((F - K) if is_call else (K - F), 0.0)
    upper = df * F if is_call else df * K
    norm = df * math.sqrt(F * K)
    beta = (price - lower) / norm
    u = (upper - price) / norm
    if not (beta > 0.0 and u > 0.0):
        raise ValueError("price is within rounding of a no-arbitrage bound")
    x = -abs(math.log1p((F - K) / K))   # F-K is exact near the money (Sterbenz)
    return total_vol(beta, u, x, counter) / math.sqrt(T)
