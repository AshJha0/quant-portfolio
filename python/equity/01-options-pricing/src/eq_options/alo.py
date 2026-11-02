"""Andersen-Lake-Offengeim (ALO) American option pricer.

Plain idea: an American put is a European put plus the value of being
allowed to exercise early. That premium is an integral over the exercise
boundary ``B(tau)``, and the boundary itself solves a fixed-point equation.
ALO make the fixed point fast: they store a smooth transform of the
boundary at Chebyshev nodes in ``sqrt(tau)``, evaluate the integrals by
Gauss-Legendre after a change of variable that removes the
``1/sqrt(tau - u)`` singularity, and iterate a well-conditioned form of the
equation (their FP-B). A handful of nodes gives prices accurate to 1e-6 or
better, far faster than a tree.

Formal statement (Andersen, Lake and Offengeim 2016, "High-performance
American option pricing", J. Comput. Finance 20(1), 39-87; eq. (2) premium,
eqs. (25)-(27) FP-B, section 4 the collocation). For a put with
``r > 0, q >= 0``, ``X = K min(1, r/q)``, ``H(sqrt tau) = ln(B(tau)/X)**2``
interpolated on ``n + 1`` Chebyshev-Lobatto nodes, and

``B = K exp(-(r - q) tau) N(tau, B) / D(tau, B)`` with
``N = phi(d-(tau, B/K))/(sigma sqrt tau) + r int_0^tau e^{ru} phi(d-(tau-u, B(tau)/B(u)))/(sigma sqrt(tau-u)) du``,
``D = phi(d+)/(sigma sqrt tau) + Phi(d+) + q int_0^tau e^{qu} [phi(d+)/(sigma sqrt(tau-u)) + Phi(d+)] du``.

Price: ``P = p_E + int_0^T [r K e^{-r(T-u)} Phi(-d-(T-u, S/B(u))) - q S e^{-q(T-u)} Phi(-d+(T-u, S/B(u)))] du``.
Calls use McDonald-Schroder symmetry ``C(S, K, r, q) = P(K, S, q, r)``.

Worked example: ``S = K = 100, T = 1, r = 0.05, q = 0, sigma = 0.2``, put:
6.090373 (default nodes), against 6.090371 from Leisen-Reimer at 2001 and
4001 steps with Richardson extrapolation (``2 V_4001 - V_2001``).

Negative rates and the double boundary. Exercising a put early swaps the
option for ``K - S``; while held in the exercise region, that position
earns carry at rate ``r K - q S``. The sign of that carry sorts every
``(r, q)`` pair into one of three regions (Battauz, De Donno and Sbuelz
2015, "Real options and American derivatives: the double continuation
region", Management Science 61(5), 1094-1107; Andersen and Lake 2021,
"Fast American option pricing: the double-boundary case", Wilmott
2021(116), 30-41):

* single boundary, ``r > 0`` (any ``q``) or ``r = 0 > q``: FP-B as above,
  with ``X = K`` when ``q <= 0``. The ALO derivation does not use the sign
  of ``q``; only ``X`` changes.
* no early exercise, ``r <= 0`` and ``q >= r`` (other than ``r = 0 > q``):
  the carry is never positive in the money, so the European price.
* double boundary, ``q < r < 0``: early exercise pays only inside a band
  ``Y(tau) <= S <= B(tau)`` that starts at ``[K r/q, K]`` at expiry and
  closes at a finite ``tau*``; for longer maturities no new exercise
  region appears. The value is
  ``P = p_E + int_0^{min(T, tau*)} [r K e^{-r(T-u)} (Phi(-d-(S/B(u))) - Phi(-d-(S/Y(u))))
  - q S e^{-q(T-u)} (Phi(-d+(S/B(u))) - Phi(-d+(S/Y(u))))] du``,
  and the two boundaries are pinned by value matching,
  ``K - B(tau) = P(tau, B(tau))`` and ``K - Y(tau) = P(tau, Y(tau))``.
  Two solvers for that pair are provided.

  *Collocation (default, ``double_boundary_method="collocation"``).* The
  approach of Andersen and Lake (2021): write the band as a mid-line and a
  half-width, ``B = M + W`` and ``Y = M - W``, store both at ``n_cheb + 1``
  Chebyshev-Lobatto nodes in ``x = sqrt(tau / T_c)`` on ``[0, 1]``, with
  ``T_c = min(T, tau*)``, and impose value matching for both boundaries at
  every node. The time integrals use Gauss-Legendre in ``theta`` after
  ``u = tau sin^2(theta)``, which removes the square-root behaviour at both
  ends (``u -> 0`` where the boundaries start, ``u -> tau`` where the kernel
  is singular). The interpolation weights onto the quadrature points do not
  depend on ``T_c``, so they are built once. When the band closes before
  ``T`` the closing time is an extra unknown: the half-width is written
  ``W = (1 - x^2) w(x)`` so it vanishes at ``tau*`` (the two boundaries
  meet at a corner, with width close to linear in ``tau* - tau``; checked
  against the fine march), and the extra equation is smooth fit at the tip,
  ``dP/dS = -1`` at ``S = B(tau*)``. The square system (``2 n_cheb``
  unknowns) is solved by Newton with a finite-difference Jacobian, refreshed
  every three steps. A coarse 16-step march supplies the starting guess and
  says whether the band closes. At the default ``n_cheb = 8, n_quad = 24``
  a price takes about 15 ms on one core and is within 1e-5 of the 400-step
  march and of the extrapolated Leisen-Reimer tree on the test grid. If the
  Newton solve fails (never seen on the test grid) the pricer falls back to
  the march, so the answer is never worse than the reference.
  Our closing-time treatment is our own reading of the method; the paper's
  exact parametrisation of the closing segment is not reproduced.

  *March (reference, ``double_boundary_method="march"``).* March in
  ``tau`` on a grid uniform in ``sqrt(tau)`` (trapezoid in ``u``, a damped
  2x2 Newton per node), stopping where the band closes or the monotonicity
  (``B`` falling, ``Y`` rising) fails. It is slower (about 0.2 s at the
  default ``n_march = 400``) and accurate to about 1e-5 per 100 of strike
  against an extrapolated Leisen-Reimer tree, converging at close to second
  order. Known limitation: for some inputs a very fine grid (``n_march``
  around 1600) makes the first Newton step fail and the march stops at once.

Worked example (double boundary): ``S = K = 100, T = 1, r = -0.005,
q = -0.02, sigma = 0.2``, put: 7.410683 by collocation and 7.410691 by the
400-step march, against 7.410683 from the extrapolated tree; the European
put is 7.3291.

Calls use the same symmetry, so a call with ``r < q < 0`` is the
double-boundary put with the roles of ``r`` and ``q`` swapped.
Conventions are those of :mod:`eq_options.black_scholes`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .black_scholes import OptionType, _check_rates, bs_price, validate_inputs

__all__ = ["alo_boundary", "alo_double_boundary", "alo_price"]

DOUBLE_BOUNDARY_METHODS = ("collocation", "march")

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _d(tau: np.ndarray, z: np.ndarray, r: float, q: float, sigma: float) -> tuple:
    st = sigma * np.sqrt(tau)
    base = (np.log(z) + (r - q) * tau) / st
    return base + 0.5 * st, base - 0.5 * st


class _Cheb:
    """Barycentric interpolant on Chebyshev-Lobatto nodes ``z_i = cos(i pi / n)``."""

    def __init__(self, n: int) -> None:
        self.z = np.cos(np.pi * np.arange(n + 1) / n)
        w = (-1.0) ** np.arange(n + 1)
        w[0] *= 0.5
        w[-1] *= 0.5
        self.w = w

    def __call__(self, values: np.ndarray, z: np.ndarray) -> np.ndarray:
        diff = z[:, None] - self.z[None, :]
        exact = np.isclose(diff, 0.0, atol=1e-15)
        diff[exact] = 1.0
        t = self.w / diff
        out = (t @ values) / t.sum(axis=1)
        rows, cols = np.nonzero(exact)
        out[rows] = values[cols]
        return out


def _put_boundary(K, T, r, q, sigma, n, l, iterations):
    X = K * min(1.0, r / q) if q > 0.0 else K
    cheb = _Cheb(n)
    sqrt_tmax = math.sqrt(T)
    tau = (0.5 * sqrt_tmax * (1.0 + cheb.z)) ** 2  # tau_0 = T ... tau_n = 0
    y, wy = np.polynomial.legendre.leggauss(l)

    # Start from a flat boundary slightly below X; FP-B converges from it.
    H = np.full(n + 1, 0.0)
    H[:-1] = (0.5 * sigma * np.sqrt(tau[:-1])) ** 2

    def boundary_at(u: np.ndarray, Hv: np.ndarray) -> np.ndarray:
        zz = 2.0 * np.sqrt(u) / sqrt_tmax - 1.0
        return X * np.exp(-np.sqrt(np.maximum(cheb(Hv, zz), 0.0)))

    for _ in range(iterations):
        B = X * np.exp(-np.sqrt(H))
        B_new = B.copy()
        for i in range(n):  # node n is tau = 0, B = X
            t_i, b_i = tau[i], B[i]
            # u = tau - tau (1 + y)^2 / 4: tau - u = tau (1 + y)^2/4.
            s = 0.5 * (1.0 + y)
            u = t_i * (1.0 - s * s)
            dtau = t_i * s * s
            jac = 0.5 * t_i * (1.0 + y)  # du = -tau (1 + y)/2 dy
            bu = boundary_at(u, H)
            dp, dm = _d(dtau, b_i / bu, r, q, sigma)
            # Singular parts: du / (sigma sqrt(tau - u)) = sqrt(tau)/sigma dy.
            kn = r * np.sum(wy * np.exp(r * u) * _phi(dm) * (math.sqrt(t_i) / sigma))
            kd = q * np.sum(wy * np.exp(q * u) * (
                _phi(dp) * (math.sqrt(t_i) / sigma) + ndtr(dp) * jac))
            d1, d2 = _d(np.array(t_i), np.array(b_i / K), r, q, sigma)
            st = sigma * math.sqrt(t_i)
            Nv = float(_phi(d2)) / st + kn
            Dv = float(_phi(d1)) / st + float(ndtr(d1)) + kd
            B_new[i] = K * math.exp(-(r - q) * t_i) * Nv / Dv
        B_new = np.minimum(B_new, X)
        H_new = np.log(B_new / X) ** 2
        if np.max(np.abs(H_new - H)) < 1e-14:
            H = H_new
            break
        H = H_new
    return lambda u: boundary_at(np.asarray(u, dtype=float), H)


def _double_boundary(K, T, r, q, sigma, n):
    """March the band ``[Y(tau), B(tau)]`` for a put with ``q < r < 0``.

    Returns the node times ``u`` (from 0 up to the last tau at which the
    band still exists, capped at ``T``) and the boundary values there.
    """
    u = T * (np.arange(n + 1) / n) ** 2
    B = np.empty(n + 1)
    Y = np.empty(n + 1)
    B[0], Y[0] = K, K * r / q
    last = 0
    for i in range(1, n + 1):
        t = u[i]
        dt = t - u[:i]
        steps = np.diff(u[: i + 1])
        wk = np.empty(i)
        wk[0] = 0.5 * steps[0]
        wk[1:] = 0.5 * (steps[:-1] + steps[1:])
        wi = 0.5 * steps[-1]
        sq = sigma * np.sqrt(dt)
        drift = (r - q) * dt
        er, eq_ = np.exp(-r * dt), np.exp(-q * dt)
        lb, ly = np.log(B[:i]), np.log(Y[:i])
        st = sigma * math.sqrt(t)

        def resid(x, t=t, sq=sq, drift=drift, er=er, eq_=eq_, lb=lb, ly=ly, wk=wk, wi=wi, st=st):
            lx = np.log(x)[:, None]
            dmb = (lx - lb + drift) / sq - 0.5 * sq
            dmy = (lx - ly + drift) / sq - 0.5 * sq
            p_band = ndtr(-dmb) - ndtr(-dmy)
            q_band = ndtr(-(dmb + sq)) - ndtr(-(dmy + sq))
            # The u = tau node: at S = B (or S = Y) the band probability is 1/2.
            integral = (r * K * er * p_band - q * x[:, None] * eq_ * q_band) @ wk \
                + wi * 0.5 * (r * K - q * x)
            d1 = (np.log(x / K) + (r - q + 0.5 * sigma * sigma) * t) / st
            euro = K * math.exp(-r * t) * ndtr(-(d1 - st)) - x * math.exp(-q * t) * ndtr(-d1)
            return K - x - euro - integral

        prev = np.array([B[i - 1], Y[i - 1]])
        x = prev if i < 2 else 2.0 * prev - np.array([B[i - 2], Y[i - 2]])
        x = np.array([min(x[0], B[i - 1]), max(x[1], Y[i - 1])])
        converged = False
        for _ in range(50):
            f = resid(x)
            fnorm = float(np.max(np.abs(f)))
            if fnorm < 1e-11 * K:
                converged = True
                break
            jac = np.empty((2, 2))
            for j in range(2):
                xp = x.copy()
                xp[j] += 1e-7 * x[j]
                jac[:, j] = (resid(xp) - f) / (1e-7 * x[j])
            try:
                dx = np.linalg.solve(jac, -f)
            except np.linalg.LinAlgError:
                break
            lam, step_ok = 1.0, False
            while lam > 1e-4:
                xn = x + lam * dx
                if xn[1] > 0.0 and xn[0] > xn[1] and np.max(np.abs(resid(xn))) < fnorm:
                    step_ok = True
                    break
                lam *= 0.5
            if not step_ok:
                break
            x = xn
        if (not converged or x[0] - x[1] < 1e-9 * K
                or x[0] > B[i - 1] * (1.0 + 1e-12) or x[1] < Y[i - 1] * (1.0 - 1e-12)):
            break  # the band has closed: tau* lies in (u[last], u[i])
        B[i], Y[i] = x
        last = i
    return u[: last + 1], B[: last + 1], Y[: last + 1]


def _bary_w(x):
    d = x[:, None] - x[None, :]
    np.fill_diagonal(d, 1.0)
    return 1.0 / d.prod(axis=1)


def _bary_mat(xn, x):
    wn = _bary_w(xn)
    diff = x[:, None] - xn[None, :]
    exact = np.abs(diff) < 1e-15
    diff[exact] = 1.0
    t = wn / diff
    L = t / t.sum(axis=1, keepdims=True)
    rows = np.nonzero(exact.any(axis=1))[0]
    L[rows] = exact[rows].astype(float)
    return L


def _gl_theta(l):
    th, wt = np.polynomial.legendre.leggauss(l)
    th = 0.25 * math.pi * (1 + th)
    return np.sin(th), 2 * np.sin(th) * np.cos(th) * 0.25 * math.pi * wt


class _DoubleBoundaryCollocation:
    """Collocation system for the q < r < 0 put band (see the module note)."""

    def __init__(self, K, T, r, q, sigma, n=8, l=24):
        self.K, self.T, self.r, self.q, self.sigma, self.n, self.l = K, T, r, q, sigma, n, l
        sn, self.wq = _gl_theta(l)
        self.s2 = sn * sn
        self.x = 0.5 * (1 - np.cos(np.pi * np.arange(n + 1) / n))
        xq = (self.x[1:, None] * sn[None, :]).ravel()
        self.LMq = _bary_mat(self.x, xq)
        self.LWq_close = (1 - xq * xq)[:, None] * _bary_mat(self.x[:-1], xq)
        self.LWn_close = (1 - self.x[1:] ** 2)[:, None] * _bary_mat(self.x[:-1], self.x[1:])
        self.M0, self.W0 = 0.5 * K * (1 + r / q), 0.5 * K * (1 - r / q)

    def unpack(self, z, closing):
        n = self.n
        M = np.concatenate(([self.M0], z[:n]))
        if closing:
            return M, np.concatenate(([self.W0], z[n:2 * n - 1])), z[-1]
        return M, np.concatenate(([self.W0], z[n:])), self.T

    def _P(self, S, tau, top, Bq, Yq, s2, wq):
        K, r, q, sig = self.K, self.r, self.q, self.sigma
        u = top[:, None] * s2[None, :]
        w = top[:, None] * wq[None, :]
        dt = tau[:, None] - u
        sq = sig * np.sqrt(dt)
        drift = (r - q) * dt
        lS = np.log(S)[:, None]
        dmb = (lS - np.log(Bq) + drift) / sq - 0.5 * sq
        dmy = (lS - np.log(Yq) + drift) / sq - 0.5 * sq
        f = (r * K * np.exp(-r * dt) * (ndtr(-dmb) - ndtr(-dmy))
             - q * S[:, None] * np.exp(-q * dt) * (ndtr(-(dmb + sq)) - ndtr(-(dmy + sq))))
        st = sig * np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * tau) / st
        euro = K * np.exp(-r * tau) * ndtr(-(d1 - st)) - S * np.exp(-q * tau) * ndtr(-d1)
        return euro + np.sum(w * f, axis=1)

    def resid(self, z, closing):
        M, w, Tc = self.unpack(z, closing)
        if not Tc > 0:
            return None
        n, l = self.n, self.l
        Mq = (self.LMq @ M).reshape(n, l)
        Wq = ((self.LWq_close if closing else self.LMq) @ w).reshape(n, l)
        Bq, Yq = Mq + Wq, Mq - Wq
        Wn = self.LWn_close @ w if closing else w[1:]
        B, Y = M[1:] + Wn, M[1:] - Wn
        if np.any(Yq <= 0) or np.any(Y <= 0) or np.any(Wq < 0):
            return None
        if np.any(Wn[:-1] <= 0) or (not closing and Wn[-1] <= 0):
            return None
        tau = Tc * self.x[1:] ** 2
        m = n - 1 if closing else n
        S = np.concatenate((B, Y[:m]))
        tt = np.concatenate((tau, tau[:m]))
        P = self._P(S, tt, tt, np.vstack((Bq, Bq[:m])), np.vstack((Yq, Yq[:m])), self.s2, self.wq)
        R = (self.K - S - P) / self.K
        if closing:
            h = 1e-4 * B[-1]
            Pp = self._P(np.array([B[-1] + h, B[-1] - h]), np.array([Tc, Tc]), np.array([Tc, Tc]),
                         np.vstack((Bq[-1], Bq[-1])), np.vstack((Yq[-1], Yq[-1])), self.s2, self.wq)
            R = np.concatenate((R, [(Pp[0] - Pp[1]) / (2 * h) + 1.0]))
        return R

    def boundary(self, z, closing, u):
        M, w, Tc = self.unpack(z, closing)
        xu = np.sqrt(np.clip(np.asarray(u, float) / Tc, 0.0, 1.0))
        Mu = _bary_mat(self.x, xu) @ M
        if closing:
            Wu = (1 - xu * xu) * (_bary_mat(self.x[:-1], xu) @ w)
        else:
            Wu = _bary_mat(self.x, xu) @ w
        return Mu + Wu, Mu - Wu

    def price(self, S, z, closing, nq=48):
        Tc = z[-1] if closing else self.T
        top = min(self.T, Tc)
        sn, wq = _gl_theta(nq)
        B, Y = self.boundary(z, closing, top * sn * sn)
        p = self._P(np.array([S]), np.array([self.T]), np.array([top]), B[None, :], Y[None, :],
                    sn * sn, wq)
        return float(p[0])

    def solve(self, z, closing, tol=1e-12, maxit=40):
        f = self.resid(z, closing)
        if f is None:
            return None
        J = None
        age = 0
        for _ in range(maxit):
            fn_ = np.max(np.abs(f))
            if fn_ < tol:
                return z
            if J is None or age >= 3:
                J = np.empty((f.size, z.size))
                for j in range(z.size):
                    h = 1e-7 * max(abs(z[j]), 1.0)
                    zp = z.copy()
                    zp[j] += h
                    fp = self.resid(zp, closing)
                    if fp is None:
                        zp[j] -= 2 * h
                        fp = 2 * f - self.resid(zp, closing)
                    J[:, j] = (fp - f) / h
                age = 0
                fresh = True
            else:
                fresh = False
            dz = np.linalg.solve(J, -f)
            lam = 1.0
            ok = False
            while lam > 1e-3:
                zn = z + lam * dz
                fn = self.resid(zn, closing)
                if fn is not None and np.max(np.abs(fn)) < fn_:
                    ok = True
                    break
                lam *= 0.5
            if not ok:
                if fresh:
                    return None
                J = None
                continue
            z, f = zn, fn
            age += 1
        return z if np.max(np.abs(f)) < 1e-9 else None


def _collocate(K, T, r, q, sigma, n=8, l=24, nm=16):
    """Solve the band; return ``(solver, z, closing)`` or ``None`` on failure."""
    u, B, Y = _double_boundary(K, T, r, q, sigma, nm)
    if u.size < 3:
        return None
    db = _DoubleBoundaryCollocation(K, T, r, q, sigma, n, l)
    closing = u[-1] < T
    if closing:
        wa, wb = B[-2] - Y[-2], B[-1] - Y[-1]
        Tc = u[-1] + wb * (u[-1] - u[-2]) / max(wa - wb, 1e-300)
    else:
        Tc = T
    tau_n = Tc * db.x ** 2
    Bi = np.interp(np.sqrt(tau_n), np.sqrt(u), B)
    Yi = np.interp(np.sqrt(tau_n), np.sqrt(u), Y)
    if closing:
        mid = 0.5 * (B[-1] + Y[-1])
        far = tau_n > u[-1]
        frac = (Tc - tau_n[far]) / (Tc - u[-1])
        Bi[far] = mid + frac * (B[-1] - mid)
        Yi[far] = mid - frac * (mid - Y[-1])
    Mn, Wn = 0.5 * (Bi + Yi), 0.5 * (Bi - Yi)
    if closing:
        z0 = np.concatenate((Mn[1:], Wn[1:-1] / (1 - db.x[1:-1] ** 2), [Tc]))
    else:
        z0 = np.concatenate((Mn[1:], Wn[1:]))
    z = db.solve(z0, closing)
    if z is None or (closing and not z[-1] < T):
        return None
    return db, z, closing


def _double_boundary_put_collocation(S, K, T, r, q, sigma, n, l):
    sol = _collocate(K, T, r, q, sigma, n, l)
    if sol is None:
        return None
    db, z, closing = sol
    if not closing:
        Bt, Yt = db.boundary(z, closing, [T])
        if Yt[0] <= S <= Bt[0]:
            return K - S
    return db.price(S, z, closing)


def _double_boundary_put(S, K, T, r, q, sigma, n):
    u, B, Y = _double_boundary(K, T, r, q, sigma, n)
    if u[-1] >= T and Y[-1] <= S <= B[-1]:
        return K - S
    dt = T - u
    w = np.zeros(u.size)
    steps = np.diff(u)
    w[:-1] += 0.5 * steps
    w[1:] += 0.5 * steps
    sq = sigma * np.sqrt(np.maximum(dt, 1e-300))
    drift = (r - q) * dt
    dmb = (math.log(S) - np.log(B) + drift) / sq - 0.5 * sq
    dmy = (math.log(S) - np.log(Y) + drift) / sq - 0.5 * sq
    p_band = ndtr(-dmb) - ndtr(-dmy)
    q_band = ndtr(-(dmb + sq)) - ndtr(-(dmy + sq))
    premium = np.sum(w * (r * K * np.exp(-r * dt) * p_band - q * S * np.exp(-q * dt) * q_band))
    return float(bs_price(S, K, T, r, sigma, q, "put") + premium)


def _check_db_method(method, n_cheb, n_quad_db):
    if method not in DOUBLE_BOUNDARY_METHODS:
        raise ValueError(f"double_boundary_method must be one of {DOUBLE_BOUNDARY_METHODS}")
    if method == "collocation" and (n_cheb < 3 or n_quad_db < 4):
        raise ValueError("collocation needs n_cheb >= 3 and n_quad_db >= 4")


def alo_double_boundary(K: float, T: float, r: float, sigma: float, q: float,
                        n_march: int = 400, method: str = "collocation",
                        n_cheb: int = 8, n_quad_db: int = 24):
    """Return ``(tau, lower, upper)``: the put exercise band for ``q < r < 0``.

    ``tau`` is the grid ``T_c (k / n_march)^2`` with ``T_c = min(T, tau*)``.
    With ``method="collocation"`` the boundaries are the collocation
    interpolants on that grid; when the band closes before ``T`` the grid
    stops one point short of ``tau*`` (where the two boundaries meet). With
    ``method="march"`` they are the march values, which stop at the last grid
    time before the band closes (see the module note). Collocation falls back
    to the march if its solve fails.
    """
    if not q < r < 0.0:
        raise ValueError("alo_double_boundary covers q < r < 0 (put)")
    if min(K, T, sigma) <= 0.0 or n_march < 2:
        raise ValueError("need K, T, sigma > 0 and n_march >= 2")
    _check_db_method(method, n_cheb, n_quad_db)
    if method == "collocation":
        sol = _collocate(K, T, r, q, sigma, n_cheb, n_quad_db)
        if sol is not None:
            db, z, closing = sol
            Tc = float(z[-1]) if closing else T
            u = Tc * (np.arange(n_march + 1) / n_march) ** 2
            if closing:
                u = u[:-1]
            B, Y = db.boundary(z, closing, u)
            B[0], Y[0] = K, K * r / q
            return u, Y, B
    u, B, Y = _double_boundary(K, T, r, q, sigma, n_march)
    return u, Y, B


def alo_boundary(
    K: float, T: float, r: float, sigma: float, q: float = 0.0,
    n_nodes: int = 12, n_quad: int = 32, iterations: int = 60,
):
    """Return a callable ``B(tau)``: the American put exercise boundary."""
    if not (r > 0.0 or (r == 0.0 and q < 0.0)):
        raise ValueError("alo_boundary covers the single-boundary put region r > 0 or r = 0 > q")
    return _put_boundary(K, T, r, q, sigma, n_nodes, n_quad, iterations)


def alo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "put",
    n_nodes: int = 12,
    n_quad: int = 32,
    n_price: int = 64,
    iterations: int = 60,
    n_march: int = 400,
    double_boundary_method: str = "collocation",
    n_cheb: int = 8,
    n_quad_db: int = 24,
) -> float:
    """American option price by the Andersen-Lake-Offengeim method.

    Parameters
    ----------
    S, K, T, sigma : float
        Strictly positive spot, strike, expiry in years and volatility.
    r, q : float
        Rate and dividend yield; see the module scope note.
    option_type : {"call", "put"}
    n_nodes, n_quad, n_price : int
        Chebyshev nodes, boundary quadrature points, price quadrature points.
    iterations : int
        Maximum FP-B iterations (stops early on convergence).
    n_march : int
        Time steps for the double-boundary march (``q < r < 0`` put,
        ``r < q < 0`` call); ignored elsewhere.
    double_boundary_method : {"collocation", "march"}
        Solver for the double boundary (module note). Collocation is the
        default; the march is kept as the reference.
    n_cheb, n_quad_db : int
        Chebyshev nodes and Gauss-Legendre points of the double-boundary
        collocation.

    Returns
    -------
    float
        Present value in currency units.
    """
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r, q=q)
    if min(S, K, T, sigma) <= 0.0:
        raise ValueError("alo_price needs S, K, T and sigma strictly positive")
    if option_type == "call":
        if q <= 0.0 and r >= 0.0:
            return bs_price(S, K, T, r, sigma, q, "call")  # never exercised early
        return alo_price(K, S, T, q, sigma, r, "put", n_nodes, n_quad, n_price, iterations,
                         n_march, double_boundary_method, n_cheb, n_quad_db)
    if r <= 0.0 and q >= r and not (r == 0.0 and q < 0.0):
        return bs_price(S, K, T, r, sigma, q, "put")  # carry r K - q S never positive in the money
    if r < 0.0:  # q < r < 0: the double boundary
        if n_march < 2:
            raise ValueError("n_march must be >= 2")
        _check_db_method(double_boundary_method, n_cheb, n_quad_db)
        if double_boundary_method == "collocation":
            v = _double_boundary_put_collocation(S, K, T, r, q, sigma, n_cheb, n_quad_db)
            if v is not None:
                return float(v)
        return _double_boundary_put(S, K, T, r, q, sigma, n_march)
    B = _put_boundary(K, T, r, q, sigma, n_nodes, n_quad, iterations)
    european = bs_price(S, K, T, r, sigma, q, "put")
    if S <= float(B(np.array([T]))[0]):
        return K - S
    # Integrate over v = sqrt(u) on [0, sqrt(T)], du = 2 v dv.
    y, w = np.polynomial.legendre.leggauss(n_price)
    v = 0.5 * math.sqrt(T) * (1.0 + y)
    u = v * v
    bu = B(u)
    rem = T - u
    rem = np.maximum(rem, 1e-300)
    dp, dm = _d(rem, S / bu, r, q, sigma)
    integrand = r * K * np.exp(-r * rem) * ndtr(-dm) - q * S * np.exp(-q * rem) * ndtr(-dp)
    premium = 0.5 * math.sqrt(T) * np.sum(w * integrand * 2.0 * v)
    return float(european + premium)
