// Internal Brent root finder (implementation detail; not part of the public
// API).  Mirrors scipy.optimize.brentq semantics: given f(a) f(b) <= 0,
// returns a root to absolute x-tolerance xtol.

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace fxopt::detail {

template <typename F>
double brentq(F&& f, double a, double b, double xtol = 1e-14,
              int max_iter = 200) {
    double fa = f(a);
    double fb = f(b);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa * fb > 0.0) {
        throw std::invalid_argument(
            "brentq: root not bracketed on [" + std::to_string(a) + ", " +
            std::to_string(b) + "]");
    }
    double c = a, fc = fa;
    double e = b - a, dstep = e;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fc) < std::abs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        const double tol1 = 2.0 * 2.220446049250313e-16 * std::abs(b) +
                            0.5 * xtol;
        const double xm = 0.5 * (c - b);
        if (std::abs(xm) <= tol1 || fb == 0.0) return b;
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            // Attempt inverse quadratic interpolation / secant.
            const double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                const double qq = fa / fc;
                const double r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::abs(p);
            const double min1 = 3.0 * xm * q - std::abs(tol1 * q);
            const double min2 = std::abs(e * q);
            if (2.0 * p < std::min(min1, min2)) {
                e = dstep;
                dstep = p / q;
            } else {
                dstep = xm;
                e = dstep;
            }
        } else {
            dstep = xm;
            e = dstep;
        }
        a = b;
        fa = fb;
        if (std::abs(dstep) > tol1) {
            b += dstep;
        } else {
            b += (xm > 0.0 ? tol1 : -tol1);
        }
        fb = f(b);
        if ((fb > 0.0) == (fc > 0.0)) {
            c = a;
            fc = fa;
            e = b - a;
            dstep = e;
        }
    }
    return b;
}

}  // namespace fxopt::detail
