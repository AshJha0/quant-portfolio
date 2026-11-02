// Generates the Chebyshev coefficients of qp::vm::erfc_batch (qp/vmath.hpp)
// and qp_core::vmath::erfc_batch (rust/core/src/vmath.rs).
//
// For z >= 0 write t = 2 / (2 + z) and
//     g(t) = log(erfc(z) / t) + z^2,
// a smooth function of t (the method of Press et al., Numerical Recipes,
// 3rd ed., 2007, section 6.2.2). The fit covers z in [0, kZMax], that is
// t in [t0, 1] with t0 = 2 / (2 + kZMax), mapped to u = (2t - 1 - t0) / (1 - t0)
// on [-1, 1]. Coefficients are the discrete Chebyshev transform at N
// Chebyshev nodes, computed in long double (x87, 64-bit mantissa) from
// erfcl, then truncated where they fall below 1e-17.
//
// Build and run (Linux, x86-64):
//     g++ -O2 -std=c++23 gen_erfc_cheb.cpp -o gen && ./gen
#include <cmath>
#include <cstdio>
#include <numbers>

int main() {
    constexpr int kN = 64;
    constexpr long double kZMax = 27.0L;
    const long double t0 = 2.0L / (2.0L + kZMax);
    const long double pi = std::numbers::pi_v<long double>;
    long double f[kN];
    for (int j = 0; j < kN; ++j) {
        const long double u = std::cos(pi * (j + 0.5L) / kN);
        const long double t = 0.5L * ((1.0L - t0) * u + 1.0L + t0);
        const long double z = 2.0L / t - 2.0L;
        f[j] = std::log(std::erfc(z) / t) + z * z;
    }
    std::printf("// t0 = %.17Lg\n", t0);
    int last = 0;
    long double c[kN];
    for (int k = 0; k < kN; ++k) {
        long double s = 0.0L;
        for (int j = 0; j < kN; ++j) s += f[j] * std::cos(pi * k * (j + 0.5L) / kN);
        c[k] = (k == 0 ? 1.0L : 2.0L) * s / kN;
        if (std::fabs(c[k]) > 1e-17L) last = k;
    }
    for (int k = 0; k <= last; ++k) std::printf("%a,  // c%d = %.17Lg\n", (double)c[k], k, c[k]);
    return 0;
}
