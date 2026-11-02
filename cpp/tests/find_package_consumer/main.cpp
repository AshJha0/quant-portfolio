// Calls one function from each installed library and checks the answer.
// Exit status 0 means headers, libraries and transitive targets all resolved.

#include <cmath>
#include <cstdio>

#include <eqopt/black_scholes.hpp>
#include <eqvar/stats.hpp>
#include <fxopt/garman_kohlhagen.hpp>
#include <fxvar/stats.hpp>
#include <qp/normal.hpp>

namespace {

int failures = 0;

void check(const char* what, double got, double want, double tol) {
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("FAIL %s: got %.15g, want %.15g\n", what, got, want);
        ++failures;
    } else {
        std::printf("ok   %s = %.15g\n", what, got);
    }
}

}  // namespace

int main() {
    // Black-Scholes, S=K=100, T=1, r=5%, sigma=20%, q=0: 10.4505835721856
    // (Hull, Options, Futures and Other Derivatives, standard textbook case).
    check("eqopt::bs_price", eqopt::bs_price(100.0, 100.0, 1.0, 0.05, 0.20), 10.4505835721856,
          1e-10);
    // Garman-Kohlhagen with r_f = 0 is Black-Scholes with q = 0.
    check("fxopt::gk_price",
          fxopt::gk_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, fxopt::OptionType::Call),
          10.4505835721856, 1e-10);
    check("eqvar::normal_cdf(0)", eqvar::normal_cdf(0.0), 0.5, 1e-15);
    check("fxvar::norm_cdf(0)", fxvar::norm_cdf(0.0), 0.5, 1e-15);
    check("qp::norm_pdf(0)", qp::norm_pdf(0.0), 0.398942280401433, 1e-14);
    return failures == 0 ? 0 : 1;
}
