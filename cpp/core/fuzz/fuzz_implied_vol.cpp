// Coverage-guided fuzz target for the shared implied-vol core
// (qp/iv_core.hpp), which every options engine calls.
//
// Property checked on every input:
//   1. black_implied_vol_nx never crashes, never trips a sanitizer, and
//      returns Ok, AtBound or NoConvergence (no other path).
//   2. Round trip: for a log-moneyness x <= 0 and a total vol s in
//      [1e-3, 6], beta = normalised_otm_call(x, s) is a valid OTM price;
//      when beta is not lost to underflow, total_vol_nx must return Ok and
//      either recover s to 1e-10 relative or reprice beta to 1e-9 relative.
//      Either one alone is wrong somewhere: where vega is tiny many vols
//      give the same price, so only the price can be asked for; deep out of
//      the money the price elasticity d ln(beta) / d ln(s) is about
//      x^2 / s^2 (over 1,000 at x = -0.41, s = 0.0127, the first case the
//      fuzzer found), so a vol right to 5e-13 moves beta by 1.7e-9.
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "fuzz_input.hpp"
#include "qp/iv_core.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    qp_fuzz::FuzzInput in(data, size);

    // (1) Raw quote, any finite inputs in a wide box.
    {
        const double F = in.in(1e-6, 1e6);
        const double K = in.in(1e-6, 1e6);
        const double T = in.in(1e-6, 50.0);
        const double df = in.in(1e-3, 1.0);
        const bool is_call = in.flag();
        const double upper = is_call ? df * F : df * K;
        const double price = in.in(0.0, upper);
        int evals = 0;
        double out = 0.0;
        const auto st = qp::iv_core::black_implied_vol_nx(price, F, K, T, df, is_call, evals, out);
        if (st == qp::iv_core::IvStatus::Ok && !(std::isfinite(out) && out > 0.0)) std::abort();
        if (evals < 0 || evals > 1000) std::abort();
    }

    // (2) Round trip through the normalised price.
    {
        const double x = -in.in(0.0, 10.0);
        const double s = in.in(1e-3, 6.0);
        const double beta = qp::iv_core::normalised_otm_call(x, s);
        const double u = qp::iv_core::normalised_otm_call_complement(x, s);
        if (beta > 1e-280 && u > 1e-280 && std::isfinite(beta)) {
            int evals = 0;
            double s_hat = 0.0;
            const auto st = qp::iv_core::total_vol_nx(beta, u, x, evals, s_hat);
            if (st != qp::iv_core::IvStatus::Ok) std::abort();
            const double beta_hat = qp::iv_core::normalised_otm_call(x, s_hat);
            const bool vol_ok = std::fabs(s_hat - s) <= 1e-10 * s;
            const bool price_ok = std::fabs(beta_hat - beta) <= 1e-9 * beta;
            if (!(vol_ok || price_ok)) std::abort();
        }
    }
    return 0;
}
