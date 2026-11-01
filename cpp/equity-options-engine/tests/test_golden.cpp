// Golden vectors: the engine must match the mpmath oracle (generated in
// python/equity/01-options-pricing/tests/golden/) within the per-output
// absolute tolerance carried by each record. Python and Rust consume the
// same JSON.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

#include "eqopt/binomial.hpp"
#include "eqopt/black76.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/implied_vol.hpp"
#include "golden_vectors.hpp"

namespace {

using eqopt::golden::GoldenRecord;
using eqopt::golden::kRecords;

std::map<std::string, double> evaluate(const GoldenRecord& r) {
    const auto& x = r.in;
    const std::string fn = r.fn;
    if (fn == "bs") {
        if (r.n_out == 1) return {{"price", eqopt::bs_price(x[0], x[1], x[2], x[3], x[5], x[4], r.type)}};
        const auto g = eqopt::bs_greeks(x[0], x[1], x[2], x[3], x[5], x[4], r.type);
        return {{"price", g.price}, {"delta", g.delta}, {"gamma", g.gamma}, {"vega", g.vega},
                {"theta", g.theta}, {"rho", g.rho},     {"vanna", g.vanna}, {"volga", g.volga}};
    }
    if (fn == "iv") {
        return {{"sigma", eqopt::implied_vol(x[0], x[1], x[2], x[3], x[4], x[5], r.type)}};
    }
    if (fn == "crr") {
        const auto ex = r.flag ? eqopt::ExerciseStyle::American : eqopt::ExerciseStyle::European;
        return {{"price", eqopt::crr_price(x[0], x[1], x[2], x[3], x[5], x[4], r.type, ex,
                                           static_cast<int>(x[6]))}};
    }
    if (fn == "b76") {
        const auto g = eqopt::black76_greeks(x[0], x[1], x[2], x[3], x[4], r.type);
        return {{"price", g.price}, {"delta", g.delta}, {"gamma", g.gamma},
                {"vega", g.vega},   {"theta", g.theta}, {"rho", g.rho}};
    }
    ADD_FAILURE() << "unknown golden fn " << fn;
    return {};
}

TEST(GoldenVectors, CoverageAndSize) {
    EXPECT_GE(kRecords.size(), 400u);
}

TEST(GoldenVectors, AllRecordsMatchOracle) {
    double worst = 0.0;
    for (std::size_t i = 0; i < kRecords.size(); ++i) {
        const auto& r = kRecords[i];
        const auto got = evaluate(r);
        for (int k = 0; k < r.n_out; ++k) {
            const auto& o = r.out[static_cast<std::size_t>(k)];
            const double v = got.at(o.name);
            EXPECT_LE(std::fabs(v - o.value), o.tol)
                << "record " << i << " (" << r.fn << ") output " << o.name << ": " << v
                << " vs " << o.value;
            worst = std::max(worst, std::fabs(v - o.value) / o.tol);
        }
    }
    // Worst deviation as a fraction of its tolerance: a drift towards the
    // bound shows up here before a record fails.
    RecordProperty("worst_dev_over_tol", std::to_string(worst));
    EXPECT_LE(worst, 1.0);
}

}  // namespace
