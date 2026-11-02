// Golden vectors: the engine must match the mpmath oracle (generated in
// python/fx/01-options-pricing/tests/golden/) within the per-output absolute
// tolerance carried by each record. Python and Rust consume the same JSON.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>

#include "fxopt/binomial.hpp"
#include "fxopt/deltas.hpp"
#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"
#include "golden_vectors.hpp"

namespace {

using namespace fxopt;
using fxopt::golden::GoldenRecord;
using fxopt::golden::kRecords;

constexpr std::array<DeltaConvention, 4> kConventions = {
    DeltaConvention::Spot, DeltaConvention::Forward, DeltaConvention::SpotPa,
    DeltaConvention::ForwardPa};

std::map<std::string, double> evaluate(const GoldenRecord& r) {
    const auto& x = r.in;
    const std::string fn = r.fn;
    if (fn == "gk") {
        if (r.n_out == 1) return {{"price", gk_price(x[0], x[1], x[2], x[3], x[4], x[5], r.type)}};
        const GreeksResult g = analytic_greeks(x[0], x[1], x[2], x[3], x[4], x[5], r.type);
        return {{"price", g.price},
                {"delta_spot", g.delta_spot},
                {"delta_fwd", g.delta_forward},
                {"delta_spot_pa",
                 delta(x[0], x[1], x[2], x[3], x[4], x[5], r.type, DeltaConvention::SpotPa)},
                {"delta_fwd_pa",
                 delta(x[0], x[1], x[2], x[3], x[4], x[5], r.type, DeltaConvention::ForwardPa)},
                {"gamma", g.gamma},
                {"vega", g.vega},
                {"theta", g.theta},
                {"rho_d", g.rho_domestic},
                {"rho_f", g.rho_foreign},
                {"vanna", g.vanna},
                {"volga", g.volga}};
    }
    if (fn == "iv") return {{"sigma", implied_vol(x[0], x[1], x[2], x[3], x[4], x[5], r.type)}};
    if (fn == "strike") {
        return {{"K", strike_from_delta(x[0], x[1], x[2], x[3], x[4], x[5], r.type,
                                        kConventions[static_cast<std::size_t>(r.flag)])}};
    }
    if (fn == "dns_atm") {
        const auto conv = r.flag ? DeltaConvention::SpotPa : DeltaConvention::Spot;
        return {{"K", atm_dns_strike(x[0], x[1], x[2], x[3], x[4], conv)}};
    }
    if (fn == "crr") {
        const auto ex = r.flag ? Exercise::American : Exercise::European;
        return {{"price", binomial_price(x[0], x[1], x[2], x[3], x[4], x[5], r.type,
                                         static_cast<int>(x[6]), ex)}};
    }
    if (fn == "fwd") return {{"forward", cip_forward(x[0], x[1], x[2], x[3])}};
    ADD_FAILURE() << "unknown golden fn " << fn;
    return {};
}

TEST(GoldenVectors, CoverageAndSize) { EXPECT_GE(kRecords.size(), 400u); }

TEST(GoldenVectors, AllRecordsMatchOracle) {
    double worst = 0.0;
    for (std::size_t i = 0; i < kRecords.size(); ++i) {
        const auto& r = kRecords[i];
        const auto got = evaluate(r);
        for (int k = 0; k < r.n_out; ++k) {
            const auto& o = r.out[static_cast<std::size_t>(k)];
            const double v = got.at(o.name);
            EXPECT_LE(std::fabs(v - o.value), o.tol)
                << "record " << i << " (" << r.fn << ") output " << o.name << ": " << v << " vs "
                << o.value;
            worst = std::max(worst, std::fabs(v - o.value) / o.tol);
        }
    }
    RecordProperty("worst_dev_over_tol", std::to_string(worst));
    EXPECT_LE(worst, 1.0);
}

}  // namespace
