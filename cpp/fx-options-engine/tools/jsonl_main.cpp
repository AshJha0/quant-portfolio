// JSON-lines shim over the FX engine for the cross-language differential
// test. Protocol: see python/fx/01-options-pricing/tests/test_differential.py.
// Functions: gk, iv, strike, dns_atm, crr, fwd.
#include <exception>
#include <iostream>
#include <string>

#include "fxopt/binomial.hpp"
#include "fxopt/deltas.hpp"
#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"
#include "qp/tools/jsonl.hpp"

namespace {

using namespace fxopt;

OptionType option_type(const jsonl::Request& r) {
    const std::string& t = r.str("type");
    if (t == "call") return OptionType::Call;
    if (t == "put") return OptionType::Put;
    throw std::invalid_argument("type must be call or put");
}

DeltaConvention convention(const std::string& c) {
    if (c == "spot") return DeltaConvention::Spot;
    if (c == "forward") return DeltaConvention::Forward;
    if (c == "spot_pa") return DeltaConvention::SpotPa;
    if (c == "forward_pa") return DeltaConvention::ForwardPa;
    throw std::invalid_argument("unknown convention: " + c);
}

std::string handle(const jsonl::Request& r, const std::string& id) {
    const std::string& fn = r.str("fn");
    if (fn == "gk") {
        const double S = r.num("S"), K = r.num("K"), T = r.num("T"), rd = r.num("rd"),
                     rf = r.num("rf"), sig = r.num("sigma");
        const OptionType ot = option_type(r);
        const GreeksResult g = analytic_greeks(S, K, T, rd, rf, sig, ot);
        return jsonl::ok(
            id, {{"price", g.price},
                 {"delta_spot", g.delta_spot},
                 {"delta_fwd", g.delta_forward},
                 {"delta_spot_pa", delta(S, K, T, rd, rf, sig, ot, DeltaConvention::SpotPa)},
                 {"delta_fwd_pa", delta(S, K, T, rd, rf, sig, ot, DeltaConvention::ForwardPa)},
                 {"gamma", g.gamma},
                 {"vega", g.vega},
                 {"theta", g.theta},
                 {"rho_d", g.rho_domestic},
                 {"rho_f", g.rho_foreign},
                 {"vanna", g.vanna},
                 {"volga", g.volga}});
    }
    if (fn == "iv") {
        return jsonl::ok(id,
                         {{"sigma", implied_vol(r.num("price"), r.num("S"), r.num("K"), r.num("T"),
                                                r.num("rd"), r.num("rf"), option_type(r))}});
    }
    if (fn == "strike") {
        return jsonl::ok(
            id, {{"K", strike_from_delta(r.num("delta"), r.num("S"), r.num("T"), r.num("rd"),
                                         r.num("rf"), r.num("sigma"), option_type(r),
                                         convention(r.str("convention")))}});
    }
    if (fn == "dns_atm") {
        const auto conv = r.flag("pa") ? DeltaConvention::SpotPa : DeltaConvention::Spot;
        return jsonl::ok(id, {{"K", atm_dns_strike(r.num("S"), r.num("T"), r.num("rd"), r.num("rf"),
                                                   r.num("sigma"), conv)}});
    }
    if (fn == "crr") {
        const auto ex = r.flag("american") ? Exercise::American : Exercise::European;
        return jsonl::ok(id,
                         {{"price", binomial_price(r.num("S"), r.num("K"), r.num("T"), r.num("rd"),
                                                   r.num("rf"), r.num("sigma"), option_type(r),
                                                   static_cast<int>(r.num("steps")), ex)}});
    }
    if (fn == "fwd") {
        return jsonl::ok(
            id, {{"forward", cip_forward(r.num("S"), r.num("T"), r.num("rd"), r.num("rf"))}});
    }
    throw std::invalid_argument("unknown fn: " + fn);
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::string id = "null";
        try {
            const jsonl::Request r = jsonl::parse(line);
            if (r.has("id")) id = r.str("id");
            std::cout << handle(r, id) << '\n';
        } catch (const std::exception& e) {
            std::cout << jsonl::err(id, e.what()) << '\n';
        }
        std::cout.flush();
    }
    return 0;
}
