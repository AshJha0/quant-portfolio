// JSON-lines shim over the equity engine for the cross-language
// differential test. Protocol: see python/equity/01-options-pricing/
// tests/test_differential.py. Functions: bs, iv, crr, b76.
#include <exception>
#include <iostream>
#include <string>

#include "eqopt/binomial.hpp"
#include "eqopt/black76.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/implied_vol.hpp"
#include "jsonl.hpp"

namespace {

eqopt::OptionType option_type(const jsonl::Request& r) {
    const std::string& t = r.str("type");
    if (t == "call") return eqopt::OptionType::Call;
    if (t == "put") return eqopt::OptionType::Put;
    throw std::invalid_argument("type must be call or put");
}

std::string handle(const jsonl::Request& r, const std::string& id) {
    const std::string& fn = r.str("fn");
    if (fn == "bs") {
        const auto g = eqopt::bs_greeks(r.num("S"), r.num("K"), r.num("T"), r.num("r"),
                                        r.num("sigma"), r.num("q"), option_type(r));
        return jsonl::ok(id, {{"price", g.price}, {"delta", g.delta}, {"gamma", g.gamma},
                              {"vega", g.vega}, {"theta", g.theta}, {"rho", g.rho},
                              {"vanna", g.vanna}, {"volga", g.volga}});
    }
    if (fn == "iv") {
        const double s = eqopt::implied_vol(r.num("price"), r.num("S"), r.num("K"), r.num("T"),
                                            r.num("r"), r.num("q"), option_type(r));
        return jsonl::ok(id, {{"sigma", s}});
    }
    if (fn == "crr") {
        const auto ex = r.flag("american") ? eqopt::ExerciseStyle::American
                                           : eqopt::ExerciseStyle::European;
        const double p = eqopt::crr_price(r.num("S"), r.num("K"), r.num("T"), r.num("r"),
                                          r.num("sigma"), r.num("q"), option_type(r), ex,
                                          static_cast<int>(r.num("steps")));
        return jsonl::ok(id, {{"price", p}});
    }
    if (fn == "b76") {
        const auto g = eqopt::black76_greeks(r.num("F"), r.num("K"), r.num("T"), r.num("r"),
                                             r.num("sigma"), option_type(r));
        return jsonl::ok(id, {{"price", g.price}, {"delta", g.delta}, {"gamma", g.gamma},
                              {"vega", g.vega}, {"theta", g.theta}, {"rho", g.rho}});
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
