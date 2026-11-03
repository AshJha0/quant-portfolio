// W19 in C++: adjoint Greeks against closed-form Greeks and central bumps,
// the same oracles as python/core/tests/test_aad_python.py and
// rust/aad/tests/aad.rs. A plain executable (exit code 0 = pass) so it runs
// without a test framework; CTest registers it.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "qp/aad.hpp"

#ifndef QP_AAD_FIXTURE
#error "QP_AAD_FIXTURE must point at dated_bootstrap_usd.txt"
#endif

namespace {

using namespace qp::aad;
namespace rc = qp::rates;

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

bool close(double a, double b, double rel, double floor) {
    return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b)) + floor;
}

std::string pair_str(double a, double b) {
    char buf[96];
    std::snprintf(buf, sizeof buf, " %.17g vs %.17g", a, b);
    return buf;
}

double central(const std::function<double(const std::vector<double>&)>& f,
               const std::vector<double>& x, std::size_t i, double h) {
    auto up = x, dn = x;
    up[i] += h;
    dn[i] -= h;
    return (f(up) - f(dn)) / (2.0 * h);
}

const std::vector<double> kBase{100.0, 100.0, 1.0, 0.05, 0.02, 0.20};

void test_tape() {
    Tape t;
    const Var a = t.var(1.3), b = t.var(0.7);
    const Var y =
        (a * b - a / b + 2.0 / b + (3.0 - b) - (-a)) * exp(a) + log(b) + sqrt(a) + ncdf(a - b);
    const auto g = t.gradient(y, {a, b});
    auto f = [](const std::vector<double>& v) {
        const double x0 = v[0], x1 = v[1];
        return (x0 * x1 - x0 / x1 + 2.0 / x1 + 3.0 - x1 + x0) * std::exp(x0) + std::log(x1) +
               std::sqrt(x0) + qp::norm_cdf(x0 - x1);
    };
    check(close(y.val, f({1.3, 0.7}), 1e-15, 0.0), "tape value");
    for (std::size_t i = 0; i < 2; ++i)
        check(close(g[i], central(f, {1.3, 0.7}, i, 1e-6), 1e-8, 0.0), "tape partial");

    Tape r;
    const Var x = r.var(2.0);
    const std::size_t m = r.mark();
    r.zero_adjoints();
    for (int i = 0; i < 3; ++i) {
        r.backward(x * x, 1.0, m);
        r.rewind(m);
    }
    check(close(r.adjoint(x), 12.0, 1e-15, 0.0) && r.size() == m, "rewind keeps input adjoints");

    Tape t1, t2;
    const Var z = t2.var(1.0);
    bool threw = false;
    try {
        t1.backward(z * 2.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "foreign output throws");
}

// Closed-form Greeks (Hull 2022, ch. 19), written out independently.
std::vector<double> analytic(double s, double k, double t, double r, double q, double sig,
                             bool call) {
    const double sd = sig * std::sqrt(t);
    const double d1 = (std::log(s / k) + (r - q + 0.5 * sig * sig) * t) / sd, d2 = d1 - sd;
    const double dq = std::exp(-q * t), dr = std::exp(-r * t);
    const double pdf = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * std::numbers::pi);
    const double vega = s * dq * pdf * std::sqrt(t), c = -s * dq * pdf * sig / (2.0 * std::sqrt(t));
    if (call) {
        const double n1 = qp::norm_cdf(d1), n2 = qp::norm_cdf(d2);
        return {s * dq * n1 - k * dr * n2, dq * n1,          vega,
                k * t * dr * n2,           -s * t * dq * n1, c + q * s * dq * n1 - r * k * dr * n2};
    }
    const double m1 = qp::norm_cdf(-d1), m2 = qp::norm_cdf(-d2);
    return {k * dr * m2 - s * dq * m1, -dq * m1,        vega,
            -k * t * dr * m2,          s * t * dq * m1, c - q * s * dq * m1 + r * k * dr * m2};
}

void test_bs() {
    struct Case {
        double s, k, t, sig;
    };
    for (bool call : {true, false}) {
        for (const Case& c :
             {Case{100, 100, 1.0, 0.2}, Case{80, 100, 0.25, 0.35}, Case{130, 100, 3.0, 0.15}}) {
            const Greeks a = bs_greeks(c.s, c.k, c.t, 0.05, 0.02, c.sig, call);
            const auto e = analytic(c.s, c.k, c.t, 0.05, 0.02, c.sig, call);
            const double got[6] = {a.price, a.delta, a.vega, a.rho, a.rho_q, -a.d_t};
            for (std::size_t i = 0; i < 6; ++i)
                check(close(got[i], e[i], 1e-12, 1e-14), "bs analytic" + pair_str(got[i], e[i]));
        }
    }
    const auto& b = kBase;
    const Greeks a = bs_greeks(b[0], b[1], b[2], b[3], b[4], b[5], true);
    auto f = [](const std::vector<double>& v) {
        return bs_price(v[0], v[1], v[2], v[3], v[4], v[5], 1.0);
    };
    const double got[6] = {a.delta, a.d_k, a.d_t, a.rho, a.rho_q, a.vega};
    for (std::size_t i = 0; i < 6; ++i) {
        const double bump = central(f, b, i, 1e-5 * std::max(1.0, std::abs(b[i])));
        check(close(got[i], bump, 1e-6, 1e-9), "bs bump" + pair_str(got[i], bump));
    }
    check(std::abs(a.price - 9.227005508154036) < 1e-12,
          "bs worked example (same as Python and Rust)");
}

std::vector<double> normals(std::size_t n, std::uint64_t seed) {
    // Box-Muller on mt19937_64 (the library's normal_distribution is not
    // specified bit for bit across standard libraries).
    std::mt19937_64 g(seed);
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; i += 2) {
        const double u1 = (static_cast<double>(g() >> 11) + 0.5) * 0x1.0p-53;
        const double u2 = (static_cast<double>(g() >> 11) + 0.5) * 0x1.0p-53;
        const double rad = std::sqrt(-2.0 * std::log(u1)), th = 2.0 * std::numbers::pi * u2;
        z[i] = rad * std::cos(th);
        if (i + 1 < n) z[i + 1] = rad * std::sin(th);
    }
    return z;
}

void test_mc() {
    const std::size_t steps = 12;
    const auto z = normals(2000 * steps, 20261103);
    const auto& b = kBase;
    for (Payoff payoff : {Payoff::European, Payoff::Asian}) {
        const Greeks a = mc_greeks(b[0], b[1], b[2], b[3], b[4], b[5], z, steps, payoff);
        auto f = [&](const std::vector<double>& v) {
            return mc_price(v[0], v[1], v[2], v[3], v[4], v[5], z, steps, payoff);
        };
        check(close(a.price, f(b), 1e-12, 0.0), "mc price agrees");
        const double got[6] = {a.delta, a.d_k, a.d_t, a.rho, a.rho_q, a.vega};
        for (std::size_t i = 0; i < 6; ++i) {
            const double bump = central(f, b, i, 1e-6 * std::max(1.0, std::abs(b[i])));
            check(close(got[i], bump, 1e-6, 1e-9), "mc bump" + pair_str(got[i], bump));
        }
    }
    const auto z1 = normals(20000, 7);
    const Greeks a = mc_greeks(b[0], b[1], b[2], b[3], b[4], b[5], z1, 1, Payoff::European);
    const Greeks c = bs_greeks(b[0], b[1], b[2], b[3], b[4], b[5], true);
    check(std::abs(a.price - c.price) < 3.0 * a.std_error, "mc within 3 se of BS");
    check(std::abs(a.delta - c.delta) < 0.01 && std::abs(a.vega - c.vega) / c.vega < 0.03,
          "mc Greeks near BS");
}

struct Ois {
    std::vector<rc::Quote> quotes;
    std::vector<std::int64_t> days;
    std::vector<double> dfs;
};

Ois usd_ois() {
    std::ifstream in(QP_AAD_FIXTURE);
    std::vector<std::vector<std::string>> rows;
    for (std::string line; std::getline(in, line);) {
        line = line.substr(0, line.find('#'));
        std::istringstream ss(line);
        std::vector<std::string> r;
        for (std::string w; ss >> w;) r.push_back(w);
        if (!r.empty()) rows.push_back(r);
    }
    Ois o;
    bool on = false;
    for (std::size_t i = 0; i < rows.size();) {
        const auto& r = rows[i++];
        if (r[0] == "build") {
            on = r[1] == "usd_ois";
        } else if (!on) {
            continue;
        } else if (r[0] == "end") {
            break;
        } else if (r[0] == "expect") {
            o.days.push_back(std::stoll(r[1]));
            o.dfs.push_back(std::stod(r[2]));
        } else if (r[0] == "fwd") {
            o.quotes.emplace_back(rc::ForwardQuote{std::stoll(r[1]), std::stoll(r[2]),
                                                   std::stod(r[3]), std::stod(r[4])});
        } else if (r[0] == "swap") {
            const std::size_t nf = std::stoul(r[2]), nl = std::stoul(r[3]);
            rc::SwapQuote s{std::stod(r[1]), {}, {}};
            for (std::size_t j = 0; j < nf; ++j)
                s.fixed.push_back({std::stoll(rows[i + j][1]), std::stod(rows[i + j][2])});
            for (std::size_t j = nf; j < nf + nl; ++j) {
                const auto& a = rows[i + j];
                s.floating.push_back(
                    {std::stoll(a[1]), std::stoll(a[2]), std::stoll(a[3]), std::stod(a[4])});
            }
            i += nf + nl;
            o.quotes.emplace_back(std::move(s));
        }
    }
    return o;
}

// The periods of the SOFR OIS 7Y quote, payer, 4 % fixed, 1m notional.
Swap test_swap(const Ois& o) {
    for (const auto& q : o.quotes) {
        if (const auto* s = std::get_if<rc::SwapQuote>(&q); s && s->fixed.size() == 7) {
            return {s->fixed, s->floating, 0.04, 1e6, true};
        }
    }
    throw std::runtime_error("no 7Y quote");
}

void test_swap_greeks() {
    const Ois o = usd_ois();
    check(o.quotes.size() == 18 && o.days.size() == 18, "fixture parsed");
    const Swap sw = test_swap(o);
    const SwapGreeks pg = swap_pillar_greeks(o.days, o.dfs, sw);
    check(close(pg.pv, swap_pv_from_dfs(o.days, o.dfs, sw), 1e-15, 0.0), "swap pv agrees");
    std::vector<double> x;
    for (double d : o.dfs) x.push_back(std::log(d));
    auto f = [&](const std::vector<double>& u) { return swap_pv(o.days, u, sw); };
    for (std::size_t j = 0; j < x.size(); ++j) {
        const double bump = central(f, x, j, 1e-6);
        check(close(pg.d_log_df[j], bump, 1e-6, 1e-6),
              "pillar bump" + pair_str(pg.d_log_df[j], bump));
    }
    Swap rec = sw;
    rec.payer = false;
    check(swap_pv_from_dfs(o.days, o.dfs, rec) == -pg.pv, "receiver is minus payer");

    const auto base = rc::bootstrap_dated(o.quotes).curve;
    check(base.days() == o.days, "bootstrap pillars");
    const SwapGreeks qg = swap_quote_greeks(o.quotes, base.days(), base.dfs(), sw);
    const double h = 1e-6;
    for (std::size_t k = 0; k < o.quotes.size(); ++k) {
        double pv[2];
        for (int m = 0; m < 2; ++m) {
            auto qs = o.quotes;
            std::visit([&](auto& q) { q.rate += m == 0 ? h : -h; }, qs[k]);
            const auto c = rc::bootstrap_dated(qs).curve;
            pv[m] = swap_pv_from_dfs(c.days(), c.dfs(), sw);
        }
        const double bump = (pv[0] - pv[1]) / (2.0 * h);
        check(close(qg.d_quote[k], bump, 1e-6, 1e-4), "quote bump" + pair_str(qg.d_quote[k], bump));
    }
    bool threw = false;
    try {
        (void)swap_quote_greeks({o.quotes.begin() + 1, o.quotes.end()}, base.days(), base.dfs(),
                                sw);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "quote count mismatch throws");
}

}  // namespace

int main() {
    test_tape();
    test_bs();
    test_mc();
    test_swap_greeks();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
