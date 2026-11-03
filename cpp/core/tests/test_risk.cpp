// Parity of the C++ risk kernels with the Python reference values (the
// fixture rust/core/tests/data/risk_parity.txt, shared with Rust; written by
// python/equity/04-fixed-income/tools/gen_risk_parity.py). Bound: 1e-12
// relative to the largest entry of each output.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qp/risk.hpp"

namespace rk = qp::risk;

namespace {

struct Row {
    std::string tag;
    std::vector<std::string> rest;
    std::vector<double> nums() const {
        std::vector<double> out;
        for (const auto& s : rest) out.push_back(std::stod(s));
        return out;
    }
};

std::vector<Row> load() {
    std::ifstream in(QP_RISK_FIXTURE);
    if (!in.good()) throw std::runtime_error(QP_RISK_FIXTURE);
    std::vector<Row> rows;
    for (std::string l; std::getline(in, l);) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream s(l);
        Row r;
        s >> r.tag;
        for (std::string t; s >> t;) r.rest.push_back(t);
        rows.push_back(std::move(r));
    }
    return rows;
}

void close(const std::vector<double>& got, const std::vector<double>& want, const char* what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    double scale = 0.0;
    for (double w : want) scale = std::max(scale, std::fabs(w));
    for (std::size_t k = 0; k < got.size(); ++k)
        EXPECT_LE(std::fabs(got[k] - want[k]), 1e-12 * scale)
            << what << "[" << k << "] " << got[k] << " vs " << want[k];
}

}  // namespace

TEST(Risk, MatchesPython) {
    const auto rows = load();
    int n_curve = 0, n_grid = 0, n_ta = 0;
    for (std::size_t i = 0; i < rows.size();) {
        const auto& r = rows[i];
        if (r.tag == "N") {
            const auto n = static_cast<std::size_t>(std::stoul(r.rest[0]));
            const auto jac = rows[i + 1].nums(), gz = rows[i + 2].nums(), gamz = rows[i + 3].nums();
            std::vector<double> hs;
            for (std::size_t k = 0; k < n; ++k) {
                const auto h = rows[i + 4 + k].nums();
                hs.insert(hs.end(), h.begin(), h.end());
            }
            const std::size_t b = i + 4 + n;
            const auto gq_w = rows[b].nums(), dv_w = rows[b + 1].nums(), gam_w = rows[b + 2].nums();
            const auto gq = rk::node_to_quote_risk(jac, n, gz);
            close(gq, gq_w, "quote risk");
            close(rk::par_dv01(gq), dv_w, "par DV01");
            close(rk::quote_gamma(jac, n, gamz, gq_w, hs), gam_w, "quote gamma");
            const std::vector<double> zero(n, 0.0);
            const auto shortcut = rk::quote_gamma(jac, n, gamz, zero, hs);
            double worst = 0.0, scale = 0.0;
            for (std::size_t k = 0; k < shortcut.size(); ++k) {
                worst = std::max(worst, std::fabs(shortcut[k] - gam_w[k]));
                scale = std::max(scale, std::fabs(gam_w[k]));
            }
            EXPECT_GT(worst, 0.01 * scale) << "the shortcut without the curvature term must stay wrong";
            ++n_curve;
            i = b + 3;
        } else if (r.tag == "GRID") {
            const auto d = static_cast<std::size_t>(std::stoul(r.rest[0]));
            std::vector<std::vector<double>> axes;
            for (std::size_t k = 0; k < d; ++k) axes.push_back(rows[i + 1 + k].nums());
            const rk::VolGrid grid(axes, rows[i + 1 + d].nums());
            i += 2 + d;
            std::vector<std::pair<std::vector<double>, double>> pts;
            while (rows[i].tag == "PT") {
                const auto p = rows[i].nums();
                std::vector<double> c(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(d));
                close(grid.weights(c), rows[i + 1].nums(), "weights");
                close({grid.vol(c)}, rows[i + 2].nums(), "vol");
                pts.emplace_back(c, p[d]);
                i += 3;
            }
            ASSERT_EQ(rows[i].tag, "VEGA");
            auto pv = [&](const rk::VolGrid& g) {
                double s = 0.0;
                for (const auto& [c, n] : pts) {
                    const double v = g.vol(c);
                    s += n * v * v;
                }
                return s;
            };
            close(rk::vega_grid(pv, grid, 1e-4), rows[i].nums(), "vega");
            ++n_grid;
            ++i;
        } else {
            ASSERT_EQ(r.tag, "TA");
            const bool with_g = r.rest[1] == "1";
            const auto d = rows[i + 1].nums(), m = rows[i + 2].nums();
            std::size_t k = i + 3;
            std::vector<double> g;
            if (with_g) g = rows[k++].nums();
            close(rk::taylor_attribution(d, m, g), rows[k].nums(), "attribution");
            const auto ex_in = rows[k + 1].nums();
            const auto ex = rk::explain(ex_in[0], rows[k + 2].nums(), d, m, g, ex_in[1]);
            close({ex.rtpl(), ex.unexplained(), ex.total()}, rows[k + 3].nums(), "explain");
            ++n_ta;
            i = k + 4;
        }
    }
    EXPECT_EQ(n_curve, 1);
    EXPECT_EQ(n_grid, 2);
    EXPECT_EQ(n_ta, 3);
}

TEST(Risk, RefusesSingularJacobianAndBadInputs) {
    const std::vector<double> j{1.0, 2.0, 2.0, 4.0}, g{1.0, 1.0};
    EXPECT_THROW(rk::node_to_quote_risk(j, 2, g), std::invalid_argument);
    EXPECT_THROW(rk::VolGrid({{1.0}}, {0.1}), std::invalid_argument);
    EXPECT_THROW(rk::VolGrid({{1.0, 1.0}, {0.0}}, {0.1, 0.1}), std::invalid_argument);
    const std::vector<double> one{1.0};
    EXPECT_THROW(rk::explain(std::nan(""), {}, one, one), std::invalid_argument);
}

TEST(Risk, RoundTripAndWorkedExample) {
    const std::vector<double> jac{1.02, 0.0, 0.1, 0.0, 2.0, 0.3, 0.05, 0.2, 0.9}, gz{1.0, -2.0, 3.5};
    const auto back = rk::quote_to_node_risk(jac, 3, rk::node_to_quote_risk(jac, 3, gz));
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(back[k], gz[k], 1e-14);
    const std::vector<double> d{100.0}, m{0.1}, gm{40.0}, t{-1.0};
    EXPECT_NEAR(rk::explain(10.0, t, d, m, gm).rtpl(), 9.2, 1e-12);
}

TEST(Risk, FiniteDifferenceKernelsMatchAnalyticDerivatives) {
    const std::vector<double> x{0.3, 0.7};
    auto f = [](std::span<const double> v) { return std::exp(v[0]) * std::sin(v[1]) + v[0] * v[0] * v[1]; };
    const auto jac =
        rk::fd_jacobian([&](std::span<const double> v) { return std::vector<double>{f(v)}; }, x, 1e-6);
    EXPECT_NEAR(jac[0], std::exp(0.3) * std::sin(0.7) + 2 * 0.3 * 0.7, 1e-8);
    EXPECT_NEAR(jac[1], std::exp(0.3) * std::cos(0.7) + 0.09, 1e-8);
    const auto h = rk::cross_gamma(f, x, 1e-4);
    EXPECT_NEAR(h[0], std::exp(0.3) * std::sin(0.7) + 1.4, 1e-6);
    EXPECT_NEAR(h[1], std::exp(0.3) * std::cos(0.7) + 0.6, 1e-6);
    EXPECT_EQ(h[1], h[2]);
    EXPECT_NEAR(h[3], -std::exp(0.3) * std::sin(0.7), 1e-6);
}

TEST(Risk, LinearVegaEqualsWeightsIncludingZeroVolNode) {
    const rk::VolGrid g({{1.0, 2.0}, {0.0, 1.0, 2.0}}, {0.0, 0.2, 0.3, 0.25, 0.1, 0.4});
    const std::vector<double> c{1.5, 0.25};
    const auto w = g.weights(c);
    const auto v = rk::vega_grid([&](const rk::VolGrid& gr) { return gr.vol(c); }, g, 1e-4);
    for (std::size_t k = 0; k < w.size(); ++k) EXPECT_NEAR(v[k], w[k], 1e-10);
}
