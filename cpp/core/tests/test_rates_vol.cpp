// Parity of the C++ rates-vol port with the Python reference values (the
// fixture rust/core/tests/data/rates_vol_parity.txt, shared with Rust).
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qp/rates_vol.hpp"

namespace rv = qp::rates_vol;

namespace {

std::vector<double> floats(const std::string& line) {
    std::istringstream in(line);
    std::string tag;
    in >> tag;
    std::vector<double> out;
    for (std::string s; in >> s;) out.push_back(std::stod(s));
    return out;
}

}  // namespace

TEST(RatesVol, MatchesPython) {
    std::ifstream in(QP_RATES_VOL_FIXTURE);
    ASSERT_TRUE(in.good()) << QP_RATES_VOL_FIXTURE;
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);)
        if (!l.empty() && l[0] != '#') lines.push_back(l);
    int n_s = 0, n_b = 0;
    for (std::size_t i = 0; i < lines.size();) {
        std::istringstream row(lines[i]);
        std::string tag;
        row >> tag;
        if (tag == "S") {
            double x[8];
            std::string s;
            for (double& v : x) {
                row >> s;
                v = std::stod(s);
            }
            std::string variant;
            row >> variant >> s;
            const double want = std::stod(s);
            const double got =
                rv::sabr_vol(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                             variant == "hagan" ? rv::Variant::hagan : rv::Variant::obloj);
            // 1e-14 relative: pow, log1p and expm1 differ by a few ulps between libms (glibc vs UCRT: 7 ulps seen).
            EXPECT_LE(std::fabs(got - want), 1e-14 * std::fabs(want)) << lines[i];
            ++n_s;
            ++i;
        } else {
            ASSERT_EQ(tag, "B");
            std::string sa, ss, sp, sw;
            row >> sa >> ss >> sp >> sw;
            const auto t = floats(lines[i + 1]);
            const auto d = floats(lines[i + 2]);
            const auto c = floats(lines[i + 3]);
            std::vector<bool> e;
            for (double v : floats(lines[i + 4])) e.push_back(v == 1.0);
            const double got =
                rv::bermudan_tree(std::stod(sa), std::stod(ss), t, d, c, e, sp == "1");
            EXPECT_NEAR(got, std::stod(sw), 1e-13);
            ++n_b;
            i += 5;
        }
    }
    EXPECT_GE(n_s, 80);
    EXPECT_EQ(n_b, 3);
}

TEST(RatesVol, WorkedExampleAndValidation) {
    EXPECT_NEAR(rv::sabr_vol(0.03, 0.03, 5.0, 0.02, 0.5, -0.3, 0.4), 0.121209, 5e-7);
    EXPECT_THROW(rv::sabr_vol(0.03, -0.01, 1.0, 0.02, 0.5, 0.0, 0.3), std::invalid_argument);
    const std::vector<double> t{0.0, 1.0}, d{1.0, 0.97}, c{0.0, 1.0};
    EXPECT_THROW(rv::bermudan_tree(0.0, 0.01, t, d, c, {true, false}, true), std::invalid_argument);
}
