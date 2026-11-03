// Parity of the C++ credit-index and bond-futures kernels with the Python
// reference values (rust/core/tests/data/credit_bondfut_parity.txt, shared
// with Rust; written by gen_credit_bondfut_parity.py).
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qp/bond_futures.hpp"
#include "qp/credit_index.hpp"

namespace {
std::vector<std::string> words(const std::string& line) {
    std::istringstream in(line);
    std::vector<std::string> out;
    for (std::string s; in >> s;) out.push_back(s);
    return out;
}
std::vector<double> floats(const std::string& line) {
    const auto w = words(line);
    std::vector<double> out;
    for (std::size_t i = 1; i < w.size(); ++i) out.push_back(std::stod(w[i]));
    return out;
}
}  // namespace

TEST(CreditBondFut, MatchesPython) {
    std::ifstream in(QP_CREDIT_BONDFUT_FIXTURE);
    ASSERT_TRUE(in.good()) << QP_CREDIT_BONDFUT_FIXTURE;
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);)
        if (!l.empty() && l[0] != '#') lines.push_back(l);
    int nk = 0, nr = 0, nn = 0, ni = 0, no = 0;
    for (std::size_t i = 0; i < lines.size();) {
        const auto w = words(lines[i]);
        const auto x = floats(lines[i]);
        if (w[0] == "K") {
            // pow differs by a few ulps between libms (glibc vs UCRT)
            const double got = qp::bond_futures::cme_cf(x[0], static_cast<unsigned>(std::stoul(w[2])),
                                                        static_cast<unsigned>(std::stoul(w[3])), w[4] == "1");
            EXPECT_LE(std::fabs(got - x[4]), 1e-14 * std::fabs(x[4])) << lines[i];
            ++nk;
            ++i;
        } else if (w[0] == "R") {
            EXPECT_LE(std::fabs(qp::bond_futures::implied_repo(x[0], x[1], x[2], x[3], x[4]) - x[5]), 1e-15)
                << lines[i];
            ++nr;
            ++i;
        } else if (w[0] == "N") {
            EXPECT_LE(std::fabs(qp::bond_futures::net_basis(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) - x[7]),
                      1e-12)
                << lines[i];
            ++nn;
            ++i;
        } else if (w[0] == "I") {
            std::vector<std::vector<double>> v;
            for (std::size_t k = 1; k <= 8; ++k) v.push_back(floats(lines[i + k]));
            const qp::credit_index::Schedule s{v[4], v[5], v[6], v[7]};
            const auto [pr, rp, d] = qp::credit_index::isda_legs(v[0], v[1], v[2], v[3], x[0], x[1], s, x[2]);
            EXPECT_LE(std::fabs(pr - x[3]), 1e-14) << lines[i];
            EXPECT_LE(std::fabs(rp - x[4]), 1e-13 * std::fabs(x[4])) << lines[i];
            EXPECT_LE(std::fabs(d - x[5]), 1e-15) << lines[i];
            ++ni;
            i += 9;
        } else {
            ASSERT_EQ(w[0], "O");
            const auto nb = static_cast<std::size_t>(x[0]);
            std::vector<std::vector<double>> hdr, ex, fl;
            for (std::size_t b = 0; b < nb; ++b) {
                hdr.push_back(floats(lines[i + 1 + 3 * b]));
                ex.push_back(floats(lines[i + 2 + 3 * b]));
                fl.push_back(floats(lines[i + 3 + 3 * b]));
            }
            std::vector<qp::bond_futures::Bond> bonds;
            for (std::size_t b = 0; b < nb; ++b)
                bonds.push_back({ex[b], fl[b], hdr[b][1], static_cast<unsigned>(hdr[b][0]), hdr[b][2], hdr[b][3]});
            const auto [fair, opt] = qp::bond_futures::delivery_option(bonds, x[1]);
            EXPECT_LE(std::fabs(fair - x[2]), 1e-13 * std::fabs(x[2])) << lines[i];
            EXPECT_LE(std::fabs(opt - x[3]), 1e-11) << lines[i];
            ++no;
            i += 1 + 3 * nb;
        }
    }
    EXPECT_EQ(nk, 588);
    EXPECT_EQ(nr, 3);
    EXPECT_EQ(nn, 9);
    EXPECT_EQ(ni, 12);
    EXPECT_EQ(no, 3);
}

TEST(CreditBondFut, RejectsBadInput) {
    const std::vector<double> t{1.0}, r{0.01}, one{1.0};
    const qp::credit_index::Schedule s{one, one, one, one};
    EXPECT_THROW(qp::credit_index::isda_legs(t, r, t, r, 1.0, 1.0, s, 0.0), std::invalid_argument);
    const std::vector<double> t2{2.0, 1.0}, r2{0.01, 0.01};
    EXPECT_THROW(qp::credit_index::isda_legs(t2, r2, t, r, 0.4, 1.0, s, 0.0), std::invalid_argument);
    EXPECT_THROW(qp::bond_futures::implied_repo(100.0, 100.0, 0.0, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(qp::bond_futures::delivery_option({}, 0.01), std::invalid_argument);
}
