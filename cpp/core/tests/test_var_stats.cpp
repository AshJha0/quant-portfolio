// W20 characterisation corpus for qp/var_stats.hpp (Round 8).
//
// tests/data/var_stats_corpus.txt holds the equity VaR engine's special
// functions and moments on a fixed grid, captured before the two engines'
// stats.cpp files were merged into qp/var_stats.hpp. Every row must
// reproduce within 1e-12 * max(1, |recorded|). A row marked fx_was records
// the FX engine's pre-merge value where it differed by more than that; the
// merged value must be closer to the mpmath value (documented fix).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "qp/var_stats.hpp"

namespace {

double from_hex(const std::string& h) {
    const std::uint64_t u = std::stoull(h, nullptr, 16);
    double v;
    std::memcpy(&v, &u, sizeof v);
    return v;
}

std::vector<double> sample() {
    std::vector<double> s;
    for (int i = 0; i < 257; ++i) {
        s.push_back(std::sin(i * 0.7311) * 1234.5 + ((i * i) % 17) - 3.25);
    }
    return s;
}

// Evaluates one row; `args` are the numeric fields between name and bits.
double eval(const std::string& f, const std::vector<double>& a) {
    namespace v = qp::var;
    if (f == "log_gamma") return v::log_gamma(a[0]);
    if (f == "gamma_p") return v::regularized_gamma_p(a[0], a[1]);
    if (f == "gamma_q") return v::regularized_gamma_q(a[0], a[1]);
    if (f == "normal_cdf") return qp::norm_cdf(a[0]);
    if (f == "normal_pdf") return qp::norm_pdf(a[0]);
    if (f == "normal_ppf") return v::normal_ppf(a[0]);
    if (f == "beta") return v::betainc_reg(a[0], a[1], a[2]);
    if (f == "chi2_sf") return v::chi2_sf(a[0], a[1]);
    if (f == "binom") {
        return v::binomial_cdf(static_cast<int>(a[0]), static_cast<int>(a[1]), a[2]);
    }
    if (f == "t_pdf") return v::student_t_pdf(a[0], a[1]);
    if (f == "t_cdf") return v::student_t_cdf(a[0], a[1]);
    if (f == "t_ppf") return v::student_t_ppf(a[0], a[1]);
    const auto m = v::moments(sample());
    if (f == "mean") return m.mean;
    if (f == "std") return m.stdev;
    if (f == "skew") return m.skewness;
    if (f == "kurt") return m.excess_kurtosis;
    ADD_FAILURE() << "unknown function " << f;
    return 0.0;
}

}  // namespace

TEST(VarStats, CorpusReproduces) {
    std::ifstream in(QP_VAR_STATS_CORPUS);
    ASSERT_TRUE(in) << "cannot open " << QP_VAR_STATS_CORPUS;
    std::string line;
    int rows = 0, fixes = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::vector<std::string> tok;
        for (std::string t; ss >> t;) tok.push_back(t);
        const auto mark = std::find(tok.begin(), tok.end(), "fx_was");
        const std::size_t bits_at = static_cast<std::size_t>(mark - tok.begin()) - 1;
        std::vector<double> args;
        for (std::size_t i = 1; i < bits_at; ++i) args.push_back(std::stod(tok[i]));
        const double want = from_hex(tok[bits_at]);
        const double have = eval(tok[0], args);
        EXPECT_LE(std::abs(have - want), 1e-12 * std::max(1.0, std::abs(want))) << line;
        if (mark != tok.end()) {
            const double old = from_hex(*(mark + 1));
            const double truth = std::stod(*(mark + 3));
            EXPECT_LT(std::abs(have - truth), std::abs(old - truth)) << line;
            ++fixes;
        }
        ++rows;
    }
    EXPECT_EQ(rows, 281);
    EXPECT_EQ(fixes, 1);
}

TEST(VarStats, UpperTailQuantileIsTheMirrorOfTheLowerTail) {
    for (double p : {0.6, 0.9, 0.99, 0.999999}) {
        EXPECT_EQ(qp::var::normal_ppf(p), -qp::var::normal_ppf(1.0 - p)) << p;
        EXPECT_EQ(qp::var::student_t_ppf(p, 5.0), -qp::var::student_t_ppf(1.0 - p, 5.0)) << p;
    }
}

TEST(VarStats, InvalidInputThrows) {
    EXPECT_THROW((void)qp::var::normal_ppf(0.0), std::invalid_argument);
    EXPECT_THROW((void)qp::var::student_t_ppf(0.5, -1.0), std::invalid_argument);
    EXPECT_THROW((void)qp::var::binomial_cdf(1, -1, 0.1), std::invalid_argument);
    EXPECT_THROW((void)qp::var::moments(std::vector<double>{1.0}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(qp::var::binomial_cdf(0, 0, 0.1), 1.0);
}
