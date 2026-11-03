// The C++ bootstrap against QuantLib 1.44 through the shared fixture
// python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt (the
// same file the Python and Rust tests read). Pillar discount factors to
// 1e-12; every quote reprices to 1e-10 in rate.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "qp/rates_curves.hpp"

namespace r = qp::rates;

namespace {

struct Block {
  std::vector<r::Quote> quotes;
  std::vector<r::XccyBasisQuote> xbs;
  std::vector<std::pair<std::int64_t, double>> expect;
  std::vector<std::pair<std::int64_t, double>> nodes;
};

struct Fixture {
  std::map<std::string, Block> blocks;
  std::vector<std::vector<double>> hwconv;
};

Fixture load() {
  std::ifstream in(QP_RATES_FIXTURE);
  if (!in) throw std::runtime_error("fixture not found");
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    line = line.substr(0, line.find('#'));
    std::istringstream ss(line);
    std::vector<std::string> tok;
    std::string w;
    while (ss >> w) tok.push_back(w);
    if (!tok.empty()) rows.push_back(tok);
  }
  Fixture fx;
  std::string cur;
  auto fl = [](const std::vector<std::string>& a) {
    return r::FloatPeriod{std::stoll(a[1]), std::stoll(a[2]), std::stoll(a[3]), std::stod(a[4])};
  };
  std::size_t i = 0;
  while (i < rows.size()) {
    const auto& t = rows[i++];
    if (t[0] == "build" || t[0] == "given") {
      cur = t[1];
      fx.blocks[cur] = Block{};
    } else if (t[0] == "end") {
    } else if (t[0] == "node") {
      fx.blocks[cur].nodes.emplace_back(std::stoll(t[1]), std::stod(t[2]));
    } else if (t[0] == "expect") {
      fx.blocks[cur].expect.emplace_back(std::stoll(t[1]), std::stod(t[2]));
    } else if (t[0] == "fwd") {
      fx.blocks[cur].quotes.emplace_back(r::ForwardQuote{std::stoll(t[1]), std::stoll(t[2]), std::stod(t[3]), std::stod(t[4])});
    } else if (t[0] == "swap") {
      const auto nf = std::stoul(t[2]);
      const auto nl = std::stoul(t[3]);
      r::SwapQuote s{std::stod(t[1]), {}, {}};
      for (std::size_t k = 0; k < nf; ++k) s.fixed.push_back({std::stoll(rows[i + k][1]), std::stod(rows[i + k][2])});
      for (std::size_t k = 0; k < nl; ++k) s.floating.push_back(fl(rows[i + nf + k]));
      i += nf + nl;
      fx.blocks[cur].quotes.emplace_back(std::move(s));
    } else if (t[0] == "xbs") {
      const auto nf = std::stoul(t[4]);
      const auto nd = std::stoul(t[5]);
      r::XccyBasisQuote q{std::stod(t[1]), std::stoll(t[2]), std::stoll(t[3]), {}, {}};
      for (std::size_t k = 0; k < nf; ++k) q.foreign.push_back(fl(rows[i + k]));
      for (std::size_t k = 0; k < nd; ++k) q.domestic.push_back(fl(rows[i + nf + k]));
      i += nf + nd;
      fx.blocks[cur].xbs.push_back(std::move(q));
    } else if (t[0] == "hwconv") {
      std::vector<double> v;
      for (std::size_t k = 1; k < t.size(); ++k) v.push_back(std::stod(t[k]));
      fx.hwconv.push_back(v);
    } else {
      throw std::runtime_error("unknown fixture line " + t[0]);
    }
  }
  return fx;
}

const Fixture& fixture() {
  static const Fixture fx = load();
  return fx;
}

r::DatedCurve curve_of(const std::vector<std::pair<std::int64_t, double>>& pairs) {
  std::vector<std::int64_t> d;
  std::vector<double> v;
  for (const auto& [a, b] : pairs) {
    d.push_back(a);
    v.push_back(b);
  }
  return {d, v};
}

void check(const r::BootstrapResult& res, const std::vector<std::pair<std::int64_t, double>>& expect) {
  ASSERT_EQ(res.curve.days().size(), expect.size());
  for (std::size_t k = 0; k < expect.size(); ++k) {
    EXPECT_EQ(res.curve.days()[k], expect[k].first);
    EXPECT_NEAR(res.curve.dfs()[k], expect[k].second, 1e-12) << "pillar " << expect[k].first;
  }
  EXPECT_LT(res.max_abs_residual(), 1e-10);
}

}  // namespace

TEST(RatesCurves, OisMatchesQuantLib) {
  const auto& b = fixture().blocks.at("usd_ois");
  const auto res = r::bootstrap_dated(b.quotes);
  EXPECT_EQ(res.diagnostics.size(), 18U);
  check(res, b.expect);
}

TEST(RatesCurves, ProjectionMatchesQuantLib) {
  const auto ois = r::bootstrap_dated(fixture().blocks.at("usd_ois").quotes).curve;
  const auto& b = fixture().blocks.at("term3m");
  check(r::bootstrap_dated(b.quotes, &ois, nullptr), b.expect);
}

TEST(RatesCurves, XccyMatchesQuantLib) {
  const auto usd = curve_of(fixture().blocks.at("usd_ois").expect);
  const auto eur = curve_of(fixture().blocks.at("eur_ois").nodes);
  const auto& b = fixture().blocks.at("eur_xccy");
  EXPECT_EQ(b.expect.size(), 6U);
  check(r::bootstrap_xccy_discount(b.xbs, eur, usd), b.expect);
}

TEST(RatesCurves, HullWhiteConvexityMatchesQuantLib) {
  ASSERT_EQ(fixture().hwconv.size(), 3U);
  for (const auto& h : fixture().hwconv) {
    const double got = r::hull_white_convexity(h[0], h[1], h[2], h[3], h[4]);
    EXPECT_NEAR(got / h[5], 1.0, 1e-13);
  }
  EXPECT_THROW(static_cast<void>(r::hull_white_convexity(96.0, 2.0, 1.0, 0.01, 0.03)), std::invalid_argument);
}

TEST(RatesCurves, ExtrapolationAndErrors) {
  const auto c = r::bootstrap_dated(fixture().blocks.at("usd_ois").quotes).curve;
  const double last = static_cast<double>(c.days().back());
  EXPECT_NEAR(c.forward_rate(last - 10.0), c.forward_rate(last + 1000.0), 1e-14);
  EXPECT_THROW(r::DatedCurve({5, 5}, {0.9, 0.8}), std::invalid_argument);
  EXPECT_THROW(r::DatedCurve({5}, {-0.9}), std::invalid_argument);
  EXPECT_TRUE(std::isnan(c.log_df(-1.0)));
  EXPECT_THROW(static_cast<void>(r::bootstrap_dated({})), std::invalid_argument);
  const r::Quote q = r::ForwardQuote{0, 30, 30.0 / 360.0, 0.05};
  const r::Quote q2 = r::ForwardQuote{10, 30, 20.0 / 360.0, 0.05};
  EXPECT_THROW(static_cast<void>(r::bootstrap_dated({q, q2})), std::invalid_argument);
  const r::Quote crazy = r::ForwardQuote{0, 30, 30.0 / 360.0, -50.0};
  EXPECT_THROW(static_cast<void>(r::bootstrap_dated({crazy})), std::runtime_error);
  EXPECT_THROW(static_cast<void>(r::bootstrap_dated({q}, &c, &c)), std::invalid_argument);
  EXPECT_DOUBLE_EQ(r::futures_rate(95.0, 1e-4), 0.0499);
}
