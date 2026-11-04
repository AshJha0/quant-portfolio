// Parity of the C++ FRTB and collateral kernels with the Python reference
// values (the fixture rust/core/tests/data/frtb_collateral_parity.txt, shared
// with Rust; written by
// python/equity/04-fixed-income/tools/gen_frtb_collateral_parity.py). Bound:
// 1e-12 relative to the largest entry of each output.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qp/collateral.hpp"
#include "qp/frtb.hpp"

namespace fr = qp::frtb;
namespace co = qp::collateral;

namespace {

struct Row {
    std::string tag;
    std::vector<std::string> rest;
    std::vector<double> nums(std::size_t from = 0) const {
        std::vector<double> out;
        for (std::size_t i = from; i < rest.size(); ++i) out.push_back(std::stod(rest[i]));
        return out;
    }
};

std::vector<Row> load() {
    std::ifstream in(QP_FRTB_COLLATERAL_FIXTURE);
    if (!in.good()) throw std::runtime_error(QP_FRTB_COLLATERAL_FIXTURE);
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

std::vector<double> charges(const fr::Charges& c) { return {c.medium, c.high, c.low, c.charge}; }

std::size_t position(const std::vector<int>& ids, int b) {
    return static_cast<std::size_t>(std::find(ids.begin(), ids.end(), b) - ids.begin());
}

co::Margin margin(const Row& r) {
    const auto x = r.nums(1);
    if (r.rest[0] == "S") return co::Margin::scalar(x[0]);
    if (r.rest[0] == "T") return co::Margin::by_time(x);
    if (r.rest[0] == "P") return co::Margin::by_path(x);
    return co::Margin::full(x);
}

}  // namespace

TEST(Frtb, AggregateMatchesPython) {
    const auto rows = load();
    int cases = 0;
    for (std::size_t i = 0; i < rows.size();) {
        if (rows[i].tag != "AGG") {
            ++i;
            continue;
        }
        const auto nb = static_cast<std::size_t>(std::stoul(rows[i].rest[0]));
        ++i;
        std::map<int, fr::BucketInput> buckets;
        std::vector<int> ids;
        for (std::size_t b = 0; b < nb; ++b, i += 3) {
            const int id = std::stoi(rows[i].rest[0]);
            ids.push_back(id);
            buckets[id] = {rows[i + 1].nums(), rows[i + 2].nums(), rows[i].rest[1] == "1"};
        }
        const auto gam = rows[i].nums();
        const auto got = fr::aggregate(
            buckets, [&](int b, int c) { return gam[position(ids, b) * nb + position(ids, c)]; });
        close(charges(got), rows[i + 1].nums(), "aggregate");
        i += 2;
        ++cases;
    }
    EXPECT_EQ(cases, 7);
}

TEST(Frtb, CurvatureMatchesPython) {
    const auto rows = load();
    int cases = 0;
    for (std::size_t i = 0; i < rows.size();) {
        if (rows[i].tag != "CURV") {
            ++i;
            continue;
        }
        const auto nb = static_cast<std::size_t>(std::stoul(rows[i].rest[0]));
        ++i;
        std::map<int, std::vector<fr::CvrRow<std::string>>> buckets;
        std::map<int, std::pair<std::vector<std::string>, std::vector<double>>> rho;
        std::set<int> other;
        std::vector<int> ids;
        for (std::size_t b = 0; b < nb; ++b) {
            const int id = std::stoi(rows[i].rest[0]);
            ids.push_back(id);
            if (rows[i].rest[1] == "1") other.insert(id);
            const auto n = static_cast<std::size_t>(std::stoul(rows[i].rest[2]));
            rho[id] = {rows[i + 1].rest, rows[i + 2].nums()};
            for (std::size_t r = 0; r < n; ++r) {
                const auto& row = rows[i + 3 + r];
                buckets[id].push_back(
                    {row.rest[0], std::stod(row.rest[1]), std::stod(row.rest[2])});
            }
            i += 3 + n;
        }
        const auto gam = rows[i].nums();
        const auto delta_rho = [&](int b, const std::string& k, const std::string& l) {
            const auto& [keys, m] = rho.at(b);
            const auto at = [&](const std::string& x) {
                return static_cast<std::size_t>(std::find(keys.begin(), keys.end(), x) -
                                                keys.begin());
            };
            return m[at(k) * keys.size() + at(l)];
        };
        const auto got = fr::curvature_charge(
            buckets, delta_rho,
            [&](int b, int c) { return gam[position(ids, b) * nb + position(ids, c)]; }, other);
        close(charges(got), rows[i + 1].nums(), "curvature");
        i += 2;
        ++cases;
    }
    EXPECT_EQ(cases, 6);
}

TEST(Frtb, NetCvrMatchesPython) {
    int cases = 0;
    for (const auto& r : load()) {
        if (r.tag != "CVR") continue;
        const auto x = r.nums();
        const auto [p, m] = fr::net_cvr(x[0], x[1], x[2], x[3], x[4]);
        close({p, m}, {x[5], x[6]}, "net_cvr");
        ++cases;
    }
    EXPECT_EQ(cases, 6);
}

TEST(Collateral, LaggedGridMatchesPython) {
    const auto rows = load();
    int cases = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].tag != "LG") continue;
        const auto x = rows[i].nums();
        const std::vector<double> t(x.begin() + 1, x.end());
        close(co::lagged_grid(t, x[0]), rows[i + 1].nums(), "lagged_grid");
        ++cases;
    }
    EXPECT_EQ(cases, 5);
}

TEST(Collateral, MatchesPython) {
    const auto rows = load();
    int cases = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].tag != "COL") continue;
        const auto paths = static_cast<std::size_t>(std::stoul(rows[i].rest[0]));
        const auto n = static_cast<std::size_t>(std::stoul(rows[i].rest[1]));
        const auto t = rows[i + 1].nums();
        const auto v = rows[i + 2].nums();
        const auto c = rows[i + 3].nums();
        co::Csa csa;
        csa.threshold_cpty = c[0];
        csa.threshold_bank = c[1];
        csa.mta = c[2];
        csa.mpor = c[3];
        csa.initial = c[4];
        csa.im_received = margin(rows[i + 4]);
        csa.im_posted = margin(rows[i + 5]);
        close(co::collateral_balance(v, paths, n, csa), rows[i + 6].nums(), "balance");
        std::vector<std::size_t> want;
        for (const auto& s : rows[i + 7].rest)
            want.push_back(static_cast<std::size_t>(std::stoul(s)));
        EXPECT_EQ(co::lag_index(t, csa.mpor), want);
        const auto e = co::collateralised_exposure(t, v, paths, csa);
        close(e.collateral, rows[i + 8].nums(), "collateral");
        close(e.exposure, rows[i + 9].nums(), "exposure");
        close(e.negative_exposure, rows[i + 10].nums(), "negative");
        ++cases;
    }
    EXPECT_EQ(cases, 11);
}

TEST(Frtb, DocstringCases) {
    const double r = 0.35 * 0.999;
    const std::map<int, fr::BucketInput> two{{3, {{50.0, -50.0}, {1.0, r, r, 1.0}, false}}};
    EXPECT_NEAR(fr::aggregate(two, [](int, int) { return 0.0; }).medium, 57.02, 5e-3);
    const std::map<int, std::vector<fr::CvrRow<int>>> one{{5, {{0, 12.0, 9.0}}}};
    const auto c =
        fr::curvature_charge(one, [](int, int, int) { return 0.0; }, [](int, int) { return 0.0; });
    EXPECT_EQ(c.charge, 12.0);
}

TEST(Frtb, EmptyAndOtherBuckets) {
    const auto zero = [](int, int) { return 0.0; };
    EXPECT_EQ(fr::aggregate(std::map<int, fr::BucketInput>{}, zero).charge, 0.0);
    const std::map<int, fr::BucketInput> other{{16, {{3.0, -4.0}, {}, true}}};
    EXPECT_EQ(fr::aggregate(other, zero).charge, 7.0);
    const std::map<int, std::vector<fr::CvrRow<int>>> none;
    EXPECT_EQ(fr::curvature_charge(none, [](int, int, int) { return 0.0; }, zero).charge, 0.0);
}

TEST(Collateral, DocstringCases) {
    const std::vector<double> t{0.0, 1.0, 2.0}, v{0.0, 10.0, 4.0};
    co::Csa csa;
    csa.threshold_cpty = 3.0;
    EXPECT_EQ(co::collateralised_exposure(t, v, 1, csa).exposure,
              (std::vector<double>{0.0, 3.0, 3.0}));
    EXPECT_EQ(co::lagged_grid(t, 0.25), (std::vector<double>{0.0, 0.75, 1.0, 1.75, 2.0}));
}

TEST(Collateral, InfiniteThresholdIsTheUncollateralisedLimit) {
    const std::vector<double> t{0.0, 1.0}, v{2.0, -3.0};
    co::Csa csa;
    csa.threshold_cpty = csa.threshold_bank = std::numeric_limits<double>::infinity();
    const auto e = co::collateralised_exposure(t, v, 1, csa);
    EXPECT_EQ(e.exposure, (std::vector<double>{2.0, 0.0}));
    EXPECT_EQ(e.negative_exposure, (std::vector<double>{0.0, -3.0}));
}

TEST(FrtbCollateral, BadInputIsRefused) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    using V = std::vector<double>;
    EXPECT_THROW((void)fr::within_bucket(V{1.0, 2.0}, V{1.0}), std::invalid_argument);
    EXPECT_THROW((void)fr::within_bucket(V{nan}, V{1.0}), std::invalid_argument);
    EXPECT_THROW((void)fr::across_buckets(V{1.0}, V{1.0, 2.0}, V{0.0}), std::invalid_argument);
    const std::map<int, fr::BucketInput> two{{1, {{1.0}, {1.0}, false}},
                                             {2, {{1.0}, {1.0}, false}}};
    EXPECT_THROW((void)fr::aggregate(two, [&](int, int) { return nan; }), std::invalid_argument);
    const std::map<int, std::vector<fr::CvrRow<int>>> bad{{1, {{0, nan, 0.0}}}};
    EXPECT_THROW((void)fr::curvature_charge(
                     bad, [](int, int, int) { return 0.0; }, [](int, int) { return 0.0; }),
                 std::invalid_argument);

    co::Csa neg;
    neg.mta = -1.0;
    EXPECT_THROW((void)co::collateral_balance(V{1.0}, 1, 1, neg), std::invalid_argument);
    co::Csa nan_mpor;
    nan_mpor.mpor = nan;
    EXPECT_THROW((void)co::collateral_balance(V{1.0}, 1, 1, nan_mpor), std::invalid_argument);
    const co::Csa ok;
    EXPECT_THROW((void)co::collateral_balance(V{1.0}, 2, 1, ok), std::invalid_argument);
    EXPECT_THROW((void)co::collateralised_exposure(V{}, V{}, 0, ok), std::invalid_argument);
    EXPECT_THROW((void)co::collateralised_exposure(V{1.0, 1.0}, V{0.0, 0.0}, 1, ok),
                 std::invalid_argument);
    co::Csa im;
    im.im_received = co::Margin::scalar(-1.0);
    EXPECT_THROW((void)co::collateralised_exposure(V{0.0}, V{0.0}, 1, im), std::invalid_argument);
    co::Csa shortm;
    shortm.im_posted = co::Margin::by_time({1.0});
    EXPECT_THROW((void)co::collateralised_exposure(V{0.0, 1.0}, V{0.0, 0.0}, 1, shortm),
                 std::invalid_argument);
    EXPECT_THROW((void)co::lagged_grid(V{}, 0.1), std::invalid_argument);
    EXPECT_THROW((void)co::lag_index(V{1.0, 0.0}, 0.1), std::invalid_argument);
}
