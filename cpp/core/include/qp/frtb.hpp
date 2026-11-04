// FRTB sensitivities-based aggregation and the curvature charge.
//
// C++ port of aggregate in python/core/src/qp_core/capital/frtb_classes.py
// (with scenario_corr, within_bucket and across_buckets from frtb_sbm.py) and
// of curvature_charge and net_cvr in frtb_curvature.py; twin of
// rust/core/src/frtb.rs. Plain idea, formal statement, worked examples and
// sources (BCBS d457, MAR21) are in the Python docstrings.
//
// Parity: cpp/core/tests/test_frtb_collateral.cpp reproduces the Python
// reference values (rust/core/tests/data/frtb_collateral_parity.txt, shared
// with Rust) to 1e-12 relative.
//
// Three differences from the Python, none of which changes a value on valid
// input. NaN and infinite inputs are rejected here. Factor keys are sorted with
// operator< here and by repr in Python, which only changes the order of a sum.
// Python 3.12 sum is compensated and the sums here are plain, so if K+ and K-
// agree to the last bit by accident rather than exactly, the side chosen may
// differ; exact ties, the ones the rule is written for, agree.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qp::frtb {

// MAR21.6 correlation scenarios, in the order Python runs them.
enum class Scenario { Medium, High, Low };
inline constexpr std::array<Scenario, 3> kScenarios{Scenario::Medium, Scenario::High,
                                                    Scenario::Low};

// Charge under each scenario and charge, the largest of the three.
struct Charges {
    double medium = 0.0;
    double high = 0.0;
    double low = 0.0;
    double charge = 0.0;
};

// Correlation under a MAR21.6 scenario.
[[nodiscard]] inline double scenario_corr(double rho, Scenario scenario) noexcept {
    switch (scenario) {
        case Scenario::High:
            return std::min(1.25 * rho, 1.0);
        case Scenario::Low:
            return std::max(2.0 * rho - 1.0, 0.75 * rho);
        case Scenario::Medium:
            break;
    }
    return rho;
}

namespace detail {

inline void require_finite(std::span<const double> values, const char* name) {
    for (double v : values)
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(name) + " must be finite");
}

// (v' M) v for a row-major n x n matrix.
[[nodiscard]] inline double quad(std::span<const double> v, std::span<const double> m) noexcept {
    const std::size_t n = v.size();
    double out = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double y = 0.0;
        for (std::size_t i = 0; i < n; ++i) y += v[i] * m[i * n + j];
        out += y * v[j];
    }
    return out;
}

[[nodiscard]] inline double sum(std::span<const double> v) noexcept {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

[[nodiscard]] inline double psi(double a, double b) noexcept {
    return (a < 0.0 && b < 0.0) ? 0.0 : 1.0;
}

[[nodiscard]] inline Charges from_scenarios(const std::array<double, 3>& r) noexcept {
    return Charges{r[0], r[1], r[2], std::max({r[0], r[1], r[2]})};
}

// rho with the scenario applied off the diagonal. A wrong length is passed
// through for within_bucket to refuse.
[[nodiscard]] inline std::vector<double> scenario_matrix(std::span<const double> rho, std::size_t n,
                                                         Scenario sc) {
    std::vector<double> out(rho.begin(), rho.end());
    for (std::size_t idx = 0; idx < out.size(); ++idx)
        if (n == 0 || idx / n != idx % n) out[idx] = scenario_corr(out[idx], sc);
    return out;
}

template <class R>
[[nodiscard]] double k_side(std::span<const double> c, const R& rho) {
    double q = 0.0;
    for (double x : c) {
        const double p = std::max(x, 0.0);
        q += p * p;
    }
    for (std::size_t i = 0; i < c.size(); ++i)
        for (std::size_t j = 0; j < c.size(); ++j)
            if (i != j) q += rho(i, j) * c[i] * c[j] * psi(c[i], c[j]);
    return std::sqrt(std::max(q, 0.0));
}

}  // namespace detail

// K_b = sqrt(max(0, WS' R WS)) (MAR21.4(4)). rho is row-major and its diagonal
// is used as given.
[[nodiscard]] inline double within_bucket(std::span<const double> ws, std::span<const double> rho) {
    if (rho.size() != ws.size() * ws.size())
        throw std::invalid_argument("rho must be square and match ws");
    detail::require_finite(ws, "ws");
    detail::require_finite(rho, "rho");
    return std::sqrt(std::max(0.0, detail::quad(ws, rho)));
}

// sqrt(sum K_b^2 + sum_{b != c} gamma S_b S_c), with the clipped S_b fallback
// when the radicand is negative (MAR21.4(5)). gamma is row-major; its diagonal
// is ignored.
[[nodiscard]] inline double across_buckets(std::span<const double> k, std::span<const double> s,
                                           std::span<const double> gamma) {
    const std::size_t n = k.size();
    if (s.size() != n || gamma.size() != n * n)
        throw std::invalid_argument("gamma must be square and match k and s");
    detail::require_finite(k, "k");
    detail::require_finite(s, "s");
    detail::require_finite(gamma, "gamma");
    std::vector<double> g(gamma.begin(), gamma.end());
    for (std::size_t i = 0; i < n; ++i) g[i * n + i] = 0.0;
    double kk = 0.0;
    for (double x : k) kk += x * x;
    double q = kk + detail::quad(s, g);
    if (q < 0.0) {
        // Not std::clamp: lower > upper is undefined there, and numpy's clip
        // accepts it.
        std::vector<double> clipped(n);
        for (std::size_t i = 0; i < n; ++i) clipped[i] = std::min(std::max(s[i], -k[i]), k[i]);
        q = kk + detail::quad(clipped, g);
    }
    return std::sqrt(std::max(q, 0.0));
}

// Weighted sensitivities of one bucket and their medium correlations (row-major
// n x n; not read for an other bucket). other aggregates as sum |WS|
// (MAR21.56, 21.79).
struct BucketInput {
    std::vector<double> ws;
    std::vector<double> rho;
    bool other = false;
};

// Charge per MAR21.6 scenario plus their maximum. gamma(b, c) is the medium
// cross-bucket correlation, asked for b != c in both orders. No buckets gives
// zero.
template <class G>
[[nodiscard]] Charges aggregate(const std::map<int, BucketInput>& buckets, const G& gamma) {
    const std::size_t n = buckets.size();
    std::array<double, 3> res{};
    for (std::size_t s = 0; s < kScenarios.size(); ++s) {
        const Scenario sc = kScenarios[s];
        std::vector<double> ks, ss;
        for (const auto& [id, bi] : buckets) {
            if (bi.other) {
                detail::require_finite(bi.ws, "ws");
                double k = 0.0;
                for (double w : bi.ws) k += std::fabs(w);
                ks.push_back(k);
            } else {
                ks.push_back(
                    within_bucket(bi.ws, detail::scenario_matrix(bi.rho, bi.ws.size(), sc)));
            }
            ss.push_back(detail::sum(bi.ws));
        }
        std::vector<double> g(n * n, 0.0);
        std::size_t i = 0;
        for (const auto& [b, bi] : buckets) {
            std::size_t j = 0;
            for (const auto& [c, ci] : buckets) {
                if (i != j) g[i * n + j] = scenario_corr(gamma(b, c), sc);
                ++j;
            }
            ++i;
        }
        res[s] = across_buckets(ks, ss, g);
    }
    return detail::from_scenarios(res);
}

// (CVR+, CVR-) of one instrument (MAR21.5(2)). rw * s is the delta P&L of the
// shock, so for a linear instrument both are 0.
[[nodiscard]] inline std::pair<double, double> net_cvr(double v0, double v_up, double v_down,
                                                       double rw, double s) noexcept {
    return {v0 - v_up + rw * s, v0 - v_down - rw * s};
}

// One curvature row: a risk factor key (equal keys are summed) with its CVR+
// and CVR-.
template <class K>
struct CvrRow {
    K key;
    double up = 0.0;
    double down = 0.0;
};

// Curvature charge per scenario and their maximum (MAR21.96-101).
// delta_rho(b, k, l) and delta_gamma(b, c) are the medium delta correlations,
// squared here. Buckets in other take the sum of the positive CVRs on each side.
template <class K, class R, class G>
[[nodiscard]] Charges curvature_charge(const std::map<int, std::vector<CvrRow<K>>>& buckets,
                                       const R& delta_rho, const G& delta_gamma,
                                       const std::set<int>& other = {}) {
    struct Net {
        int id;
        std::vector<const K*> keys;
        std::vector<double> ups, dns;
    };
    std::vector<Net> nets;
    nets.reserve(buckets.size());
    for (const auto& [b, rows] : buckets) {
        std::map<K, std::pair<double, double>> acc;
        std::map<K, const K*> origin;
        for (const auto& r : rows) {
            const std::array<double, 2> cvr{r.up, r.down};
            detail::require_finite(cvr, "CVR");
            auto& e = acc[r.key];
            e.first += r.up;
            e.second += r.down;
            origin.emplace(r.key, &r.key);
        }
        Net net{b, {}, {}, {}};
        for (const auto& [key, v] : acc) {
            net.keys.push_back(origin.at(key));
            net.ups.push_back(v.first);
            net.dns.push_back(v.second);
        }
        nets.push_back(std::move(net));
    }
    std::array<double, 3> res{};
    for (std::size_t s = 0; s < kScenarios.size(); ++s) {
        const Scenario sc = kScenarios[s];
        std::vector<double> ks, ws;
        for (const auto& net : nets) {
            double kp = 0.0, km = 0.0;
            if (other.contains(net.id)) {
                for (double x : net.ups) kp += std::max(x, 0.0);
                for (double x : net.dns) km += std::max(x, 0.0);
            } else {
                const auto rho = [&](std::size_t i, std::size_t j) {
                    const double d = delta_rho(net.id, *net.keys[i], *net.keys[j]);
                    if (!std::isfinite(d)) throw std::invalid_argument("delta_rho must be finite");
                    return scenario_corr(d * d, sc);
                };
                kp = detail::k_side(net.ups, rho);
                km = detail::k_side(net.dns, rho);
            }
            const double su = detail::sum(net.ups), sd = detail::sum(net.dns);
            const bool up_sel = kp > km || (kp == km && su > sd);
            ks.push_back(up_sel ? kp : km);
            ws.push_back(up_sel ? su : sd);
        }
        double q = 0.0;
        for (double k : ks) q += k * k;
        for (std::size_t i = 0; i < nets.size(); ++i)
            for (std::size_t j = 0; j < nets.size(); ++j)
                if (i != j) {
                    const double d = delta_gamma(nets[i].id, nets[j].id);
                    if (!std::isfinite(d))
                        throw std::invalid_argument("delta_gamma must be finite");
                    q += scenario_corr(d * d, sc) * ws[i] * ws[j] * detail::psi(ws[i], ws[j]);
                }
        res[s] = std::sqrt(std::max(q, 0.0));
    }
    return detail::from_scenarios(res);
}

}  // namespace qp::frtb
