// qp::aad: tape-based reverse-mode algorithmic differentiation (plan W19),
// the C++ port of python/core/src/qp_core/aad and rust/aad.
//
// Plain idea: every arithmetic step of a pricer is recorded on a tape with
// its local partial derivatives; one backward walk over the tape gives the
// derivative of the price with respect to every input at once.
// Formal statement: for nodes v_i = phi_i(v_j) the adjoints
// vbar_j = sum_i vbar_i d phi_i / d v_j, seeded with ybar = 1 and evaluated
// in reverse order (Griewank and Walther 2008, Evaluating Derivatives, ch. 3;
// Savine 2018, Modern Computational Finance, ch. 9).
//
// Layout: at most two parents per node in one flat vector, mark()/rewind()
// for per-path Monte Carlo sweeps. Pricers are templates over the number
// type: double gives the plain price (the denominator of the cost ratio),
// Var gives the recorded price.
//
// Worked example:
//   qp::aad::Tape tape;
//   auto x0 = tape.var(1.0), x1 = tape.var(2.0);
//   auto y = x0 * x1 + exp(x0);
//   auto g = tape.gradient(y, {x0, x1});   // {2 + e, 1}
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

#include "qp/normal.hpp"
#include "qp/rates_curves.hpp"

namespace qp::aad {

class Tape;

// A value recorded on a Tape.
struct Var {
  Tape* tape;
  std::uint32_t idx;
  double val;
};

class Tape {
 public:
  static constexpr std::uint32_t kNone = 0xFFFFFFFFu;

  Tape() = default;
  explicit Tape(std::size_t capacity) {
    nodes_.reserve(capacity);
    adj_.reserve(capacity);
  }
  Tape(const Tape&) = delete;
  Tape& operator=(const Tape&) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }

  Var push(double val, std::uint32_t a, double da, std::uint32_t b = kNone, double db = 0.0) {
    nodes_.push_back({a, b, da, db});
    return {this, static_cast<std::uint32_t>(nodes_.size() - 1), val};
  }

  // A new independent input.
  Var var(double val) { return push(val, kNone, 0.0); }

  [[nodiscard]] std::size_t mark() const noexcept { return nodes_.size(); }

  // Forget every node recorded after mark; adjoints before it stay.
  void rewind(std::size_t mark) {
    nodes_.resize(std::min(mark, nodes_.size()));
    adj_.resize(std::min(mark, adj_.size()));
  }

  void zero_adjoints() { adj_.assign(nodes_.size(), 0.0); }

  // Add seed * d output / d node into every adjoint, sweeping from output
  // down to node stop (nodes below stop only receive).
  void backward(const Var& output, double seed = 1.0, std::size_t stop = 0) {
    if (output.tape != this) throw std::invalid_argument("output was recorded on a different tape");
    if (adj_.size() < nodes_.size()) adj_.resize(nodes_.size(), 0.0);
    adj_[output.idx] += seed;
    propagate(static_cast<std::size_t>(output.idx) + 1, stop);
  }

  // Push the adjoints of nodes start-1 down to stop into their parents.
  void propagate(std::size_t start, std::size_t stop = 0) {
    if (adj_.size() < nodes_.size()) adj_.resize(nodes_.size(), 0.0);
    for (std::size_t i = start; i-- > stop;) {
      const double w = adj_[i];
      if (w == 0.0) continue;
      const Node& n = nodes_[i];
      if (n.a != kNone) {
        adj_[n.a] += w * n.da;
        if (n.b != kNone) adj_[n.b] += w * n.db;
      }
    }
  }

  [[nodiscard]] double adjoint(const Var& v) const noexcept { return v.idx < adj_.size() ? adj_[v.idx] : 0.0; }

  std::vector<double> gradient(const Var& output, const std::vector<Var>& inputs) {
    zero_adjoints();
    backward(output);
    std::vector<double> g;
    g.reserve(inputs.size());
    for (const auto& x : inputs) g.push_back(adjoint(x));
    return g;
  }

 private:
  struct Node {
    std::uint32_t a;
    std::uint32_t b;
    double da;
    double db;
  };
  std::vector<Node> nodes_;
  std::vector<double> adj_;
};

// ---- arithmetic -------------------------------------------------------
inline Var operator+(const Var& x, const Var& y) { return x.tape->push(x.val + y.val, x.idx, 1.0, y.idx, 1.0); }
inline Var operator-(const Var& x, const Var& y) { return x.tape->push(x.val - y.val, x.idx, 1.0, y.idx, -1.0); }
inline Var operator*(const Var& x, const Var& y) { return x.tape->push(x.val * y.val, x.idx, y.val, y.idx, x.val); }
inline Var operator/(const Var& x, const Var& y) {
  const double q = x.val / y.val;
  return x.tape->push(q, x.idx, 1.0 / y.val, y.idx, -q / y.val);
}
inline Var operator-(const Var& x) { return x.tape->push(-x.val, x.idx, -1.0); }
inline Var operator+(const Var& x, double c) { return x.tape->push(x.val + c, x.idx, 1.0); }
inline Var operator+(double c, const Var& x) { return x + c; }
inline Var operator-(const Var& x, double c) { return x.tape->push(x.val - c, x.idx, 1.0); }
inline Var operator-(double c, const Var& x) { return x.tape->push(c - x.val, x.idx, -1.0); }
inline Var operator*(const Var& x, double c) { return x.tape->push(x.val * c, x.idx, c); }
inline Var operator*(double c, const Var& x) { return x * c; }
inline Var operator/(const Var& x, double c) { return x.tape->push(x.val / c, x.idx, 1.0 / c); }
inline Var operator/(double c, const Var& x) {
  const double q = c / x.val;
  return x.tape->push(q, x.idx, -q / x.val);
}
inline Var exp(const Var& x) {
  const double e = std::exp(x.val);
  return x.tape->push(e, x.idx, e);
}
inline Var log(const Var& x) { return x.tape->push(std::log(x.val), x.idx, 1.0 / x.val); }
inline Var sqrt(const Var& x) {
  const double s = std::sqrt(x.val);
  return x.tape->push(s, x.idx, 0.5 / s);
}
inline Var ncdf(const Var& x) { return x.tape->push(qp::norm_cdf(x.val), x.idx, qp::norm_pdf(x.val)); }
inline double exp(double x) { return std::exp(x); }
inline double log(double x) { return std::log(x); }
inline double sqrt(double x) { return std::sqrt(x); }
inline double ncdf(double x) { return qp::norm_cdf(x); }
inline double value(double x) { return x; }
inline double value(const Var& x) { return x.val; }

// ---- Black-Scholes (Hull 2022, ch. 19) ---------------------------------
template <class T>
T bs_price(T s, T k, T t, T r, T q, T sigma, double phi) {
  const T sd = sigma * sqrt(t);
  const T d1 = (log(s / k) + (r - q + sigma * sigma * 0.5) * t) / sd;
  const T d2 = d1 - sd;
  return (s * exp(-(q * t)) * ncdf(d1 * phi) - k * exp(-(r * t)) * ncdf(d2 * phi)) * phi;
}

struct Greeks {
  double price, delta, vega, rho, rho_q, d_k, d_t, std_error;
};

inline Greeks bs_greeks(double s, double k, double t, double r, double q, double sigma, bool is_call) {
  Tape tape(64);
  std::vector<Var> x{tape.var(s), tape.var(k), tape.var(t), tape.var(r), tape.var(q), tape.var(sigma)};
  const Var v = bs_price(x[0], x[1], x[2], x[3], x[4], x[5], is_call ? 1.0 : -1.0);
  const auto g = tape.gradient(v, x);
  return {v.val, g[0], g[5], g[3], g[4], g[1], g[2], 0.0};
}

// ---- Monte Carlo, pathwise adjoint (Giles and Glasserman 2006) ----------
enum class Payoff { European, Asian };

template <class T>
std::optional<T> path_payoff(T log_s0, T drift, T vol, T k, const double* z, std::size_t steps, Payoff payoff) {
  std::optional<T> under;
  if (payoff == Payoff::European) {
    // ln S_T depends on the inputs only through sum(z): one exp per path.
    double zsum = 0.0;
    for (std::size_t i = 0; i < steps; ++i) zsum += z[i];
    under = exp(log_s0 + drift * static_cast<double>(steps) + vol * zsum);
  } else {
    T x = log_s0;
    std::optional<T> total;
    for (std::size_t i = 0; i < steps; ++i) {
      x = x + drift + vol * z[i];
      const T s = exp(x);
      total = total ? *total + s : s;
    }
    under = *total / static_cast<double>(steps);
  }
  if (value(*under) > value(k)) return *under - k;
  return std::nullopt;
}

inline double mc_price(double s, double k, double t, double r, double q, double sigma, const std::vector<double>& z,
                       std::size_t steps, Payoff payoff) {
  const std::size_t paths = z.size() / steps;
  const double dt = t / static_cast<double>(steps);
  const double drift = (r - q - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt), ls = std::log(s);
  double acc = 0.0;
  for (std::size_t p = 0; p < paths; ++p) {
    if (auto v = path_payoff(ls, drift, vol, k, z.data() + p * steps, steps, payoff)) acc += *v;
  }
  return std::exp(-r * t) * acc / static_cast<double>(paths);
}

inline Greeks mc_greeks(double s, double k, double t, double r, double q, double sigma, const std::vector<double>& z,
                        std::size_t steps, Payoff payoff) {
  const std::size_t paths = z.size() / steps;
  const double n = static_cast<double>(paths);
  Tape tape(64 + 8 * steps);
  std::vector<Var> x{tape.var(s), tape.var(k), tape.var(t), tape.var(r), tape.var(q), tape.var(sigma)};
  const Var dt = x[2] / static_cast<double>(steps);
  const Var drift = (x[3] - x[4] - x[5] * x[5] * 0.5) * dt;
  const Var vol = x[5] * sqrt(dt);
  const Var ls = log(x[0]);
  const Var disc = exp(-(x[3] * x[2]));
  const std::size_t mark = tape.mark();
  tape.zero_adjoints();
  double total = 0.0, sq = 0.0;
  for (std::size_t p = 0; p < paths; ++p) {
    if (auto pay = path_payoff(ls, drift, vol, x[1], z.data() + p * steps, steps, payoff)) {
      const double pv = disc.val * pay->val;
      total += pv;
      sq += pv * pv;
      tape.backward(*pay, disc.val / n, mark);
    }
    tape.rewind(mark);
  }
  tape.backward(disc, total / disc.val / n, mark);  // d price / d disc = mean payoff
  tape.propagate(mark, 0);
  const double price = total / n;
  const double var = std::max(sq / n - price * price, 0.0);
  return {price,
          tape.adjoint(x[0]),
          tape.adjoint(x[5]),
          tape.adjoint(x[3]),
          tape.adjoint(x[4]),
          tape.adjoint(x[1]),
          tape.adjoint(x[2]),
          std::sqrt(var / n)};
}

// ---- swap PV off the dated bootstrap (Henrard 2014, ch. 4) --------------
struct Swap {
  std::vector<qp::rates::FixedPeriod> fixed;
  std::vector<qp::rates::FloatPeriod> floating;
  double rate;
  double notional;
  bool payer;
};

// ln P(day) from pillar log discount factors xs on days (implicit ln P(0) = 0),
// log-linear in ACT/365F time, flat forward beyond the last pillar.
template <class T>
T log_df(const std::vector<std::int64_t>& days, const std::vector<T>& xs, double day) {
  const double t = day / qp::rates::kDaysPerYear;
  auto tt = [&](std::size_t j) { return j == 0 ? 0.0 : static_cast<double>(days[j - 1]) / qp::rates::kDaysPerYear; };
  const std::size_t n = days.size();
  if (t >= tt(n)) {
    const double w = (t - tt(n)) / (tt(n) - tt(n - 1));
    return n == 1 ? xs[0] + xs[0] * w : xs[n - 1] + (xs[n - 1] - xs[n - 2]) * w;
  }
  std::size_t lo = 0, hi = n;
  while (hi - lo > 1) {
    const std::size_t mid = (lo + hi) / 2;
    if (tt(mid) <= t) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const double w = (t - tt(lo)) / (tt(hi) - tt(lo));
  return lo == 0 ? xs[hi - 1] * w : xs[lo - 1] + (xs[hi - 1] - xs[lo - 1]) * w;
}

template <class T>
T df(const std::vector<std::int64_t>& days, const std::vector<T>& xs, std::int64_t day) {
  return exp(log_df(days, xs, static_cast<double>(day)));
}

template <class T>
std::pair<T, T> legs(const std::vector<std::int64_t>& days, const std::vector<T>& xs,
                     const std::vector<qp::rates::FixedPeriod>& fixed,
                     const std::vector<qp::rates::FloatPeriod>& floating) {
  if (fixed.empty() || floating.empty()) throw std::invalid_argument("both legs need periods");
  std::optional<T> flt, ann;
  for (const auto& p : floating) {
    const T c = df(days, xs, p.pay) * (df(days, xs, p.start) / df(days, xs, p.end) - 1.0);
    flt = flt ? *flt + c : c;
  }
  for (const auto& f : fixed) {
    const T c = df(days, xs, f.pay) * f.accrual;
    ann = ann ? *ann + c : c;
  }
  return {*flt, *ann};
}

template <class T>
T swap_pv(const std::vector<std::int64_t>& days, const std::vector<T>& xs, const Swap& sw) {
  const auto [flt, ann] = legs(days, xs, sw.fixed, sw.floating);
  const T v = (flt - ann * sw.rate) * sw.notional;
  return sw.payer ? v : -v;
}

// Model minus quoted rate on a self-discounted, self-projected curve.
template <class T>
T quote_residual(const qp::rates::Quote& q, const std::vector<std::int64_t>& days, const std::vector<T>& xs) {
  if (const auto* f = std::get_if<qp::rates::ForwardQuote>(&q)) {
    return (df(days, xs, f->start) / df(days, xs, f->end) - 1.0) / f->accrual - f->rate;
  }
  const auto& s = std::get<qp::rates::SwapQuote>(q);
  const auto [flt, ann] = legs(days, xs, s.fixed, s.floating);
  return flt / ann - s.rate;
}

inline double swap_pv_from_dfs(const std::vector<std::int64_t>& days, const std::vector<double>& dfs, const Swap& sw) {
  std::vector<double> xs(dfs.size());
  std::transform(dfs.begin(), dfs.end(), xs.begin(), [](double d) { return std::log(d); });
  return swap_pv(days, xs, sw);
}

struct SwapGreeks {
  double pv;
  std::vector<double> d_log_df;  // dPV / d ln P(days[j])
  std::vector<double> d_quote;   // dPV / d quote_k (quote order); empty for pillar Greeks only
};

inline SwapGreeks swap_pillar_greeks(const std::vector<std::int64_t>& days, const std::vector<double>& dfs,
                                     const Swap& sw) {
  Tape tape(dfs.size() + 40 * (sw.fixed.size() + sw.floating.size()));
  std::vector<Var> xs;
  for (double d : dfs) xs.push_back(tape.var(std::log(d)));
  const Var v = swap_pv(days, xs, sw);
  return {v.val, tape.gradient(v, xs), {}};
}

// Solve a x = b (row-major n x n), Gaussian elimination with partial pivoting.
inline std::vector<double> solve(std::vector<double> a, std::vector<double> b) {
  const std::size_t n = b.size();
  for (std::size_t c = 0; c < n; ++c) {
    std::size_t p = c;
    for (std::size_t i = c + 1; i < n; ++i) {
      if (std::abs(a[i * n + c]) > std::abs(a[p * n + c])) p = i;
    }
    if (a[p * n + c] == 0.0) throw std::domain_error("singular residual Jacobian");
    if (p != c) {
      for (std::size_t j = 0; j < n; ++j) std::swap(a[p * n + j], a[c * n + j]);
      std::swap(b[p], b[c]);
    }
    for (std::size_t i = c + 1; i < n; ++i) {
      const double f = a[i * n + c] / a[c * n + c];
      if (f == 0.0) continue;
      for (std::size_t j = c; j < n; ++j) a[i * n + j] -= f * a[c * n + j];
      b[i] -= f * b[c];
    }
  }
  std::vector<double> x(n, 0.0);
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t j = i + 1; j < n; ++j) s -= a[i * n + j] * x[j];
    x[i] = s / a[i * n + i];
  }
  return x;
}

// Quote Greeks through the implicit function theorem: dPV/dq = g^T J^{-1},
// J = dR/dx (one adjoint sweep per quote), g = dPV/dx. days/dfs must be the
// curve bootstrapped from quotes, one pillar per quote.
inline SwapGreeks swap_quote_greeks(const std::vector<qp::rates::Quote>& quotes,
                                    const std::vector<std::int64_t>& days, const std::vector<double>& dfs,
                                    const Swap& sw) {
  if (quotes.size() != days.size()) throw std::invalid_argument("need one quote per pillar");
  SwapGreeks out = swap_pillar_greeks(days, dfs, sw);
  const std::size_t n = days.size();
  Tape tape(4096);
  std::vector<Var> xs;
  for (double d : dfs) xs.push_back(tape.var(std::log(d)));
  const std::size_t mark = tape.mark();
  std::vector<double> jt(n * n);
  for (std::size_t k = 0; k < n; ++k) {
    const auto row = tape.gradient(quote_residual(quotes[k], days, xs), xs);
    for (std::size_t j = 0; j < n; ++j) jt[j * n + k] = row[j];
    tape.rewind(mark);
  }
  out.d_quote = solve(std::move(jt), out.d_log_df);
  return out;
}

}  // namespace qp::aad
