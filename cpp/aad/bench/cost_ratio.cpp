// AAD cost ratio: time for price plus all Greeks over time for the price.
// Best of kReps timings each; the plan's bar (W19) is "all Greeks under 5x
// one PV". Build with -O2 or -O3; numbers depend on the machine and load.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "qp/aad.hpp"

namespace {
constexpr int kReps = 21;
volatile double g_sink = 0.0;

template <class F>
double best(F&& f, int inner) {
  double b = std::numeric_limits<double>::infinity();
  for (int r = 0; r < kReps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < inner; ++i) f();
    const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
    b = std::min(b, dt.count() / inner);
  }
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  using namespace qp::aad;
  volatile double s0 = 100.0;
  const double b[6] = {100.0, 100.0, 1.0, 0.05, 0.02, 0.20};
  double p = best([&] { g_sink = bs_price<double>(s0, b[1], b[2], b[3], b[4], b[5], 1.0); }, 100000);
  double g = best([&] { g_sink = bs_greeks(s0, b[1], b[2], b[3], b[4], b[5], true).vega; }, 100000);
  std::printf("bs     price %10.3f us  aad %10.3f us  ratio %.2f\n", p * 1e6, g * 1e6, g / p);

  const std::size_t steps = argc > 1 ? std::stoul(argv[1]) : 52;
  std::mt19937_64 rng(1);
  std::normal_distribution<double> nd;
  std::vector<double> z(20000 * steps);
  for (auto& v : z) v = nd(rng);
  for (Payoff pay : {Payoff::European, Payoff::Asian}) {
    p = best([&] { g_sink = mc_price(s0, b[1], b[2], b[3], b[4], b[5], z, steps, pay); }, 3);
    g = best([&] { g_sink = mc_greeks(s0, b[1], b[2], b[3], b[4], b[5], z, steps, pay).vega; }, 3);
    std::printf("mc %-8s price %10.3f ms  aad %10.3f ms  ratio %.2f\n", pay == Payoff::European ? "European" : "Asian",
                p * 1e3, g * 1e3, g / p);
  }

  // 10Y annual swap on an 18-pillar curve shaped like the USD OIS fixture.
  const std::vector<std::int64_t> days{3, 13, 39, 98, 173, 264, 355, 446, 557, 739, 1104, 1469, 1833, 2565, 3660, 5487, 7314, 10966};
  std::vector<double> dfs;
  for (auto d : days) dfs.push_back(std::exp(-0.04 * static_cast<double>(d) / 365.0));
  Swap sw{{}, {}, 0.04, 1e6, true};
  for (std::int64_t y = 1; y <= 10; ++y) {
    sw.fixed.push_back({365 * y + 5, 1.0139});
    sw.floating.push_back({365 * (y - 1), 365 * y, 365 * y + 5, 1.0139});
  }
  p = best([&] { g_sink = swap_pv_from_dfs(days, dfs, sw) + s0; }, 100000);
  g = best([&] { g_sink = swap_pillar_greeks(days, dfs, sw).pv + s0; }, 100000);
  std::printf("swap   pv    %10.3f us  aad %10.3f us  ratio %.2f\n", p * 1e6, g * 1e6, g / p);
  return 0;
}
