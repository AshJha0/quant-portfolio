// W4(b) for the FX engine: scenario s on Philox stream (seed, s) with
// AS 241 normals. The pinned words below are this engine's output (GCC
// 15.2, glibc, -ffp-contract=off); rust/fx-var-engine/tests/test_w4b.rs
// pins the same words, so the two tests together check C++/Rust bit parity.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fxvar/kernels.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/monte_carlo.hpp"

using namespace fxvar;

namespace {

struct Pin {
  int dist;
  std::size_t s, i;
  std::uint64_t bits;
};

constexpr Pin kPinned[] = {
    {0, 0, 0, 0x3f5169b979aa098cULL}, {0, 0, 1, 0xbf86b7306eb673e3ULL},
    {0, 0, 2, 0xbf94594746d60c6fULL}, {0, 1, 0, 0xbf7e2f36a9e33b5aULL},
    {0, 1, 1, 0x3f9194a0f89ac69dULL}, {0, 1, 2, 0x3f6496dfb0aca743ULL},
    {0, 2, 0, 0x3f8694c20ebe07d8ULL}, {0, 2, 1, 0x3f9fe2c5b6884ca4ULL},
    {0, 2, 2, 0x3f74a1138da23b8cULL}, {0, 3, 0, 0x3f9b762aff9934edULL},
    {0, 3, 1, 0x3fa21d3c98b692d6ULL}, {0, 3, 2, 0x3f92644d20a3da6fULL},
    {1, 0, 0, 0x3f41418fa1bcf832ULL}, {1, 0, 1, 0xbf7682cb90d27836ULL},
    {1, 0, 2, 0xbf842a57f4fbf190ULL}, {1, 1, 0, 0xbf76fa017cf7b336ULL},
    {1, 1, 1, 0x3f8ac3d108973e7fULL}, {1, 1, 2, 0x3f5f587299f61997ULL},
    {1, 2, 0, 0x3f82d80f52840f28ULL}, {1, 2, 1, 0x3f9a9bdb9e5d2cdcULL},
    {1, 2, 2, 0x3f713712f2c36ebcULL}, {1, 3, 0, 0x3f8d08ca0ab874b9ULL},
    {1, 3, 1, 0x3f9326d0476157f7ULL}, {1, 3, 2, 0x3f8371f2b488df08ULL},
    {2, 0, 0, 0x3f9dc2a3c8c18a12ULL}, {2, 0, 1, 0xbf86b7306eb673e3ULL},
    {2, 0, 2, 0xbfb646e48b31f81eULL}, {2, 1, 0, 0x3f81ad0050f1b64bULL},
    {2, 1, 1, 0x3f9194a0f89ac69dULL}, {2, 1, 2, 0xbfa6453768d26736ULL},
    {2, 2, 0, 0x3f97941b65944369ULL}, {2, 2, 1, 0x3f9fe2c5b6884ca4ULL},
    {2, 2, 2, 0xbf9f4fc65fae9621ULL}, {2, 3, 0, 0x3fac111998600f33ULL},
    {2, 3, 1, 0x3fa21d3c98b692d6ULL}, {2, 3, 2, 0xbfb08a4045a7296fULL},
};
constexpr std::uint64_t kBootSeBits = 0x402de47205bbe00bULL;

FactorCov cov3() {
  FactorCov cov;
  cov.factors = {"FX:EUR", "FX:GBP", "FX:JPY"};
  cov.cov = Matrix::from_rows(
      {{3.6e-5, 1.1e-5, 0.4e-5}, {1.1e-5, 4.9e-5, 0.9e-5}, {0.4e-5, 0.9e-5, 2.5e-5}});
  return cov;
}

JumpSpec jumps() {
  JumpSpec j;
  j.prob = 0.3;
  j.mean = {{"FX:JPY", -0.01}, {"FX:EUR", 0.004}};
  j.stdev = {{"FX:JPY", 0.005}, {"FX:EUR", 0.0}};
  return j;
}

}  // namespace

TEST(W4b, ScenariosArePinnedWords) {
  const McDist dists[] = {McDist::kNormal, McDist::kStudentT, McDist::kJump};
  for (int d = 0; d < 3; ++d) {
    const ReturnsMatrix r = simulate_factor_returns(cov3(), 4, dists[d], 5.0, jumps(), 2026, 10.0);
    for (const Pin& p : kPinned) {
      if (p.dist != d) continue;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(r.data(p.s, p.i)), p.bits)
          << "dist " << d << " scenario " << p.s << " factor " << p.i;
    }
  }
}

TEST(W4b, NoexceptKernelGivesTheSameScenarios) {
  const FactorCov cov = cov3();
  Matrix a = cov.cov;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) a(i, j) *= 10.0;
  std::vector<double> lower(9), out(4 * 3);
  ASSERT_TRUE(try_cholesky_lower(a, lower).has_value());
  for (McDist d : {McDist::kNormal, McDist::kStudentT}) {
    ASSERT_TRUE(try_simulate_factor_returns(lower, 3, 4, d, 5.0, 2026, out).has_value());
    const ReturnsMatrix r = simulate_factor_returns(cov, 4, d, 5.0, {}, 2026, 10.0);
    for (std::size_t s = 0; s < 4; ++s)
      for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(out[s * 3 + i]),
                  std::bit_cast<std::uint64_t>(r.data(s, i)));
  }
}

TEST(W4b, ScenarioDependsOnlyOnItsIndex) {
  // The first 4 rows of a 50-row run equal a 4-row run: no shared stream.
  const ReturnsMatrix a = simulate_factor_returns(cov3(), 50, McDist::kJump, 5.0, jumps(), 7, 3.0);
  const ReturnsMatrix b = simulate_factor_returns(cov3(), 4, McDist::kJump, 5.0, jumps(), 7, 3.0);
  for (std::size_t s = 0; s < 4; ++s)
    for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(a.data(s, i), b.data(s, i));
}

TEST(W4b, BootstrapSeIsPinnedWord) {
  std::vector<double> pnl(300);
  for (int i = 0; i < 300; ++i) pnl[i] = std::sin(1.7 * i) * 1000.0 + i;
  const double se = var_standard_error_bootstrap(pnl, 0.01, 200, 99);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(se), kBootSeBits) << se;
}
