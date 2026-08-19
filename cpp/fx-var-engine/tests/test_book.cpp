// Book tests: triangulation identity, forward CIP consistency, base-ccy
// P&L convention, edge cases.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxvar/book.hpp"

using namespace fxvar;

namespace {

Market test_market() {
  return Market({{"EUR", 1.10}, {"JPY", 0.0090}, {"GBP", 1.27}, {"CHF", 1.12}},
                {{"USD", 0.050}, {"EUR", 0.030}, {"JPY", 0.001}, {"GBP", 0.045}});
}

}  // namespace

TEST(PairUtils, SplitAndFactors) {
  const PairLegs legs = split_pair("eurusd");
  EXPECT_EQ(legs.base, "EUR");
  EXPECT_EQ(legs.quote, "USD");
  EXPECT_EQ(fx_factor("jpy"), "FX:JPY");
  EXPECT_EQ(ir_factor("usd"), "IR:USD");
  EXPECT_THROW(split_pair("EUR"), std::invalid_argument);
  EXPECT_THROW(split_pair("EUREUR"), std::invalid_argument);
  EXPECT_THROW(split_pair("EUR US"), std::invalid_argument);
  EXPECT_THROW(fx_factor("USD"), std::invalid_argument);
}

TEST(Market, SpotCrossForwardAndValidation) {
  const Market m = test_market();
  EXPECT_DOUBLE_EQ(m.spot("USD"), 1.0);
  EXPECT_DOUBLE_EQ(m.spot("EUR"), 1.10);
  // EURJPY by triangulation: 1.10 / 0.0090.
  EXPECT_NEAR(m.cross("EURJPY"), 1.10 / 0.0090, 1e-12);
  // CIP: F(EURUSD, 1y) = 1.10 * exp((r_usd - r_eur) * 1).
  EXPECT_NEAR(m.forward("EURUSD", 1.0), 1.10 * std::exp(0.020), 1e-12);
  EXPECT_THROW(m.spot("XXX"), std::invalid_argument);
  EXPECT_THROW(m.rate("CHF"), std::invalid_argument);
  EXPECT_THROW(Market({{"EUR", -1.0}}), std::invalid_argument);
  EXPECT_THROW(Market({{"USD", 1.05}}), std::invalid_argument);
}

TEST(Book, FactorEnumeration) {
  Book book({SpotPosition{"EURUSD", 1e6, {}},
             ForwardPosition{"USDJPY", 2e6, 0.5, {}}});
  const auto f = book.factors();
  const std::vector<std::string> expected{"FX:EUR", "FX:JPY", "IR:JPY",
                                          "IR:USD"};
  EXPECT_EQ(f, expected);
}

TEST(Book, TriangulationIdentityEURJPY) {
  // A cross position decomposes into its two USD legs: long N EURJPY
  // == long N EURUSD + long N*X(EURUSD) USDJPY, scenario by scenario.
  const Market m = test_market();
  const double n = 7'000'000.0;
  Book cross({SpotPosition{"EURJPY", n, {}}});
  Book legs({SpotPosition{"EURUSD", n, {}},
             SpotPosition{"USDJPY", n * m.cross("EURUSD"), {}}});
  const CompiledBook cb_cross(cross, m);
  const CompiledBook cb_legs(legs, m);
  const std::map<std::string, double> shocks{{"FX:EUR", 0.013},
                                             {"FX:JPY", -0.021}};
  const double p_cross = cb_cross.pnl(shocks);
  const double p_legs = cb_legs.pnl(shocks);
  EXPECT_NEAR(p_cross, p_legs, 1e-12 * std::abs(p_cross) + 1e-12);
  EXPECT_GT(std::abs(p_cross), 1.0);  // the scenario is not trivial
}

TEST(Book, ForwardZeroValueAtInceptionAndCip) {
  const Market m = test_market();
  // ATM CIP strike => zero initial value.
  Book atm({ForwardPosition{"EURUSD", 5e6, 0.75, {}}});
  const CompiledBook cb(atm, m);
  EXPECT_NEAR(cb.value0_usd(), 0.0, 1e-10 * 5e6);
  // Explicit strike: value = N e^{-r_f T} S_f - N K e^{-r_d T} S_d.
  const double k = 1.08;
  Book struck({ForwardPosition{"EURUSD", 5e6, 0.75, k}});
  const CompiledBook cb2(struck, m);
  const double expect = 5e6 * std::exp(-0.030 * 0.75) * 1.10 -
                        5e6 * k * std::exp(-0.050 * 0.75) * 1.0;
  EXPECT_NEAR(cb2.value0_usd(), expect, 1e-10 * std::abs(expect));
}

TEST(Book, ForwardMatchesTwoLegDepositDecomposition) {
  // For pure FX shocks the forward P&L must equal the P&L of its two
  // discounted cash legs: +N e^{-r_f T} EUR and -N K e^{-r_d T} USD.
  const Market m = test_market();
  const double n = 5e6, t = 0.5;
  const double k = m.forward("EURUSD", t);
  Book fwd({ForwardPosition{"EURUSD", n, t, k}});
  Book cash({CashPosition{"EUR", n * std::exp(-0.030 * t)},
             CashPosition{"USD", -n * k * std::exp(-0.050 * t)}});
  const CompiledBook cb_f(fwd, m);
  const CompiledBook cb_c(cash, m);
  const std::map<std::string, double> shocks{{"FX:EUR", -0.045}};
  const double pf = cb_f.pnl(shocks);
  const double pc = cb_c.pnl(shocks);
  EXPECT_NEAR(pf, pc, 1e-10 * std::abs(pf) + 1e-10);
}

TEST(Book, ForwardRateLegSensitivities) {
  // dV/dr_d = +T N K e^{-r_d T} S_d and dV/dr_f = -T N e^{-r_f T} S_f
  // (in USD): check the finite-difference exposures against closed form.
  const Market m = test_market();
  const double n = 5e6, t = 0.5;
  const double k = m.forward("EURUSD", t);
  Book fwd({ForwardPosition{"EURUSD", n, t, k}});
  const CompiledBook cb(fwd, m);
  const auto factors = cb.factors();  // FX:EUR, IR:EUR, IR:USD
  const auto w = cb.linear_exposures();
  ASSERT_EQ(factors.size(), 3u);
  EXPECT_EQ(factors[1], "IR:EUR");
  EXPECT_EQ(factors[2], "IR:USD");
  const double dv_dr_eur = -t * n * std::exp(-0.030 * t) * 1.10;
  const double dv_dr_usd = +t * n * k * std::exp(-0.050 * t);
  EXPECT_NEAR(w[1], dv_dr_eur, 1e-4 * std::abs(dv_dr_eur));
  EXPECT_NEAR(w[2], dv_dr_usd, 1e-4 * std::abs(dv_dr_usd));
}

TEST(Book, BaseCurrencyPositionHasZeroRisk) {
  // A GBP cash balance in a GBP-based book is riskless even when FX:GBP
  // moves violently.
  const Market m = test_market();
  Book book({CashPosition{"GBP", 25e6}}, "GBP");
  const CompiledBook cb(book, m);
  for (double shock : {-0.20, -0.05, 0.0, 0.07, 0.30}) {
    const double p = cb.pnl(std::map<std::string, double>{{"FX:GBP", shock}});
    EXPECT_NEAR(p, 0.0, 1e-9 * 25e6);
  }
}

TEST(Book, NonUsdBasePnlConvention) {
  // USD cash in a EUR book: PnL_eur = A/S1 - A/S0 exactly.
  const Market m = test_market();
  const double a = 10e6;
  Book book({CashPosition{"USD", a}}, "EUR");
  const CompiledBook cb(book, m);
  const double shock = 0.02;  // EURUSD up 2% (log)
  const double s0 = 1.10, s1 = 1.10 * std::exp(shock);
  const double expect = a / s1 - a / s0;
  const double got = cb.pnl(std::map<std::string, double>{{"FX:EUR", shock}});
  EXPECT_NEAR(got, expect, 1e-9 * std::abs(expect));
}

TEST(Book, EmptyBookThrows) {
  const Market m = test_market();
  EXPECT_THROW(CompiledBook(Book{}, m), std::invalid_argument);
}

TEST(Book, UsdShockRejectedAndUnknownFactorsIgnored) {
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 1e6, {}}});
  const CompiledBook cb(book, m);
  EXPECT_THROW(cb.pnl(std::map<std::string, double>{{"FX:USD", 0.01}}),
               std::invalid_argument);
  // Shocks for factors the book does not carry are ignored (scenario
  // library convention).
  EXPECT_DOUBLE_EQ(cb.pnl(std::map<std::string, double>{{"FX:TRY", -0.3}}),
                   0.0);
}

TEST(Book, PositionValidation) {
  EXPECT_THROW(Book({SpotPosition{"EURUSD", 1e6, -1.0}}),
               std::invalid_argument);
  EXPECT_THROW(Book({ForwardPosition{"EURUSD", 1e6, -0.5, {}}}),
               std::invalid_argument);
  EXPECT_THROW(Book({ForwardPosition{"EURUSD", 1e6, 0.5, 0.0}}),
               std::invalid_argument);
  EXPECT_THROW(Book({SpotPosition{"E2RUSD", 1e6, {}}}), std::invalid_argument);
}

TEST(Book, SingleCurrencyBookSpotExposureIsNotional) {
  // Long 10m EURUSD: FX:EUR delta = N * S_eur (USD P&L per unit log ret).
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}}});
  const CompiledBook cb(book, m);
  const auto w = cb.linear_exposures();
  ASSERT_EQ(w.size(), 1u);
  EXPECT_NEAR(w[0], 10e6 * 1.10, 1e-3);
}


TEST(Market, TriangulationIdentitiesAreExact) {
  // The USD-pivot factor set makes cross rates consistent by construction:
  // EURJPY = EURUSD * USDJPY, and a pair times its inverse is 1.
  const Market m = test_market();
  EXPECT_NEAR(m.cross("EURJPY"), m.cross("EURUSD") * m.cross("USDJPY"),
              1e-12 * m.cross("EURJPY"));
  EXPECT_NEAR(m.cross("EURUSD") * m.cross("USDEUR"), 1.0, 1e-15);
  EXPECT_NEAR(m.cross("JPYUSD") * m.cross("USDJPY"), 1.0, 1e-15);
  // Triangulating through a third currency gives the same cross.
  EXPECT_NEAR(m.cross("EURJPY"),
              m.cross("EURUSD") / m.cross("JPYUSD"),
              1e-12 * m.cross("EURJPY"));
  // Forward triangulation is consistent with CIP through the same pivot.
  const double t = 1.5;
  EXPECT_NEAR(m.forward("EURJPY", t),
              m.cross("EURJPY") *
                  std::exp((m.rate("JPY") - m.rate("EUR")) * t),
              1e-10 * m.forward("EURJPY", t));
}

TEST(Book, SingleCurrencyBookInNonUsdBaseHasNoFxRisk) {
  // A EUR-base book holding only EUR cash carries no risk at all: its one
  // factor (FX:EUR) cancels between the position and the reporting ccy.
  const Market m = test_market();
  Book book({CashPosition{"EUR", 4.2e6}}, "EUR");
  const CompiledBook cb(book, m);
  const auto w = cb.linear_exposures();
  ASSERT_EQ(w.size(), 1u);
  EXPECT_NEAR(w[0], 0.0, 1e-3);
  for (const double shock : {-0.25, -0.01, 0.0, 0.02, 0.30}) {
    EXPECT_NEAR(cb.pnl(std::map<std::string, double>{{"FX:EUR", shock}}), 0.0,
                1e-6);
  }
  // The same balance held in a USD-base book is fully exposed.
  Book usd_book({CashPosition{"EUR", 4.2e6}}, "USD");
  const CompiledBook cb_usd(usd_book, m);
  EXPECT_NEAR(cb_usd.linear_exposures()[0], 4.2e6 * 1.10, 1e-3);
}
