// Gateway tests: kill-contract parity (golden vectors, byte for byte),
// pre-trade parity (vectors generated from the Python reference), the
// contract rules individually, and the gateway's gate order.
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "qp/ull/gateway.hpp"
#include "qp/ull/json.hpp"
#include "qp/ull/testing.hpp"

using namespace qp::ull;
using qp::ull::testing::slurp;

#ifndef QP_REPO_ROOT
#error "QP_REPO_ROOT must be defined by the build"
#endif

static std::string repo(const char* rel) { return std::string(QP_REPO_ROOT) + "/" + rel; }

// ---- kill contract -----------------------------------------------------------

TEST(kill_vectors_replay_byte_for_byte) {
  auto doc = json::parse(slurp(repo("python/oms/kill/vectors.json")));
  CHECK_EQ(doc.at("version").i64(), 1);
  int cases = 0;
  for (const auto& c : doc.at("cases").arr) {
    auto got = kill::replay(c.at("commands"));
    const auto& exp = c.at("expected").arr;
    CHECK_EQ(got.size(), exp.size());
    for (std::size_t i = 0; i < got.size() && i < exp.size(); ++i) {
      if (got[i] != exp[i].str())
        std::printf("  case %s line %zu\n    got %s\n    exp %s\n", c.at("name").str().c_str(), i, got[i].c_str(),
                    exp[i].str().c_str());
      CHECK(got[i] == exp[i].str());
    }
    ++cases;
  }
  CHECK(cases >= 3);
}

TEST(kill_scope_validation) {
  CHECK(kill::validate_scope({"QP", "*", "*", "*"}) == kill::Level::FIRM);
  CHECK(kill::validate_scope({"QP", "R", "M", "S"}) == kill::Level::SYMBOL);
  CHECK_THROWS(kill::validate_scope({"QP", "*", "MM1", "*"}));
  CHECK_THROWS(kill::validate_scope({"*", "*", "*", "*"}));
}

TEST(kill_k5_nesting_narrower_survives_wider_clear) {
  kill::KillSwitch ks;
  auto [firm, c1] = ks.engage({"QP", "*", "*", "*"}, "alice", "");
  auto [sym, c2] = ks.engage({"QP", "R", "M", "S"}, "carol", "");
  CHECK(ks.check({"QP", "R", "M", "S"}).value() == "KILLED:FIRM:QP/*/*/*");  // widest named
  CHECK(ks.request_reset(firm, "alice"));
  CHECK(ks.approve_reset(firm, "bob"));
  CHECK(ks.check({"QP", "R", "M", "S"}).value() == "KILLED:SYMBOL:QP/R/M/S");
  CHECK(!ks.check({"QP", "R", "M", "T"}).has_value());
  CHECK_THROWS(ks.request_reset(99, "x"));
}

TEST(kill_k7_idempotent_engage) {
  kill::KillSwitch ks;
  ks.on_order_working(3, {"QP", "R", "M", "S"});
  auto a = ks.engage({"QP", "R", "*", "*"}, "a", "");
  auto b = ks.engage({"QP", "R", "*", "*"}, "b", "");
  CHECK_EQ(a.first, b.first);
  CHECK_EQ(a.second.size(), 1u);
  CHECK(b.second.empty());
  CHECK_EQ(ks.events.back().event, std::string("engage_dup"));
}

TEST(json_escape_matches_cpython_ensure_ascii) {
  // Expected strings are json.dumps(s) from CPython 3.12.
  CHECK_EQ(json::escape("a\"b\\c"), std::string("\"a\\\"b\\\\c\""));
  CHECK_EQ(json::escape("\n\t\x01\x7f"), std::string("\"\\n\\t\\u0001\\u007f\""));
  CHECK_EQ(json::escape("\xc3\xa9"), std::string("\"\\u00e9\""));                // e acute
  CHECK_EQ(json::escape("\xf0\x9f\x98\x80"), std::string("\"\\ud83d\\ude00\""));  // U+1F600
  auto v = json::parse("[\"\\ud83d\\ude00\", 1.5e3, -0, true, null, {}]");
  CHECK_EQ(json::escape(v.at(0).str()), std::string("\"\\ud83d\\ude00\""));
  CHECK_THROWS(json::parse("[1,]"));
  CHECK_THROWS(json::parse("\"\\ud83d\""));
  CHECK_THROWS(json::parse("01"));
}

// ---- pre-trade parity ----------------------------------------------------------

static double num(const json::Value& v) { return std::strtod(v.str().c_str(), nullptr); }

TEST(pretrade_vectors_match_python_reference) {
  auto doc = json::parse(slurp(repo("cpp/ull/gateway/tests/pretrade_vectors.json")));
  const auto& L = doc.at("limits");
  risk::Limits lim{L.at("max_order_qty").f64(), L.at("max_order_notional").f64(), L.at("price_band").f64(),
                   L.at("position_limit").f64(), L.at("credit_limit").f64(),
                   int(L.at("throttle_msgs").i64()), L.at("throttle_window_ns").i64(), L.at("dup_window_ns").i64()};
  long ops = 0;
  for (const auto& c : doc.at("cases").arr) {
    risk::RiskBook b(lim);
    for (const auto& [k, v] : doc.at("ref_px").obj) b.ref_px[k] = v.f64();
    for (const auto& r : doc.at("restricted").arr) b.restricted.insert(r.str());
    for (const auto& op : c.at("ops").arr) {
      const std::string& kind = op.at("op").str();
      risk::Reason got;
      bool has_expect = false;
      if (kind == "new") {
        const auto& o = op.at("order");
        risk::OrderIntent oi{o.at("cl_ord_id").str(), o.at("account").str(), o.at("session").str(),
                             o.at("symbol").str(), o.at("side").str() == "BUY" ? risk::Side::BUY : risk::Side::SELL,
                             o.at("qty").f64(), o.at("px").f64(), o.at("parent_id").str()};
        got = b.check_new(oi, op.at("now").i64());
        has_expect = true;
      } else if (kind == "amend") {
        got = b.check_amend(op.at("id").str(), op.at("qty").f64(), op.at("now").i64());
        has_expect = true;
      } else if (kind == "amend_result") {
        b.on_amend_result(op.at("id").str(), op.at("accepted").b);
      } else if (kind == "fill") {
        b.on_fill(op.at("id").str(), op.at("qty").f64());
      } else {
        b.on_terminal(op.at("id").str());
      }
      if (has_expect) {
        const auto& e = op.at("expect");
        bool ok = e.is_null() ? !got.has_value() : (got.has_value() && *got == e.str());
        if (!ok) std::printf("  seed %lld op %ld: got %s\n", (long long)c.at("seed").i64(), ops, got ? got->c_str() : "null");
        CHECK(ok);
      }
      const auto& p = op.at("probe");
      auto [lng, sht] = b.worst_case(p.at("account").str(), p.at("symbol").str());
      CHECK(lng == num(p.at("long")));
      CHECK(sht == num(p.at("short")));
      CHECK(b.working_notional(p.at("account").str()) == num(p.at("notional")));
      ++ops;
    }
  }
  CHECK(ops > 1000);
}

static risk::Limits small_limits() { return {100, 20'000, 0.05, 300, 50'000, 5, 1'000, 1'000'000'000}; }

TEST(pretrade_throttle_rejects_never_queues) {
  risk::RiskBook b(small_limits());
  b.ref_px["X"] = 100;
  for (int i = 0; i < 5; ++i)
    CHECK(!b.check_new({"c" + std::to_string(i), "A", "S", "X", risk::Side::BUY, 1, 100.0 - i * 0.1, ""}, i).has_value());
  CHECK(b.check_new({"c9", "A", "S", "X", risk::Side::BUY, 1, 99, ""}, 5).value() == "THROTTLE");
  CHECK(!b.check_new({"c10", "A", "S", "X", risk::Side::BUY, 1, 98.9, ""}, 1'000).has_value());  // window rolled
}

TEST(pretrade_amend_counts_max_of_old_and_new_until_answered) {
  risk::RiskBook b(small_limits());
  b.ref_px["X"] = 100;
  CHECK(!b.check_new({"c1", "A", "S", "X", risk::Side::BUY, 100, 100, ""}, 0).has_value());
  CHECK(!b.check_amend("c1", 50, 10).has_value());
  CHECK_EQ(b.worst_case("A", "X").first, 100.0);  // still 100 while the amend is in flight
  b.on_amend_result("c1", true);
  CHECK_EQ(b.worst_case("A", "X").first, 50.0);
}

// ---- gateway ---------------------------------------------------------------------

static gateway::NewOrder order(std::int64_t id, const char* strat, const char* sym, double px = 100) {
  gateway::NewOrder o;
  o.order_id = id;
  o.path = {"QP", "RATES", strat, sym};
  o.intent = {"c" + std::to_string(id), "A", "S1", sym, risk::Side::BUY, 1, px, ""};
  return o;
}

TEST(gateway_kill_then_reset) {
  gateway::Gateway g(small_limits());
  g.risk.ref_px["X"] = 100;
  g.risk.ref_px["Y"] = 100;
  CHECK(g.submit(order(1, "MM1", "X"), 0).accepted);
  auto cancels = g.engage({"QP", "RATES", "*", "*"}, "alice", "desk loss limit");
  CHECK(cancels == std::vector<std::int64_t>{1});
  auto d = g.submit(order(2, "MM2", "Y", 99), 1);
  CHECK(!d.accepted);
  CHECK_EQ(d.reason, std::string("KILLED:DESK:QP/RATES/*/*"));
  g.on_terminal(1);
  CHECK(g.kill.state(1) == kill::KillState::KILLED);
  CHECK(g.kill.request_reset(1, "alice"));
  CHECK(!g.kill.approve_reset(1, "alice"));
  CHECK(g.kill.approve_reset(1, "bob"));
  CHECK(g.submit(order(3, "MM2", "Y", 99), 2).accepted);
}

TEST(gateway_kill_gate_runs_before_risk_and_spends_no_throttle) {
  gateway::Gateway g(small_limits());
  g.risk.ref_px["X"] = 100;
  g.engage({"QP", "*", "*", "*"}, "alice", "");
  for (int i = 0; i < 20; ++i) CHECK(!g.submit(order(10 + i, "MM1", "X"), i).accepted);
  CHECK(g.kill.request_reset(1, "alice"));
  CHECK(g.kill.approve_reset(1, "bob"));
  CHECK(g.submit(order(50, "MM1", "X"), 21).accepted);  // throttle untouched by the killed burst
}

TEST(gateway_stale_book_pulls_quotes_and_blocks) {
  gateway::Gateway g(small_limits());
  g.risk.ref_px["X"] = 100;
  CHECK(g.submit(order(1, "MM1", "X"), 0).accepted);
  CHECK(g.submit(order(2, "MM1", "X", 99.5), 1).accepted);
  auto pull = g.on_book_stale("X", true);
  CHECK((pull == std::vector<std::int64_t>{1, 2}));
  CHECK_EQ(g.submit(order(3, "MM1", "X", 99), 2).reason, std::string("STALE_BOOK"));
  g.on_book_stale("X", false);
  CHECK(g.submit(order(4, "MM1", "X", 99), 3).accepted);
}

TEST(gateway_cancel_on_disconnect_kills_session_strategies) {
  gateway::Gateway g(small_limits());
  g.risk.ref_px["X"] = 100;
  g.risk.ref_px["Y"] = 100;
  CHECK(g.submit(order(1, "MM1", "X"), 0).accepted);
  CHECK(g.submit(order(2, "MM2", "Y"), 1).accepted);
  auto c = g.on_disconnect("S1");
  CHECK((c == std::vector<std::int64_t>{1, 2}));
  CHECK_EQ(g.kill.events[0].actor, std::string("system:cod"));
  CHECK_EQ(g.submit(order(3, "MM1", "X", 99), 2).reason, std::string("KILLED:STRATEGY:QP/RATES/MM1/*"));
  CHECK_EQ(g.submit(order(4, "MM1", "X", 99), 2).reason.substr(0, 7), std::string("KILLED:"));
  CHECK_EQ(g.submit(order(5, "MM1", "Z", 99), 2).reason, std::string("KILLED:STRATEGY:QP/RATES/MM1/*"));
}

QPT_MAIN()
