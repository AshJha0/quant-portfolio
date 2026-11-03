// Order gateway: the last software check before an order reaches a venue.
//
// Plain idea: every new order passes three gates in a fixed order, and the
// first gate that says no names the reason.
//   1. Kill switch (K1 of docs/KILL_CONTRACT.md): a kill covering the order's
//      (firm, desk, strategy, symbol) path rejects it as KILLED:<level>:<scope>.
//   2. Stale market data: if the feed handler has flagged the symbol's book
//      stale, new orders on it are rejected (STALE_BOOK) and, when the flag
//      goes up, every working order on the symbol is cancelled ("pull quotes").
//   3. W28d pre-trade risk (risk::RiskBook), the C++ port of the Python checks.
// The kill gate is first so that a killed strategy cannot even consume
// throttle budget; the stale gate is second because a price band checked
// against a stale reference price is meaningless.
//
// Formal statement: submit() is accepted iff kill.check(path) is empty, the
// symbol is not stale and risk.check_new() accepts. Accepted orders are
// registered as working with both the kill switch (so a later kill cancels
// them, K2) and the risk book (so they count as exposure). Cancels are
// returned to the caller, never sent from here: the gateway has no socket.
//
// Worked example (tests/test_gateway.cpp, "kill_then_reset"): order 1 on
// QP/RATES/MM1/X is accepted; alice engages DESK QP/RATES -> cancel [1];
// a new order on QP/RATES/MM2/Y is rejected KILLED:DESK:QP/RATES/*/*; the
// cancel ack arrives, alice requests reset, bob approves, orders flow again.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "qp/ull/kill.hpp"
#include "qp/ull/pretrade.hpp"

namespace qp::ull::gateway {

struct NewOrder {
  std::int64_t order_id = 0;  // gateway-internal id used by the kill contract
  kill::Path path;            // (firm, desk, strategy, symbol); symbol == intent.symbol
  risk::OrderIntent intent;
};

struct Decision {
  bool accepted = false;
  std::string reason;  // empty when accepted
};

class Gateway {
 public:
  explicit Gateway(risk::Limits limits) : risk(limits) {}

  kill::KillSwitch kill;
  risk::RiskBook risk;

  Decision submit(const NewOrder& o, std::int64_t now_ns) {
    if (o.path[3] != o.intent.symbol) return {false, "PATH_SYMBOL_MISMATCH"};
    if (auto r = kill.check(o.path)) return {false, *r};
    if (stale_.count(o.intent.symbol)) return {false, "STALE_BOOK"};
    if (auto r = risk.check_new(o.intent, now_ns)) return {false, *r};
    kill.on_order_working(o.order_id, o.path);
    orders_[o.order_id] = o;
    return {true, ""};
  }

  // Venue reports a fill; `terminal` when the order has no quantity left.
  void on_fill(std::int64_t order_id, double qty, bool terminal) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return;
    risk.on_fill(it->second.intent.cl_ord_id, qty);
    if (terminal) on_terminal(order_id);
  }

  // Cancel ack, reject or expiry.
  void on_terminal(std::int64_t order_id) {
    auto it = orders_.find(order_id);
    if (it != orders_.end()) {
      risk.on_terminal(it->second.intent.cl_ord_id);
      orders_.erase(it);
    }
    kill.on_order_terminal(order_id);
  }

  // K2: engage a kill; returns the order ids the caller must cancel.
  std::vector<std::int64_t> engage(const kill::Path& scope, const std::string& actor, const std::string& reason) {
    return kill.engage(scope, actor, reason).second;
  }

  // K6: a session dropped. Kills every strategy that has working orders on it.
  std::vector<std::int64_t> on_disconnect(const std::string& session) {
    std::vector<kill::Path> paths;
    for (const auto& [id, o] : orders_)
      if (o.intent.session == session) paths.push_back(o.path);
    std::vector<std::int64_t> out;
    for (auto& [kid, c] : kill.cancel_on_disconnect(paths)) out.insert(out.end(), c.begin(), c.end());
    return out;
  }

  // Feed-handler seam: the stale flag for a symbol changed. Going stale
  // returns the working orders on that symbol to cancel (pull quotes).
  std::vector<std::int64_t> on_book_stale(const std::string& symbol, bool stale) {
    std::vector<std::int64_t> out;
    if (!stale) {
      stale_.erase(symbol);
      return out;
    }
    stale_.insert(symbol);
    for (const auto& [id, o] : orders_)
      if (o.intent.symbol == symbol) out.push_back(id);
    return out;
  }

  bool is_working(std::int64_t order_id) const { return orders_.count(order_id) != 0; }

 private:
  std::map<std::int64_t, NewOrder> orders_;
  std::set<std::string> stale_;
};

}  // namespace qp::ull::gateway
