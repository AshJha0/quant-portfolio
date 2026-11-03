// Pre-trade risk checks: the C++ port of python/oms/risk/pretrade.py (W28d).
//
// Plain idea: before an order leaves, ask "if every working order and this one
// filled in full, would any limit break?" If yes, reject. A pending cancel
// counts in full until acknowledged; a pending amend counts at max(old, new)
// until the venue answers. That is what keeps the limit under amend/cancel
// races.
//
// Formal statement, per account a and symbol s, signed position q, working buy
// quantity B and sell quantity S (pending amends at max(old, new)):
//
//     worst_long  = q + B (+ new buy qty)     <= position_limit
//     worst_short = -q + S (+ new sell qty)   <= position_limit
//     sum over working orders of qty * px (+ new notional) <= credit_limit
//
// The check order and reject strings are the reference's, so a reject reason
// means the same thing in every language; tests/test_pretrade_parity.cpp
// replays vectors generated from the Python module. Floating sums run in the
// reference's order (insertion order of working orders) so results agree to
// the bit, not just to a tolerance.
//
// Throttle rule (stated, as in the reference): a message over the per-session
// limit is REJECTED, never queued.
//
// Sources: SEC Rule 15c3-5(c)(1); FIA Best Practices for Automated Trading Risk
// Controls (2024): price bands, fat finger, throttles, self-trade prevention.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qp::ull::risk {

enum class Side { BUY, SELL };

struct OrderIntent {
  std::string cl_ord_id;
  std::string account;
  std::string session;
  std::string symbol;
  Side side = Side::BUY;
  double qty = 0;
  double px = 0;
  std::string parent_id;  // set for EMS child slices
};

struct Limits {
  double max_order_qty = 0;
  double max_order_notional = 0;
  double price_band = 0;  // fraction of reference price
  double position_limit = 0;
  double credit_limit = 0;
  int throttle_msgs = 0;
  std::int64_t throttle_window_ns = 0;
  std::int64_t dup_window_ns = 1'000'000'000;
};

using Reason = std::optional<std::string>;  // nullopt = accepted

class RiskBook {
 public:
  explicit RiskBook(Limits l) : limits(l) {}

  Limits limits;
  std::map<std::string, double> ref_px;
  std::set<std::string> restricted;
  std::map<std::pair<std::string, std::string>, double> positions;

  std::pair<double, double> worst_case(const std::string& account, const std::string& symbol) const {
    auto it = positions.find({account, symbol});
    double q = it == positions.end() ? 0.0 : it->second;
    double b = 0.0, s = 0.0;
    for (const auto& [id, w] : working_) {
      if (w.intent.account == account && w.intent.symbol == symbol) {
        if (w.intent.side == Side::BUY) b += exposure(w); else s += exposure(w);
      }
    }
    return {q + b, -q + s};
  }

  double working_notional(const std::string& account) const {
    // The reference uses Python's built-in sum(), which since CPython 3.12 is
    // Neumaier-compensated (bltinmodule.c, builtin_sum_impl). A plain loop
    // differs in the last bits (sum([0.1]*10) is 1.0 there, 0.999... naively),
    // which would flip a CREDIT_LIMIT decision sitting exactly on the limit.
    double f = 0.0, c = 0.0;
    for (const auto& [id, w] : working_) {
      if (w.intent.account != account) continue;
      double x = exposure(w) * w.intent.px;
      double t = f + x;
      if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x; else c += (x - t) + f;
      f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
  }

  Reason check_new(const OrderIntent& o, std::int64_t now_ns) {
    const Limits& L = limits;
    if (auto r = throttle(o.session, now_ns)) return r;
    if (o.qty <= 0 || o.px <= 0) return "BAD_QTY_OR_PX";
    if (restricted.count(o.symbol)) return "RESTRICTED";
    if (seen_ids_.count(o.cl_ord_id)) return "DUP_CLORDID";
    Key key{o.account, o.symbol, o.side, o.qty, o.px};
    while (!recent_.empty() && recent_.front().first <= now_ns - L.dup_window_ns) recent_.pop_front();
    // EMS slices of one parent repeat size and price on purpose: exempt from
    // the content check (ClOrdID dedupe still applies).
    if (o.parent_id.empty())
      for (const auto& [t, k] : recent_)
        if (k == key) return "DUP_CONTENT";
    if (o.qty > L.max_order_qty) return "FAT_FINGER_QTY";
    if (o.qty * o.px > L.max_order_notional) return "MAX_NOTIONAL";
    auto ref = ref_px.find(o.symbol);
    if (ref == ref_px.end()) return "NO_REF_PRICE";
    if (std::fabs(o.px - ref->second) > L.price_band * ref->second) return "PRICE_BAND";
    if (self_trade(o)) return "SELF_TRADE";
    auto [lng, sht] = worst_case(o.account, o.symbol);
    if ((o.side == Side::BUY ? lng + o.qty : sht + o.qty) > L.position_limit) return "POSITION_LIMIT";
    if (working_notional(o.account) + o.qty * o.px > L.credit_limit) return "CREDIT_LIMIT";
    seen_ids_.insert(o.cl_ord_id);
    recent_.emplace_back(now_ns, key);
    working_.emplace_back(o.cl_ord_id, Working{o, o.qty, std::nullopt});
    return std::nullopt;
  }

  Reason check_amend(const std::string& cl_ord_id, double new_qty, std::int64_t now_ns) {
    Working* w = find(cl_ord_id);
    if (w == nullptr) return "UNKNOWN_ORDER";
    if (w->pending_qty) return "AMEND_IN_FLIGHT";
    if (auto r = throttle(w->intent.session, now_ns)) return r;
    if (new_qty <= 0 || new_qty > limits.max_order_qty) return "FAT_FINGER_QTY";
    double extra = std::max(0.0, new_qty - w->qty);
    auto [lng, sht] = worst_case(w->intent.account, w->intent.symbol);
    if ((w->intent.side == Side::BUY ? lng : sht) + extra > limits.position_limit) return "POSITION_LIMIT";
    if (working_notional(w->intent.account) + extra * w->intent.px > limits.credit_limit) return "CREDIT_LIMIT";
    w->pending_qty = new_qty;
    return std::nullopt;
  }

  void on_amend_result(const std::string& cl_ord_id, bool accepted) {
    Working& w = must(cl_ord_id);
    if (accepted && w.pending_qty) w.qty = *w.pending_qty;
    w.pending_qty.reset();
  }

  void on_fill(const std::string& cl_ord_id, double qty) {
    Working& w = must(cl_ord_id);
    double sign = w.intent.side == Side::BUY ? 1.0 : -1.0;
    auto k = std::make_pair(w.intent.account, w.intent.symbol);
    auto it = positions.find(k);
    positions[k] = (it == positions.end() ? 0.0 : it->second) + sign * qty;
    w.qty -= qty;
    if (w.pending_qty) w.pending_qty = std::max(0.0, *w.pending_qty - qty);
    // Python: `not w.pending_qty` is true for None and for 0.0.
    if (w.qty <= 1e-12 && (!w.pending_qty || *w.pending_qty == 0.0)) erase(cl_ord_id);
  }

  // Cancel ack, reject or expiry: exposure released only now.
  void on_terminal(const std::string& cl_ord_id) { erase(cl_ord_id); }

  bool is_working(const std::string& cl_ord_id) const {
    return std::any_of(working_.begin(), working_.end(), [&](const auto& p) { return p.first == cl_ord_id; });
  }

 private:
  struct Working {
    OrderIntent intent;
    double qty;
    std::optional<double> pending_qty;
  };
  using Key = std::tuple<std::string, std::string, Side, double, double>;

  // Insertion-ordered, like the reference's dict, so float sums match exactly.
  std::vector<std::pair<std::string, Working>> working_;
  std::set<std::string> seen_ids_;
  std::deque<std::pair<std::int64_t, Key>> recent_;
  std::map<std::string, std::deque<std::int64_t>> msgs_;

  static double exposure(const Working& w) { return w.pending_qty ? std::max(w.qty, *w.pending_qty) : w.qty; }

  Working* find(const std::string& id) {
    for (auto& [k, w] : working_) if (k == id) return &w;
    return nullptr;
  }
  Working& must(const std::string& id) {
    Working* w = find(id);
    if (w == nullptr) throw std::out_of_range("unknown order " + id);
    return *w;
  }
  void erase(const std::string& id) {
    working_.erase(std::remove_if(working_.begin(), working_.end(), [&](const auto& p) { return p.first == id; }),
                   working_.end());
  }
  bool self_trade(const OrderIntent& o) const {  // cancel-newest STP
    for (const auto& [id, w] : working_) {
      const OrderIntent& r = w.intent;
      if (r.account != o.account || r.symbol != o.symbol || r.side == o.side) continue;
      if ((o.side == Side::BUY && o.px >= r.px) || (o.side == Side::SELL && o.px <= r.px)) return true;
    }
    return false;
  }
  Reason throttle(const std::string& session, std::int64_t now_ns) {
    auto& dq = msgs_[session];
    while (!dq.empty() && dq.front() <= now_ns - limits.throttle_window_ns) dq.pop_front();
    if (static_cast<int>(dq.size()) >= limits.throttle_msgs) return "THROTTLE";
    dq.push_back(now_ns);
    return std::nullopt;
  }
};

}  // namespace qp::ull::risk
