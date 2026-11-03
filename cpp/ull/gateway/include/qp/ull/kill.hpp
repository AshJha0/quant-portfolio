// Kill-switch hierarchy: the C++ implementation of docs/KILL_CONTRACT.md.
//
// The Python OMS (python/oms/kill/contract.py) is the reference. This port
// must replay python/oms/kill/vectors.json to the same event stream, byte for
// byte after canonical JSON encoding; tests/test_kill_parity.cpp checks that.
// Rules K1-K7 are named where they are enforced. Iteration orders mirror the
// reference exactly (kills by ascending id, which is Python dict insertion
// order there), because event order is part of the contract.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qp/ull/json.hpp"

namespace qp::ull::kill {

inline const std::string kWild = "*";

using Path = std::array<std::string, 4>;  // (firm, desk, strategy, symbol)

enum class Level { FIRM = 1, DESK = 2, STRATEGY = 3, SYMBOL = 4 };
enum class KillState { ENGAGED, KILLED, RESET_PENDING, CLEARED };

inline const char* to_string(Level l) {
  switch (l) {
    case Level::FIRM: return "FIRM";
    case Level::DESK: return "DESK";
    case Level::STRATEGY: return "STRATEGY";
    case Level::SYMBOL: return "SYMBOL";
  }
  return "?";
}
inline const char* to_string(KillState s) {
  switch (s) {
    case KillState::ENGAGED: return "ENGAGED";
    case KillState::KILLED: return "KILLED";
    case KillState::RESET_PENDING: return "RESET_PENDING";
    case KillState::CLEARED: return "CLEARED";
  }
  return "?";
}

struct ScopeError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};
struct UnknownKill : std::out_of_range {
  using std::out_of_range::out_of_range;
};

// Truncate an order path to a kill scope at `level` (suffix wildcards).
inline Path make_scope(Level level, const Path& p) {
  int n = static_cast<int>(level);
  Path s = p;
  for (int i = n; i < 4; ++i) s[std::size_t(i)] = kWild;
  return s;
}

// Level of a scope; throws if wildcards are not a pure suffix after firm.
inline Level validate_scope(const Path& scope) {
  int n = 0;
  while (n < 4 && scope[std::size_t(n)] != kWild) ++n;
  if (n == 0) throw ScopeError("invalid scope: firm may not be a wildcard");
  for (int i = n; i < 4; ++i)
    if (scope[std::size_t(i)] != kWild) throw ScopeError("invalid scope: wildcards must be a suffix");
  return static_cast<Level>(n);
}

inline bool covers(const Path& scope, const Path& path) {
  for (std::size_t i = 0; i < 4; ++i)
    if (scope[i] != kWild && scope[i] != path[i]) return false;
  return true;
}

struct Kill {
  int kill_id = 0;
  Level level = Level::FIRM;
  Path scope;
  std::string actor;
  std::string reason;
  KillState state = KillState::ENGAGED;
  std::set<std::int64_t> outstanding;
  std::optional<std::string> reset_requester;
};

struct KillEvent {
  int seq = 0;
  std::string event;
  int kill_id = 0;
  std::string level;
  Path scope;
  std::string actor;
  std::string state;
  std::vector<std::int64_t> cancel;
  std::string reason;

  // Canonical JSON: keys sorted, no spaces, ensure_ascii (the parity form).
  std::string to_wire() const {
    using json::escape;
    std::string o = "{\"actor\":" + escape(actor) + ",\"cancel\":[";
    for (std::size_t i = 0; i < cancel.size(); ++i) {
      if (i) o += ',';
      o += std::to_string(cancel[i]);
    }
    o += "],\"event\":" + escape(event) + ",\"kill_id\":" + std::to_string(kill_id) +
         ",\"level\":" + escape(level) + ",\"reason\":" + escape(reason) + ",\"scope\":[";
    for (std::size_t i = 0; i < 4; ++i) {
      if (i) o += ',';
      o += escape(scope[i]);
    }
    o += "],\"seq\":" + std::to_string(seq) + ",\"state\":" + escape(state) + "}";
    return o;
  }
};

// One firm-wide kill hierarchy. It sends nothing: it tells the caller which
// orders to cancel and whether a new order may go out.
class KillSwitch {
 public:
  std::vector<KillEvent> events;

  // ---- order tracking ----------------------------------------------------
  void on_order_working(std::int64_t order_id, const Path& path) { working_[order_id] = path; }

  // K3: a terminal report (cancel ack, fill, reject, expiry) for an order.
  void on_order_terminal(std::int64_t order_id) {
    working_.erase(order_id);
    for (auto& [id, k] : kills_) {
      if (k.outstanding.erase(order_id) != 0 && k.state == KillState::ENGAGED && k.outstanding.empty()) {
        k.state = KillState::KILLED;
        emit("killed", k, "system", "");
      }
    }
  }

  // ---- engage --------------------------------------------------------------
  // Returns (kill_id, sorted order ids to cancel). K2, K7.
  std::pair<int, std::vector<std::int64_t>> engage(const Path& scope, const std::string& actor,
                                                   const std::string& reason = "") {
    Level level = validate_scope(scope);
    for (auto& [id, k] : kills_) {
      if (k.scope == scope && k.state != KillState::CLEARED) {
        emit("engage_dup", k, actor, reason);
        return {k.kill_id, {}};
      }
    }
    Kill k;
    k.kill_id = next_id_++;
    k.level = level;
    k.scope = scope;
    k.actor = actor;
    k.reason = reason;
    std::vector<std::int64_t> cancels;
    for (const auto& [oid, p] : working_)  // std::map: ascending ids
      if (covers(scope, p)) cancels.push_back(oid);
    k.outstanding.insert(cancels.begin(), cancels.end());
    if (cancels.empty()) k.state = KillState::KILLED;
    auto& ref = kills_[k.kill_id] = std::move(k);
    emit("engage", ref, actor, reason, cancels);
    return {ref.kill_id, cancels};
  }

  // K6: engage a STRATEGY kill for every strategy on a dropped session.
  std::vector<std::pair<int, std::vector<std::int64_t>>> cancel_on_disconnect(const std::vector<Path>& paths) {
    std::set<Path> strategies;  // lexicographic, same as Python sorted() of tuples
    for (const auto& p : paths) strategies.insert(make_scope(Level::STRATEGY, p));
    std::vector<std::pair<int, std::vector<std::int64_t>>> out;
    for (const auto& s : strategies) out.push_back(engage(s, "system:cod", "cancel_on_disconnect"));
    return out;
  }

  // ---- block ---------------------------------------------------------------
  // K1: nullopt if allowed, else the reject reason naming the widest kill.
  std::optional<std::string> check(const Path& path) const {
    const Kill* best = nullptr;
    for (const auto& [id, k] : kills_) {
      if (k.state == KillState::CLEARED || !covers(k.scope, path)) continue;
      if (best == nullptr || static_cast<int>(k.level) < static_cast<int>(best->level)) best = &k;
    }
    if (best == nullptr) return std::nullopt;
    std::string r = std::string("KILLED:") + to_string(best->level) + ":";
    for (std::size_t i = 0; i < 4; ++i) {
      if (i) r += '/';
      r += best->scope[i];
    }
    return r;
  }

  // ---- reset ---------------------------------------------------------------
  bool request_reset(int kill_id, const std::string& actor) {
    Kill& k = get(kill_id);
    if (k.state != KillState::KILLED) {
      emit("reset_rejected", k, actor, std::string("state=") + to_string(k.state));
      return false;
    }
    k.state = KillState::RESET_PENDING;
    k.reset_requester = actor;
    emit("reset_requested", k, actor, "");
    return true;
  }

  // K4: the approver must differ from the requester.
  bool approve_reset(int kill_id, const std::string& actor) {
    Kill& k = get(kill_id);
    if (k.state != KillState::RESET_PENDING) {
      emit("reset_rejected", k, actor, std::string("state=") + to_string(k.state));
      return false;
    }
    if (k.reset_requester && actor == *k.reset_requester) {
      emit("reset_rejected", k, actor, "same_actor");
      return false;
    }
    k.state = KillState::CLEARED;
    emit("cleared", k, actor, "");
    return true;
  }

  KillState state(int kill_id) const {
    auto it = kills_.find(kill_id);
    if (it == kills_.end()) throw UnknownKill("unknown kill id " + std::to_string(kill_id));
    return it->second.state;
  }

 private:
  std::map<int, Kill> kills_;
  std::map<std::int64_t, Path> working_;
  int next_id_ = 1;
  int seq_ = 0;

  Kill& get(int kill_id) {
    auto it = kills_.find(kill_id);
    if (it == kills_.end()) throw UnknownKill("unknown kill id " + std::to_string(kill_id));
    return it->second;
  }
  void emit(const char* ev, const Kill& k, const std::string& actor, const std::string& reason,
            std::vector<std::int64_t> cancel = {}) {
    events.push_back(KillEvent{++seq_, ev, k.kill_id, to_string(k.level), k.scope, actor, to_string(k.state),
                               std::move(cancel), reason});
  }
};

// Run a golden-vector command list and return the wire lines (contract.replay).
inline Path path_of(const json::Value& v) {
  if (!v.is_array() || v.arr.size() != 4) throw ScopeError("path must have four fields");
  return {v.at(0).str(), v.at(1).str(), v.at(2).str(), v.at(3).str()};
}

inline std::vector<std::string> replay(const json::Value& commands) {
  KillSwitch ks;
  std::vector<std::string> out;
  for (const auto& c : commands.arr) {
    std::size_t before = ks.events.size();
    const std::string& op = c.at("op").str();
    if (op == "working") ks.on_order_working(c.at("id").i64(), path_of(c.at("path")));
    else if (op == "terminal") ks.on_order_terminal(c.at("id").i64());
    else if (op == "engage") ks.engage(path_of(c.at("scope")), c.at("actor").str(), c.has("reason") ? c.at("reason").str() : "");
    else if (op == "disconnect") {
      std::vector<Path> ps;
      for (const auto& p : c.at("paths").arr) ps.push_back(path_of(p));
      ks.cancel_on_disconnect(ps);
    } else if (op == "request_reset") ks.request_reset(int(c.at("kill_id").i64()), c.at("actor").str());
    else if (op == "approve_reset") ks.approve_reset(int(c.at("kill_id").i64()), c.at("actor").str());
    else if (op == "check") {
      Path p = path_of(c.at("path"));
      auto r = ks.check(p);
      std::string line = "{\"check\":[";
      for (std::size_t i = 0; i < 4; ++i) {
        if (i) line += ',';
        line += json::escape(p[i]);
      }
      line += "],\"result\":" + (r ? json::escape(*r) : std::string("null")) + "}";
      out.push_back(std::move(line));
      continue;
    } else throw std::invalid_argument("unknown op " + op);
    for (std::size_t i = before; i < ks.events.size(); ++i) out.push_back(ks.events[i].to_wire());
  }
  return out;
}

}  // namespace qp::ull::kill
