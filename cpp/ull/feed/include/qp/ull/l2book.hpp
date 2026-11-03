// Price-level book used by the crypto feed handler.
//
// Plain idea: a level-2 book is, per side, a map from price to total size.
// A level-3 book (Coinbase `full`) also knows each resting order; its levels
// are the per-price sums of those orders. This is the feed handler's own
// small book, keyed by exact fixed-point prices. The flat price-indexed L3
// book for QPBF is part 1 (cpp/ull/book, lane r13-l2-ull-cpp); crypto prices
// span too wide a range for a dense array without a tick-size table, which
// this handler does not assume.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qp/ull/decimal.hpp"

namespace qp::ull::feed {

enum class Side { Bid, Ask };

struct Level {
  std::int64_t px;
  std::int64_t qty;
  bool operator==(const Level&) const = default;
};

class L2Book {
 public:
  // Set the absolute size at a level; qty 0 removes it (Binance and Coinbase
  // level2 semantics).
  void set(Side s, std::int64_t px, std::int64_t qty) {
    if (qty < 0) throw std::invalid_argument("negative level size");
    if (s == Side::Bid) { if (qty == 0) bids_.erase(px); else bids_[px] = qty; }
    else { if (qty == 0) asks_.erase(px); else asks_[px] = qty; }
  }
  void add(Side s, std::int64_t px, std::int64_t dq) {
    std::int64_t cur = get(s, px) + dq;
    if (cur < 0) throw std::logic_error("level size went negative");
    set(s, px, cur);
  }
  std::int64_t get(Side s, std::int64_t px) const {
    if (s == Side::Bid) { auto it = bids_.find(px); return it == bids_.end() ? 0 : it->second; }
    auto it = asks_.find(px);
    return it == asks_.end() ? 0 : it->second;
  }
  void clear() { bids_.clear(); asks_.clear(); }
  std::vector<Level> levels(Side s) const {  // best first
    std::vector<Level> o;
    if (s == Side::Bid) for (auto& [p, q] : bids_) o.push_back({p, q});
    else for (auto& [p, q] : asks_) o.push_back({p, q});
    return o;
  }
  bool empty() const { return bids_.empty() && asks_.empty(); }
  bool crossed() const { return !bids_.empty() && !asks_.empty() && bids_.begin()->first >= asks_.begin()->first; }
  bool operator==(const L2Book& o) const { return bids_ == o.bids_ && asks_ == o.asks_; }

  // Canonical text form "B px qty\n...A px qty\n" for fixture comparisons.
  std::string dump() const {
    std::string o;
    for (auto& [p, q] : bids_) o += "B " + format_fixed(p) + " " + format_fixed(q) + "\n";
    for (auto& [p, q] : asks_) o += "A " + format_fixed(p) + " " + format_fixed(q) + "\n";
    return o;
  }

 private:
  std::map<std::int64_t, std::int64_t, std::greater<>> bids_;
  std::map<std::int64_t, std::int64_t> asks_;
};

// Level-3 book: individual orders, with L2 levels kept in step.
class L3Book {
 public:
  struct Order {
    Side side;
    std::int64_t px;
    std::int64_t qty;
  };
  void clear() { orders_.clear(); l2_.clear(); }
  void open(const std::string& id, Side s, std::int64_t px, std::int64_t qty) {
    if (orders_.count(id)) throw std::logic_error("duplicate order id " + id);
    orders_[id] = {s, px, qty};
    l2_.add(s, px, qty);
  }
  // Remove if resting; orders that never rested (market, IOC) are not here.
  void done(const std::string& id) {
    auto it = orders_.find(id);
    if (it == orders_.end()) return;
    l2_.add(it->second.side, it->second.px, -it->second.qty);
    orders_.erase(it);
  }
  void reduce(const std::string& id, std::int64_t by) {
    auto it = orders_.find(id);
    if (it == orders_.end()) return;
    if (by > it->second.qty) throw std::logic_error("match larger than resting size for " + id);
    it->second.qty -= by;
    l2_.add(it->second.side, it->second.px, -by);
  }
  void resize(const std::string& id, std::int64_t new_qty) {
    auto it = orders_.find(id);
    if (it == orders_.end()) return;
    l2_.add(it->second.side, it->second.px, new_qty - it->second.qty);
    it->second.qty = new_qty;
  }
  const L2Book& l2() const { return l2_; }
  std::size_t order_count() const { return orders_.size(); }
  const std::unordered_map<std::string, Order>& orders() const { return orders_; }

 private:
  std::unordered_map<std::string, Order> orders_;
  L2Book l2_;
};

}  // namespace qp::ull::feed
