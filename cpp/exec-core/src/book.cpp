// Fast book; every function mirrors the one of the same name in
// python/execution/src/qp_exec/lob/book.py.
#include "qp/exec/book.hpp"

#include <algorithm>
#include <cstdlib>
#include <set>

namespace qp::exec {

namespace {
constexpr std::uint64_t kFnvOffset = 0xCBF29CE484222325ULL;
constexpr std::uint64_t kFnvPrime = 0x100000001B3ULL;
inline std::uint64_t fnv(std::uint64_t h, std::int64_t w) {
  return (h ^ static_cast<std::uint64_t>(w)) * kFnvPrime;
}
inline Side opp(Side s) { return s == Side::Buy ? Side::Sell : Side::Buy; }
// Python floor division semantics for the midpoint.
inline std::int64_t floor_div2(std::int64_t s) { return s >= 0 ? s / 2 : -((-s + 1) / 2); }
}  // namespace

std::optional<std::int64_t> Book::best(Side s) const {
  if (s == Side::Buy) return bids_.empty() ? std::nullopt : std::optional{bids_.begin()->first};
  return asks_.empty() ? std::nullopt : std::optional{asks_.begin()->first};
}

std::optional<std::int64_t> Book::best_displayed(Side s) const {
  std::optional<std::int64_t> r;
  for_levels(s, [&](std::int64_t p, const Level& l) {
    if (l.dh >= 0) { r = p; return false; }
    return true;
  });
  return r;
}

std::optional<std::int64_t> Book::ref_best(Side s) const {
  std::optional<std::int64_t> r;
  for_levels(s, [&](std::int64_t p, const Level& l) {
    bool hit = false;
    for_queue(l.dh, [&](std::int32_t i) { hit = hit || at(i).peg == Peg::None; });
    for_queue(l.hh, [&](std::int32_t i) { hit = hit || at(i).peg == Peg::None; });
    if (hit) { r = p; return false; }
    return true;
  });
  return r;
}

Book::Level* Book::level(Side s, std::int64_t p) {
  if (s == Side::Buy) { auto it = bids_.find(p); return it == bids_.end() ? nullptr : &it->second; }
  auto it = asks_.find(p);
  return it == asks_.end() ? nullptr : &it->second;
}

std::int32_t Book::alloc(const Ord& o) {
  std::int32_t i;
  if (!free_.empty()) { i = free_.back(); free_.pop_back(); at(i) = o; }
  else { i = static_cast<std::int32_t>(pool_.size()); pool_.push_back(o); }
  ids_[o.id] = i;
  return i;
}

void Book::insert(std::int32_t i) {
  Ord& o = at(i);
  Level& l = o.side == Side::Buy ? bids_[o.price] : asks_[o.price];
  std::int32_t& h = o.displayed_class() ? l.dh : l.hh;
  std::int32_t& t = o.displayed_class() ? l.dt : l.ht;
  o.prev = t;
  o.next = -1;
  if (t >= 0) at(t).next = i; else h = i;
  t = i;
}

void Book::unlink(std::int32_t i) {
  Ord& o = at(i);
  Level* l = level(o.side, o.price);
  std::int32_t& h = o.displayed_class() ? l->dh : l->hh;
  std::int32_t& t = o.displayed_class() ? l->dt : l->ht;
  if (o.prev >= 0) at(o.prev).next = o.next; else h = o.next;
  if (o.next >= 0) at(o.next).prev = o.prev; else t = o.prev;
  o.prev = o.next = -1;
  if (l->empty()) {
    if (o.side == Side::Buy) bids_.erase(o.price); else asks_.erase(o.price);
  }
}

void Book::drop(std::int32_t i) {
  Ord& o = at(i);
  if (o.active) unlink(i);
  ids_.erase(o.id);
  if (o.peg != Peg::None) pegs_.erase(std::find(pegs_.begin(), pegs_.end(), o.id));
  free_.push_back(i);
}

void Book::consume(std::int32_t i, std::int64_t q) {
  while (q > 0) {
    Ord& o = at(i);
    const std::int64_t t = std::min(q, o.visible);
    o.visible -= t;
    q -= t;
    if (o.visible == 0) {
      if (o.reserve == 0) { drop(i); return; }
      unlink(i);
      Ord& r = at(i);
      r.visible = std::min(r.display, r.reserve);
      r.reserve -= r.visible;
      r.seq = ++seq_;
      insert(i);
    }
  }
}

bool Book::crosses(Side s, std::optional<std::int64_t> lim, std::int64_t px) {
  if (!lim) return true;
  return s == Side::Buy ? px <= *lim : px >= *lim;
}

std::int64_t Book::available(Side s, std::optional<std::int64_t> lim, std::int64_t need) const {
  std::int64_t tot = 0;
  for_levels(opp(s), [&](std::int64_t p, const Level& l) {
    if (!crosses(s, lim, p)) return false;
    for_queue(l.dh, [&](std::int32_t i) { tot += at(i).total(); });
    for_queue(l.hh, [&](std::int32_t i) { tot += at(i).total(); });
    return tot < need;
  });
  return tot;
}

std::int64_t Book::match(std::int64_t id, Side s, std::optional<std::int64_t> lim, std::int64_t qty,
                         std::int64_t ts, std::vector<Fill>& fills) {
  const Side o = opp(s);
  std::int64_t rem = qty;
  while (rem > 0) {
    auto px = best(o);
    if (!px || !crosses(s, lim, *px)) break;
    const std::int32_t h = level(o, *px)->head();
    const std::int64_t q = std::min(rem, at(h).visible);
    fills.push_back({id, at(h).id, *px, q, ts, static_cast<std::uint8_t>(s)});
    rem -= q;
    consume(h, q);
  }
  return rem;
}

std::optional<std::int64_t> Book::peg_target(const Ord& o) const {
  auto bb = ref_best(Side::Buy);
  auto ba = ref_best(Side::Sell);
  if (o.peg == Peg::Primary) return o.side == Side::Buy ? bb : ba;
  if (!bb || !ba) return std::nullopt;
  const std::int64_t s = *bb + *ba;
  return o.side == Side::Buy ? floor_div2(s) : -floor_div2(-s);
}

void Book::after_event(std::int64_t ts, std::vector<Fill>& fills) {
  if (phase_ != Phase::Continuous || pegs_.empty()) return;
  std::vector<std::pair<std::int64_t, std::optional<std::int64_t>>> targets;
  targets.reserve(pegs_.size());
  for (auto id : pegs_) targets.emplace_back(id, peg_target(at(ids_.at(id))));
  for (auto& [id, tgt] : targets) {
    const std::int32_t i = ids_.at(id);
    if (!tgt) {
      if (at(i).active) { unlink(i); at(i).active = false; }
    } else if (!at(i).active || at(i).price != *tgt) {
      if (at(i).active) unlink(i);
      at(i).price = *tgt;
      at(i).seq = ++seq_;
      at(i).active = true;
      insert(i);
    }
  }
  for (;;) {
    auto bb = best(Side::Buy);
    auto ba = best(Side::Sell);
    if (!bb || !ba || *bb < *ba) return;
    const std::int32_t b = level(Side::Buy, *bb)->head();
    const std::int32_t a = level(Side::Sell, *ba)->head();
    const bool b_maker = at(b).seq < at(a).seq;
    const std::int32_t mk = b_maker ? b : a;
    const std::int32_t tk = b_maker ? a : b;
    const std::int64_t q = std::min(at(b).visible, at(a).visible);
    fills.push_back({at(tk).id, at(mk).id, at(mk).price, q, ts, static_cast<std::uint8_t>(at(tk).side)});
    consume(b, q);
    consume(a, q);
  }
}

bool Book::valid(const NewOrder& o) {
  if (o.qty <= 0 || o.display_qty < 0) return false;
  if (o.peg != Peg::None)
    return o.type == OrdType::Limit && o.tif == Tif::Day && !o.post_only && !o.hidden && o.display_qty == 0;
  if (o.type == OrdType::Market)
    return o.tif != Tif::Day && !o.post_only && !o.hidden && o.display_qty == 0;
  if (o.price <= 0) return false;
  if (o.hidden && o.display_qty) return false;
  if (o.post_only && o.tif != Tif::Day) return false;
  if ((o.display_qty || o.hidden) && o.tif != Tif::Day) return false;
  return true;
}

std::int32_t Book::rest_new(const NewOrder& o, std::int64_t qty) {
  const std::int64_t vis = o.display_qty ? std::min(o.display_qty, qty) : qty;
  Ord r{o.id, o.price, vis, qty - vis, o.display_qty, ++seq_, o.side, Peg::None, o.hidden, true};
  const std::int32_t i = alloc(r);
  insert(i);
  return i;
}

AddResult Book::add(const NewOrder& o, std::int64_t ts, std::vector<Fill>& fills) {
  if (!valid(o)) return {false, Reject::Invalid, 0};
  if (ids_.contains(o.id)) return {false, Reject::DuplicateId, 0};
  if (phase_ == Phase::Auction) {
    if (o.type != OrdType::Limit || o.tif != Tif::Day || o.post_only || o.peg != Peg::None)
      return {false, Reject::NotInAuction, 0};
    const std::int32_t i = rest_new(o, o.qty);
    return {true, Reject::None, at(i).total()};
  }
  auto live_total = [&](std::int64_t id) -> std::int64_t {
    auto it = ids_.find(id);
    return it == ids_.end() ? 0 : at(it->second).total();
  };
  if (o.peg != Peg::None) {
    Ord r{o.id, 0, o.qty, 0, 0, 0, o.side, o.peg, false, false};
    alloc(r);
    pegs_.push_back(o.id);
    after_event(ts, fills);
    return {true, Reject::None, live_total(o.id)};
  }
  std::optional<std::int64_t> lim;
  if (o.type != OrdType::Market) lim = o.price;
  if (o.post_only) {
    auto px = best(opp(o.side));
    if (px && crosses(o.side, lim, *px)) return {false, Reject::PostOnlyWouldCross, 0};
  }
  if (o.tif == Tif::Fok && available(o.side, lim, o.qty) < o.qty) return {false, Reject::FokUnfillable, 0};
  const std::int64_t rem = match(o.id, o.side, lim, o.qty, ts, fills);
  if (rem > 0 && o.tif == Tif::Day) rest_new(o, rem);
  after_event(ts, fills);
  return {true, Reject::None, live_total(o.id)};
}

bool Book::cancel(std::int64_t id, std::int64_t ts, std::vector<Fill>& fills) {
  auto it = ids_.find(id);
  if (it == ids_.end()) return false;
  drop(it->second);
  after_event(ts, fills);
  return true;
}

bool Book::modify(std::int64_t id, std::int64_t new_qty, std::int64_t ts, std::vector<Fill>& fills) {
  auto it = ids_.find(id);
  if (it == ids_.end()) return false;
  if (new_qty <= 0) return cancel(id, ts, fills);
  const std::int32_t i = it->second;
  Ord& o = at(i);
  if (new_qty < o.total()) {
    const std::int64_t cut = o.total() - new_qty;
    const std::int64_t r = std::min(cut, o.reserve);
    o.reserve -= r;
    o.visible -= cut - r;
  } else if (new_qty > o.total()) {
    const std::int64_t add = new_qty - o.total();
    if (o.display) o.reserve += add; else o.visible += add;
    const bool act = o.active;
    if (act) unlink(i);
    at(i).seq = ++seq_;
    if (act) insert(i);
  }
  after_event(ts, fills);
  return true;
}

void Book::start_auction() {
  phase_ = Phase::Auction;
  for (auto id : pegs_) {
    const std::int32_t i = ids_.at(id);
    if (at(i).active) { unlink(i); at(i).active = false; }
  }
}

std::pair<std::optional<std::int64_t>, std::int64_t> Book::auction_price(std::int64_t ref) const {
  std::set<std::int64_t> cands;
  std::map<std::int64_t, std::int64_t> btot, stot;
  for (auto& [p, l] : bids_) {
    cands.insert(p);
    for_queue(l.dh, [&](std::int32_t i) { btot[p] += at(i).total(); });
    for_queue(l.hh, [&](std::int32_t i) { btot[p] += at(i).total(); });
  }
  for (auto& [p, l] : asks_) {
    cands.insert(p);
    for_queue(l.dh, [&](std::int32_t i) { stot[p] += at(i).total(); });
    for_queue(l.hh, [&](std::int32_t i) { stot[p] += at(i).total(); });
  }
  std::optional<std::int64_t> best_px;
  std::int64_t bv = 0, bimb = 0, bdist = 0;
  for (auto p : cands) {
    std::int64_t d = 0, s = 0;
    for (auto& [q, v] : btot) if (q >= p) d += v;
    for (auto& [q, v] : stot) if (q <= p) s += v;
    const std::int64_t v = std::min(d, s);
    const std::int64_t imb = std::llabs(d - s), dist = std::llabs(p - ref);
    if (v <= 0) continue;
    // key (-v, imb, dist, p) ascending; candidates come in ascending p.
    if (!best_px || v > bv || (v == bv && (imb < bimb || (imb == bimb && dist < bdist)))) {
      best_px = p; bv = v; bimb = imb; bdist = dist;
    }
  }
  return {best_px, bv};
}

void Book::uncross(std::int64_t ts, std::int64_t ref_price, std::vector<Fill>& fills) {
  auto [px, vol] = auction_price(ref_price);
  if (px) {
    std::vector<std::int32_t> buys, sells;
    for (auto& [p, l] : bids_) {
      if (p < *px) break;
      for_queue(l.dh, [&](std::int32_t i) { buys.push_back(i); });
      for_queue(l.hh, [&](std::int32_t i) { buys.push_back(i); });
    }
    for (auto& [p, l] : asks_) {
      if (p > *px) break;
      for_queue(l.dh, [&](std::int32_t i) { sells.push_back(i); });
      for_queue(l.hh, [&](std::int32_t i) { sells.push_back(i); });
    }
    std::vector<std::int64_t> btot, brem, stot, srem;
    for (auto i : buys) { btot.push_back(at(i).total()); }
    for (auto i : sells) { stot.push_back(at(i).total()); }
    brem = btot;
    srem = stot;
    std::size_t bi = 0, si = 0;
    std::int64_t left = vol;
    while (left > 0) {
      const std::int64_t q = std::min({brem[bi], srem[si], left});
      fills.push_back({at(buys[bi]).id, at(sells[si]).id, *px, q, ts, 0});
      brem[bi] -= q;
      srem[si] -= q;
      left -= q;
      if (brem[bi] == 0) ++bi;
      if (srem[si] == 0) ++si;
    }
    for (std::size_t k = 0; k < buys.size(); ++k)
      if (btot[k] != brem[k]) consume(buys[k], btot[k] - brem[k]);
    for (std::size_t k = 0; k < sells.size(); ++k)
      if (stot[k] != srem[k]) consume(sells[k], stot[k] - srem[k]);
  }
  phase_ = Phase::Continuous;
  after_event(ts, fills);
}

void Book::state_words(std::vector<std::int64_t>& w) const {
  w.push_back(static_cast<std::int64_t>(phase_));
  for (Side s : {Side::Buy, Side::Sell}) {
    for_levels(s, [&](std::int64_t p, const Level& l) {
      w.push_back(p);
      for_queue(l.dh, [&](std::int32_t i) { w.push_back(at(i).id); w.push_back(at(i).visible); w.push_back(at(i).reserve); });
      w.push_back(-1);
      for_queue(l.hh, [&](std::int32_t i) { w.push_back(at(i).id); w.push_back(at(i).visible); });
      w.push_back(-2);
      return true;
    });
    w.push_back(-3);
  }
  for (auto id : pegs_)
    if (!at(ids_.at(id)).active) w.push_back(id);
  w.push_back(-4);
}

std::uint64_t Book::digest(const std::vector<Fill>& fills) const {
  std::uint64_t h = kFnvOffset;
  for (const auto& f : fills) {
    h = fnv(h, f.taker_id); h = fnv(h, f.maker_id); h = fnv(h, f.price); h = fnv(h, f.qty);
  }
  h = fnv(h, -5);
  std::vector<std::int64_t> w;
  state_words(w);
  for (auto x : w) h = fnv(h, x);
  return h;
}

}  // namespace qp::exec
