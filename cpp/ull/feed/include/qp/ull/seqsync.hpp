// Sequence synchronisation: A/B arbitration, gap detection, snapshot
// buffering and buffered-delta apply, in one small state machine.
//
// Plain idea. Each update carries a sequence range [first, last]: a single
// number for Coinbase `full` (first == last == sequence), and Binance's
// [U, u] for diff-depth. We remember the last sequence applied to the book,
// so the next update we need is `next = applied + 1`. Then:
//   - an update with last < next is old: a duplicate (often the other line
//     of an A/B pair delivering the same message second). Drop it.
//   - an update with first <= next <= last is the one we need. Apply it.
//   - an update with first > next has arrived early: there is a hole. Hold
//     it, and mark the book stale, because the book is now known to be
//     behind the market.
// Holes fill themselves when the update was only reordered, or arrives on
// the other line. When the hole is older than `gap_timeout_ns`, or more than
// `reorder_window` updates are held, the hole is declared lost and a
// snapshot is requested. Updates keep being buffered while we wait; when the
// snapshot (with its own sequence S) arrives, the book is rebuilt from it,
// held updates with last <= S are discarded, and the rest are applied in
// order. The book is stale from the first sight of a hole until it has none.
//
// Formal statement. Invariants after every call:
//   (I1) applied is strictly increasing over applied updates, and each
//        applied update u satisfies first(u) <= applied_before + 1 <= last(u)
//        (Binance's documented rule; for single sequences, exact succession).
//   (I2) stale() == !synced || !held.empty(), where held contains only
//        updates with first > applied + 1.
//   (I3) no update is applied twice (A/B duplicates and replays are no-ops).
// Coinbase Exchange websocket docs ("full channel": sequence numbers, drop
// messages with sequence <= the REST book's sequence) and Binance spot
// "How to manage a local order book correctly" state the same procedure.
//
// Worked example (tests/test_feed.cpp, seqsync_worked_example): snapshot at
// S=10; updates 11, 13, 12 arrive. 11 applies; 13 is held and the book goes
// stale; 12 applies, then 13 drains from the buffer; the book is fresh again.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>

namespace qp::ull::feed {

struct SyncConfig {
  std::size_t reorder_window = 64;     // held updates before a hole is declared lost
  std::int64_t gap_timeout_ns = 50'000'000;  // 50 ms hole age before declaring it lost
  std::size_t max_buffer = 1'000'000;  // hard cap while waiting for a snapshot
};

struct SyncStats {
  std::uint64_t applied = 0, duplicates = 0, held = 0, gaps = 0, snapshots = 0, discarded_by_snapshot = 0,
                overflow_resets = 0, stale_snapshots = 0;
};

// Msg is the decoded update; the caller supplies how to apply it.
template <class Msg>
class SeqSync {
 public:
  using Apply = std::function<void(const Msg&)>;
  using Notify = std::function<void()>;

  SeqSync(SyncConfig cfg, Apply apply) : cfg_(cfg), apply_(std::move(apply)) {}

  Notify on_snapshot_needed;           // the caller should fetch a REST snapshot
  std::function<void(bool)> on_stale;  // stale flag changed (W26 pulls quotes on true)

  bool stale() const { return !synced_ || !held_.empty(); }
  bool synced() const { return synced_; }
  std::int64_t applied_seq() const { return applied_; }
  const SyncStats& stats() const { return st_; }
  std::size_t held_count() const { return held_.size(); }

  // One update from either line. now_ns drives the gap timeout.
  void on_update(std::int64_t first, std::int64_t last, Msg m, std::int64_t now_ns) {
    bool was = stale();
    if (synced_ && last < applied_ + 1) { ++st_.duplicates; return; }
    auto it = held_.find(first);
    if (it != held_.end() && it->second.last == last) { ++st_.duplicates; return; }
    if (synced_ && first <= applied_ + 1) {
      do_apply(m, last);
      drain();
    } else {
      if (held_.size() >= cfg_.max_buffer) {  // waiting too long: start over
        held_.clear();
        ++st_.overflow_resets;
        request();
      }
      held_.emplace(first, Held{last, now_ns, std::move(m)});
      ++st_.held;
      if (synced_ && !requested_ && held_.size() > cfg_.reorder_window) declare_gap();
    }
    if (!synced_ && !requested_) request();
    // The hole's age runs from the arrival of the oldest update still held:
    // after a partial drain the remaining hole is a newer one.
    hole_since_ = held_.empty() ? -1 : held_.begin()->second.arrived_ns;
    tick(now_ns);
    notify(was);
  }

  // Timer: declares an old hole lost.
  void tick(std::int64_t now_ns) {
    if (synced_ && !held_.empty() && !requested_ && hole_since_ >= 0 && now_ns - hole_since_ >= cfg_.gap_timeout_ns)
      declare_gap();
  }

  // A REST snapshot at sequence s has been loaded into the book by the caller
  // (via `load`). Discards held updates it covers and applies the rest.
  void on_snapshot(std::int64_t s, const std::function<void()>& load) {
    bool was = stale();
    ++st_.snapshots;
    if (synced_ && s <= applied_) {
      // The snapshot is older than the book we already hold (the hole filled
      // itself while the request was in flight). Loading it would silently
      // lose the updates in (s, applied]. Keep our book; ask again only if a
      // hole is still open.
      ++st_.stale_snapshots;
      requested_ = false;
      if (!held_.empty()) request();
      notify(was);
      return;
    }
    load();
    synced_ = true;
    requested_ = false;
    applied_ = s;
    while (!held_.empty() && held_.begin()->second.last <= s) {
      held_.erase(held_.begin());
      ++st_.discarded_by_snapshot;
    }
    drain();
    if (!held_.empty()) request();  // still a hole after the snapshot: ask again
    else hole_since_ = -1;
    notify(was);
  }

  // Line reconnect or explicit reset: forget everything and resynchronise.
  void reset() {
    bool was = stale();
    synced_ = false;
    held_.clear();
    requested_ = false;
    request();
    notify(was);
  }

 private:
  struct Held {
    std::int64_t last;
    std::int64_t arrived_ns;
    Msg m;
  };
  SyncConfig cfg_;
  Apply apply_;
  SyncStats st_;
  bool synced_ = false;
  bool requested_ = false;
  std::int64_t applied_ = 0;
  std::int64_t hole_since_ = -1;
  std::map<std::int64_t, Held> held_;  // by first sequence

  void do_apply(const Msg& m, std::int64_t last) {
    apply_(m);
    applied_ = last;
    ++st_.applied;
  }
  void drain() {
    while (!held_.empty()) {
      auto it = held_.begin();
      if (it->second.last < applied_ + 1) { held_.erase(it); ++st_.duplicates; continue; }
      if (it->first > applied_ + 1) break;
      Held h = std::move(it->second);
      held_.erase(it);
      do_apply(h.m, h.last);
    }
  }
  void declare_gap() {
    ++st_.gaps;
    request();
  }
  void request() {
    requested_ = true;
    if (on_snapshot_needed) on_snapshot_needed();
  }
  void notify(bool was) {
    bool now = stale();
    if (now != was && on_stale) on_stale(now);
  }
};

}  // namespace qp::ull::feed
