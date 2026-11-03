// QPBF feed handler: A/B line arbitration, sequence-gap detection, a reorder
// window, TCP snapshot recovery with buffering of live deltas, and a stale
// flag. Transport-free: callers hand it packet bytes and a clock reading, so
// every drop/duplicate/reorder/burst pattern is testable deterministically
// (tests/test_recovery.cpp); net.hpp supplies the sockets.
//
// Plain idea. The exchange sends the same packets on two multicast lines, A
// and B. Every message has a sequence number. We take each sequence number
// from whichever line delivers it first and ignore the second copy. If a
// number is missing we hold later messages in a window and wait a short time
// for the other line to fill the hole. If it does not, or the hole is wider
// than the window, the book can no longer be trusted: we mark it stale (a
// market maker pulls its quotes), ask the snapshot server for the full book
// as of some sequence S, keep buffering live messages meanwhile, rebuild the
// book from the snapshot, then replay buffered messages after S. When the
// sequence is contiguous again the stale flag clears.
//
// Formal statement (the invariant the tests check): whenever stale() is
// false, the sink has received exactly the messages with sequence numbers
// 1..next_seq()-1 of the current session, each once and in order, or a
// snapshot as of S followed by S+1..next_seq()-1. stale() is true from the
// moment a gap is observed until that condition holds again.
//
// Worked example: window 8, packets carry seq 1-3, 4-6, 7-9. Line A drops
// 4-6. A's 7-9 arrives: gap (next = 4), stale, 7-9 buffered. B's 4-6 arrives:
// delivered, then 7-9 drained from the window; next = 10, stale clears. No
// snapshot was needed. If B had also dropped 4-6, the gap timeout fires,
// request_snapshot() is called, and a snapshot as of 12 (say) is applied,
// buffered 7-9 are discarded as older than 12, 13.. continue.
//
// References: CME MDP 3.0 "Market Data Recovery" and the MoldUDP64 framing both
// describe A/B feeds with sequence-gap recovery; this design follows the
// common pattern, not either venue's exact protocol (QPBF.md section 7).
#pragma once

#include "../qpbf/qpbf.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace ull::mcast {

using qpbf::Msg;
using qpbf::MsgType;
using qpbf::PacketHeader;
using qpbf::Status;

struct FeedConfig {
    std::uint32_t window{1u << 16};        // reorder/buffer capacity, power of 2
    std::uint64_t gap_timeout_ns{200'000};  // wait for the other line this long
};

struct FeedStats {
    std::uint64_t packets{0}, bad_packets{0}, delivered{0}, duplicates{0}, gaps{0},
        snapshots_requested{0}, snapshots_applied{0}, overflow_evictions{0},
        stale_buffered_dropped{0}, session_resets{0};
};

// Sink contract:
//   void on_msg(std::uint64_t seq, const Msg& m)  in-order delivery; seq 0 for
//                                                 snapshot orders
//   void on_reset()                               book must be cleared
//   void request_snapshot()                       ask the snapshot server
//   void on_stale(bool stale)                     edge-triggered flag change
template <class Sink>
class FeedHandler {
public:
    FeedHandler(Sink& sink, FeedConfig cfg)
        : sink_(sink), cfg_(cfg),
          slots_(std::make_unique<Msg[]>(cfg.window)),
          seqs_(std::make_unique<std::uint64_t[]>(cfg.window)) {
        if (cfg.window < 2 || (cfg.window & (cfg.window - 1)) != 0)
            throw std::invalid_argument("FeedHandler window must be a power of two");
        mask_ = cfg.window - 1;
        for (std::uint32_t i = 0; i < cfg.window; ++i) seqs_[i] = 0;
    }

    // A packet from line 0 (A) or 1 (B). `now_ns` is any monotone clock.
    void on_packet(int /*line*/, const std::uint8_t* p, std::size_t n, std::uint64_t now_ns) noexcept {
        now_ = now_ns;
        PacketHeader h;
        // decode_packet validates the whole packet before delivering any
        // message, so a malformed packet changes no state at all.
        std::uint32_t k = 0;
        const Status s = qpbf::decode_packet(p, n, h, [&](std::uint64_t, const Msg& m) {
            pkt_[k++] = m;
        });
        if (s != Status::Ok) { ++st_.bad_packets; return; }
        if (!have_session_ || h.session_id != session_) new_session(h.session_id, h.seq);
        for (std::uint32_t i = 0; i < k; ++i) on_seq(h.seq + i, pkt_[i]);
        ++st_.packets;
        drain();
        check_timeout();
    }

    // Call periodically when no packets arrive, so a gap at the tail of the
    // stream still times out.
    void tick(std::uint64_t now_ns) noexcept {
        now_ = now_ns;
        check_timeout();
    }

    // One packet of the TCP snapshot stream (header.seq = as-of sequence).
    // Returns false on a malformed packet (the snapshot is then abandoned and
    // re-requested on the next timeout).
    bool on_snapshot_packet(const std::uint8_t* p, std::size_t n) noexcept {
        PacketHeader h;
        bool last = false, ok = true;
        if (!awaiting_snapshot_) return false;
        const Status s = qpbf::decode_packet(p, n, h, [&](std::uint64_t, const Msg& m) {
            if (m.type != MsgType::Snapshot) { ok = false; return; }
            if (!in_snapshot_) { sink_.on_reset(); in_snapshot_ = true; snap_as_of_ = m.as_of_seq; }
            if (m.as_of_seq != snap_as_of_) { ok = false; return; }
            if (m.order_id != 0) sink_.on_msg(0, m);
            if (m.flags & qpbf::kFlagLastInSnapshot) last = true;
        });
        if (s != Status::Ok || !ok || h.session_id != session_) {
            in_snapshot_ = false;
            awaiting_snapshot_ = false;  // re-request after the next timeout
            gap_since_ = now_;
            have_gap_ = true;
            return false;
        }
        if (last) finish_snapshot();
        return true;
    }

    // The sink found the book inconsistent (QPBF.md s.5 rule 5: unknown
    // order, duplicate id, over-fill). Flag stale and recover by snapshot.
    void invalidate() noexcept {
        set_stale(true);
        if (!awaiting_snapshot_) request();
    }

    [[nodiscard]] bool stale() const noexcept { return stale_; }
    [[nodiscard]] std::uint64_t next_seq() const noexcept { return next_; }
    [[nodiscard]] bool awaiting_snapshot() const noexcept { return awaiting_snapshot_; }
    [[nodiscard]] const FeedStats& stats() const noexcept { return st_; }

private:
    void set_stale(bool v) noexcept {
        if (v != stale_) { stale_ = v; sink_.on_stale(v); }
    }

    void new_session(std::uint32_t sid, std::uint64_t first_seq) noexcept {
        if (have_session_) ++st_.session_resets;
        have_session_ = true;
        session_ = sid;
        for (std::uint32_t i = 0; i < cfg_.window; ++i) seqs_[i] = 0;
        buffered_ = 0;
        in_snapshot_ = false;
        awaiting_snapshot_ = false;
        if (first_seq == 1) {
            // Session starts from an empty book: no snapshot needed.
            sink_.on_reset();
            next_ = 1;
            have_gap_ = false;
            set_stale(false);
        } else {
            // Joined mid-session: buffer from here and fetch a snapshot.
            sink_.on_reset();
            next_ = first_seq;
            base_ = first_seq;
            set_stale(true);
            request();
        }
    }

    void request() noexcept {
        awaiting_snapshot_ = true;
        in_snapshot_ = false;
        have_gap_ = true;
        ++st_.snapshots_requested;
        sink_.request_snapshot();
    }

    // Window holds sequence numbers in [base_, base_ + window). While live,
    // base_ == next_. While awaiting a snapshot, base_ slides forward so the
    // most recent `window` messages are kept.
    void on_seq(std::uint64_t seq, const Msg& m) noexcept {
        if (!awaiting_snapshot_ && seq == next_) {
            sink_.on_msg(seq, m);
            ++st_.delivered;
            ++next_;
            base_ = next_;
            return;
        }
        if (seq < next_ || (awaiting_snapshot_ && seq < base_)) { ++st_.duplicates; return; }
        if (!awaiting_snapshot_ && !have_gap_) {
            have_gap_ = true;
            gap_since_ = now_;
            ++st_.gaps;
            set_stale(true);
        }
        if (seq >= base_ + cfg_.window) {
            if (!awaiting_snapshot_) {
                request();  // hole wider than the window: only a snapshot helps
            }
            // Slide: evict everything older than seq - window + 1.
            const std::uint64_t nb = seq - cfg_.window + 1;
            for (std::uint64_t s = base_; s < nb && buffered_ > 0; ++s) {
                auto& slot = seqs_[s & mask_];
                if (slot == s) { slot = 0; --buffered_; ++st_.overflow_evictions; }
                if (s - base_ > cfg_.window) break;
            }
            base_ = nb;
        }
        auto& slot = seqs_[seq & mask_];
        if (slot == seq) { ++st_.duplicates; return; }
        if (slot != 0) --buffered_;  // stale occupant from an older lap
        slot = seq;
        slots_[seq & mask_] = m;
        ++buffered_;
    }

    void drain() noexcept {
        if (awaiting_snapshot_) return;
        const std::uint64_t start = next_;
        while (buffered_ > 0) {
            auto& slot = seqs_[next_ & mask_];
            if (slot != next_) break;
            sink_.on_msg(next_, slots_[next_ & mask_]);
            slot = 0;
            --buffered_;
            ++st_.delivered;
            ++next_;
        }
        base_ = next_;
        if (buffered_ == 0) {
            if (have_gap_) { have_gap_ = false; set_stale(false); }
        } else if (!have_gap_ || next_ != start) {
            // A hole remains. Its age counts from the last progress, so a
            // run of small holes each filled in time never trips the timeout.
            have_gap_ = true;
            gap_since_ = now_;
        }
    }

    void check_timeout() noexcept {
        if (have_gap_ && !awaiting_snapshot_ && now_ - gap_since_ >= cfg_.gap_timeout_ns) request();
    }

    void finish_snapshot() noexcept {
        ++st_.snapshots_applied;
        in_snapshot_ = false;
        awaiting_snapshot_ = false;
        next_ = snap_as_of_ + 1;
        // Drop buffered messages the snapshot already contains.
        if (buffered_ > 0) {
            for (std::uint32_t i = 0; i < cfg_.window; ++i) {
                if (seqs_[i] != 0 && seqs_[i] < next_) { seqs_[i] = 0; --buffered_; ++st_.stale_buffered_dropped; }
            }
        }
        base_ = next_;
        have_gap_ = true;
        gap_since_ = now_;
        drain();
    }

    Sink& sink_;
    FeedConfig cfg_;
    std::unique_ptr<Msg[]> slots_;
    std::unique_ptr<std::uint64_t[]> seqs_;  // 0 = empty (sequence numbers start at 1)
    std::uint64_t mask_{0};
    std::uint64_t next_{1};
    std::uint64_t base_{1};
    std::uint32_t buffered_{0};
    std::uint32_t session_{0};
    bool have_session_{false};
    bool have_gap_{false};
    bool stale_{true};
    bool awaiting_snapshot_{false};
    bool in_snapshot_{false};
    std::uint64_t snap_as_of_{0};
    std::uint64_t gap_since_{0};
    std::uint64_t now_{0};
    FeedStats st_;
    Msg pkt_[qpbf::kMaxMessagesPerPacket];
};

}  // namespace ull::mcast
