// Seeded QPBF message generator with a per-type ledger.
//
// The generator keeps its own view of every live order, so each message it
// emits is valid against the book built from the messages before it: a
// cancel or execute names a live order and never removes more than is left,
// a modify keeps the price inside the configured band. The same seed always
// yields the same byte stream (xoshiro256** seeded by splitmix64; Blackman
// and Vigna 2021, "Scrambled linear pseudorandom number generators", ACM
// TOMS 47(4)).
//
// Seam: the plan drives this generator from the W25 market simulator. That
// lane is not merged yet; until then the order flow is a simple mid random
// walk with geometric distance from the mid. Swapping the flow source changes
// only Generator::next_event.
#pragma once

#include "qpbf.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace ull::qpbf {

class Xoshiro256ss {
public:
    explicit Xoshiro256ss(std::uint64_t seed) noexcept {
        for (auto& w : s_) {
            seed += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            w = z ^ (z >> 31);
        }
    }
    std::uint64_t operator()() noexcept {
        const std::uint64_t r = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return r;
    }
    // Uniform in [0, n) by Lemire's multiply-shift (bias < n / 2^64).
    std::uint64_t below(std::uint64_t n) noexcept {
        return static_cast<std::uint64_t>((static_cast<unsigned __int128>((*this)()) * n) >> 64);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
    std::array<std::uint64_t, 4> s_{};
};

struct GeneratorConfig {
    std::uint64_t seed{1};
    std::uint32_t instruments{1};
    std::int64_t base_price{100'000};  // lowest valid tick
    std::uint32_t levels{4096};        // valid ticks: [base, base + levels)
    std::uint32_t max_live_per_instrument{20'000};
    // Relative weights of add, modify, cancel, execute, trade, heartbeat.
    std::array<std::uint32_t, 6> weights{40, 10, 30, 12, 4, 4};
};

struct Ledger {
    std::array<std::uint64_t, kNumMsgTypes> by_type{};
    std::uint64_t total() const noexcept {
        std::uint64_t t = 0;
        for (auto v : by_type) t += v;
        return t;
    }
    friend bool operator==(const Ledger&, const Ledger&) = default;
};

class Generator {
public:
    explicit Generator(const GeneratorConfig& c) : cfg_(c), rng_(c.seed), inst_(c.instruments) {
        for (auto& in : inst_) {
            in.mid = cfg_.base_price + cfg_.levels / 2;
            in.live.reserve(cfg_.max_live_per_instrument);
        }
    }

    // Produce the next message; also counts it in the ledger.
    void next(Msg& m) noexcept {
        next_event(m);
        ++ledger_.by_type[static_cast<std::size_t>(m.type)];
    }

    [[nodiscard]] const Ledger& ledger() const noexcept { return ledger_; }
    [[nodiscard]] const GeneratorConfig& config() const noexcept { return cfg_; }

private:
    struct Live {
        std::uint64_t id;
        std::int64_t price;
        std::uint32_t qty;
        Side side;
    };
    struct Inst {
        std::int64_t mid{0};
        std::vector<Live> live;
    };

    std::int64_t clamp_price(std::int64_t p) const noexcept {
        const std::int64_t lo = cfg_.base_price;
        const std::int64_t hi = cfg_.base_price + cfg_.levels - 1;
        return p < lo ? lo : (p > hi ? hi : p);
    }

    std::int64_t draw_price(const Inst& in, Side s) noexcept {
        // Geometric-ish distance: count trailing zero bits of a random word.
        std::uint64_t r = rng_() | (1ull << 40);
        const auto d = static_cast<std::int64_t>(__builtin_ctzll(r)) +
                       static_cast<std::int64_t>(rng_.below(4));
        return clamp_price(s == Side::Bid ? in.mid - 1 - d : in.mid + d);
    }

    void next_event(Msg& m) noexcept {
        m = Msg{};
        ts_ += 1 + rng_.below(2000);
        m.ts_ns = ts_;
        const auto ii = static_cast<std::uint32_t>(rng_.below(cfg_.instruments));
        m.instrument = ii;
        Inst& in = inst_[ii];

        if (rng_.below(64) == 0) {  // occasional mid move
            in.mid = clamp_price(in.mid + (rng_.below(2) ? 1 : -1));
            if (in.mid <= cfg_.base_price + 64) in.mid += 1;
            if (in.mid >= cfg_.base_price + cfg_.levels - 64) in.mid -= 1;
        }

        std::uint64_t wsum = 0;
        for (auto w : cfg_.weights) wsum += w;
        std::uint64_t pick = rng_.below(wsum);
        int kind = 0;
        while (pick >= cfg_.weights[kind]) pick -= cfg_.weights[kind++];

        // Needs a live order: fall back to add when the book is empty.
        if ((kind == 1 || kind == 2 || kind == 3) && in.live.empty()) kind = 0;
        if (kind == 0 && in.live.size() >= cfg_.max_live_per_instrument) kind = 2;

        switch (kind) {
            case 0: {
                const Side s = rng_.below(2) ? Side::Bid : Side::Ask;
                Live o{++next_id_, draw_price(in, s),
                       static_cast<std::uint32_t>(1 + rng_.below(1000)), s};
                in.live.push_back(o);
                m.type = MsgType::Add;
                m.order_id = o.id;
                m.price = o.price;
                m.qty = o.qty;
                m.side = s;
                break;
            }
            case 1: {
                Live& o = in.live[rng_.below(in.live.size())];
                if (rng_.below(2) && o.qty > 1) {
                    o.qty = static_cast<std::uint32_t>(1 + rng_.below(o.qty - 1));
                } else {
                    o.price = draw_price(in, o.side);
                    o.qty = static_cast<std::uint32_t>(1 + rng_.below(1000));
                }
                m.type = MsgType::Modify;
                m.order_id = o.id;
                m.price = o.price;
                m.qty = o.qty;
                break;
            }
            case 2:
            case 3: {
                const std::size_t k = rng_.below(in.live.size());
                Live& o = in.live[k];
                const bool full = rng_.below(3) != 0 || o.qty == 1;
                const auto q = full ? o.qty : static_cast<std::uint32_t>(1 + rng_.below(o.qty - 1));
                m.type = kind == 2 ? MsgType::Cancel : MsgType::Execute;
                m.order_id = o.id;
                m.qty = q;
                if (kind == 3) m.match_id = ++next_match_;
                if (q == o.qty) {
                    o = in.live.back();
                    in.live.pop_back();
                } else {
                    o.qty -= q;
                }
                break;
            }
            case 4:
                m.type = MsgType::Trade;
                m.match_id = ++next_match_;
                m.side = rng_.below(2) ? Side::Bid : Side::Ask;
                m.price = in.mid;
                m.qty = static_cast<std::uint32_t>(1 + rng_.below(500));
                break;
            default:
                m.type = MsgType::Heartbeat;
                m.instrument = 0;
                break;
        }
    }

    GeneratorConfig cfg_;
    Xoshiro256ss rng_;
    std::vector<Inst> inst_;
    Ledger ledger_{};
    std::uint64_t ts_{1'700'000'000'000'000'000ull};
    std::uint64_t next_id_{0};
    std::uint64_t next_match_{0};
};

}  // namespace ull::qpbf
