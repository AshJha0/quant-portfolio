// Build the QPBF snapshot stream for a set of books (publisher side).
//
// The stream is a series of ordinary QPBF packets whose header seq is the
// as-of sequence S; every message is a Snapshot message carrying S. Orders
// are emitted level by level in FIFO order so the receiver rebuilds the same
// queue positions. The last message is an end marker (order_id 0, LAST
// flag), so an empty book is still a well-formed snapshot (QPBF.md s.6).
#pragma once

#include "../book/l3_book.hpp"
#include "../qpbf/qpbf.hpp"

namespace ull::mcast {

// emit(const uint8_t* bytes, size_t n) is called once per packet.
template <class Emit>
void build_snapshot(const book::BookSet& books, std::uint32_t session, std::uint64_t as_of,
                    std::uint64_t ts_ns, Emit&& emit) {
    qpbf::PacketWriter w;
    w.begin(session, as_of);
    auto push = [&](const qpbf::Msg& m) {
        if (!w.add(m)) {
            const auto n = w.finish();
            emit(w.data(), n);
            w.begin(session, as_of);
            w.add(m);
        }
    };
    for (std::uint32_t i = 0; i < books.size(); ++i) {
        books[i].for_each_live([&](std::uint64_t id, qpbf::Side s, std::int64_t px, std::uint32_t q) {
            qpbf::Msg m;
            m.type = qpbf::MsgType::Snapshot;
            m.instrument = i;
            m.ts_ns = ts_ns;
            m.order_id = id;
            m.side = s;
            m.price = px;
            m.qty = q;
            m.as_of_seq = as_of;
            push(m);
        });
    }
    qpbf::Msg end;
    end.type = qpbf::MsgType::Snapshot;
    end.flags = qpbf::kFlagLastInSnapshot;
    end.ts_ns = ts_ns;
    end.as_of_seq = as_of;
    push(end);
    const auto n = w.finish();
    emit(w.data(), n);
}

}  // namespace ull::mcast
