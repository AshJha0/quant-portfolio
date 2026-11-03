//! QPBF, the quant-portfolio binary feed: decoder, encoder and a seeded
//! generator.
//!
//! Plain idea: a market-data feed is a stream of UDP-sized packets. Each
//! packet says which session sent it and the sequence number of its first
//! message. Every message has a fixed size per type, so a decoder can walk a
//! packet with offset arithmetic only and never copy or allocate.
//!
//! The normative specification is `docs/QPBF.md` (version 1). This crate is
//! an independent Rust implementation of it: it was first written from the
//! plan's W30 description, then converged on the shared spec once that was
//! final, without reading the C++ code. The golden bytes of spec section 10
//! are checked by the test `golden_bytes_spec_section_10`.
//!
//! # Formal layout (spec sections 3-4; little-endian; padding must be zero)
//!
//! Packet header, 16 bytes: `0 magic u16 = 0x5051`, `2 version u8 = 1`,
//! `3 msg_count u8`, `4 session_id u32`, `8 seq u64` (sequence of the first
//! message). A packet is at most 1,472 bytes and ends exactly after its last
//! message.
//!
//! Message prefix, 16 bytes: `0 length u16`, `2 type u8`, `3 flags u8`,
//! `4 instrument u32`, `8 ts_ns u64`.
//!
//! | type | value | size | body |
//! |---|---|---|---|
//! | Add | 1 | 40 | 16 order_id, 24 price i64, 32 qty u32, 36 side u8, 37 pad 3 |
//! | Modify | 2 | 40 | 16 order_id, 24 price i64, 32 qty u32, 36 pad 4 |
//! | Cancel | 3 | 32 | 16 order_id, 24 qty u32, 28 pad 4 |
//! | Execute | 4 | 40 | 16 order_id, 24 qty u32, 28 pad 4, 32 match_id u64 |
//! | Trade | 5 | 40 | 16 match_id, 24 price i64, 32 qty u32, 36 side u8, 37 pad 3 |
//! | Snapshot | 6 | 48 | as Add, plus 40 as_of_seq u64; flags bit 0 = LAST |
//! | Heartbeat | 7 | 16 | none; instrument 0 |
//!
//! Side is 1 = bid, 2 = ask. Order ids and quantities are non-zero, except
//! the Snapshot end marker (order 0 with price, qty and side all zero).
//!
//! Worked example (spec section 10): an Add for instrument 7, ts
//! `0x0102030405060708`, order 42, price -3, qty 100, ask, in a packet of
//! session `0xAABBCCDD` at sequence 1000 is the 56 bytes
//! `51 50 01 01 DD CC BB AA E8 03 00 00 00 00 00 00` then
//! `28 00 01 00 07 00 00 00 08 07 06 05 04 03 02 01`
//! `2A 00 00 00 00 00 00 00 FD FF FF FF FF FF FF FF 64 00 00 00 02 00 00 00`.

#![warn(missing_docs)]

pub mod generator;

/// Magic, `51 50` on the wire.
pub const MAGIC: u16 = 0x5051;
/// Wire version.
pub const VERSION: u8 = 1;
/// Packet header size in bytes.
pub const PACKET_HEADER_LEN: usize = 16;
/// Common message prefix size in bytes.
pub const MSG_HEADER_LEN: usize = 16;
/// Largest message size in bytes.
pub const MAX_MSG_LEN: usize = 48;
/// Largest packet: 1500-byte MTU minus IPv4 (20) and UDP (8) headers.
pub const MAX_PACKET_LEN: usize = 1472;
/// Most messages in one packet.
pub const MAX_MSGS_PER_PACKET: usize = 255;
/// Snapshot flag bit 0: last message of a snapshot.
pub const SNAP_LAST: u8 = 1;
/// Recording file magic (spec section 9).
pub const RECORDING_MAGIC: &[u8; 8] = b"QPBFREC1";

/// Book side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// Bid (wire 1).
    Buy,
    /// Ask (wire 2).
    Sell,
}

impl Side {
    fn to_byte(self) -> u8 {
        match self {
            Side::Buy => 1,
            Side::Sell => 2,
        }
    }
    fn from_byte(b: u8) -> Result<Self, DecodeError> {
        match b {
            1 => Ok(Side::Buy),
            2 => Ok(Side::Sell),
            _ => Err(DecodeError::BadSide(b)),
        }
    }
}

/// Message kinds, also the index into a per-type ledger (`type - 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MsgType {
    /// 1
    Add = 0,
    /// 2
    Modify = 1,
    /// 3
    Cancel = 2,
    /// 4
    Execute = 3,
    /// 5
    Trade = 4,
    /// 6
    Snapshot = 5,
    /// 7
    Heartbeat = 6,
}

/// Number of message types.
pub const N_TYPES: usize = 7;

impl MsgType {
    /// All types, in ledger order.
    pub const ALL: [MsgType; N_TYPES] = [
        MsgType::Add,
        MsgType::Modify,
        MsgType::Cancel,
        MsgType::Execute,
        MsgType::Trade,
        MsgType::Snapshot,
        MsgType::Heartbeat,
    ];
    /// Type value on the wire.
    pub const fn wire(self) -> u8 {
        self as u8 + 1
    }
    /// Fixed wire size.
    pub const fn size(self) -> usize {
        [40, 40, 32, 40, 40, 48, 16][self as usize]
    }
    fn from_wire(t: u8) -> Option<Self> {
        if (1..=7).contains(&t) {
            Some(Self::ALL[(t - 1) as usize])
        } else {
            None
        }
    }
}

/// One decoded message. `Copy`, fixed size, no heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Msg {
    /// New visible order, at the back of its level.
    Add {
        /// Instrument id.
        instrument: u32,
        /// Timestamp, ns since the Unix epoch.
        ts_ns: u64,
        /// Order id (non-zero).
        order_id: u64,
        /// Side.
        side: Side,
        /// Price in ticks.
        price: i64,
        /// Quantity (non-zero).
        qty: u32,
    },
    /// New price and remaining quantity of a resting order.
    Modify {
        /// Instrument id.
        instrument: u32,
        /// Timestamp, ns.
        ts_ns: u64,
        /// Order id.
        order_id: u64,
        /// New price in ticks.
        price: i64,
        /// New remaining quantity.
        qty: u32,
    },
    /// Owner removed `qty` from an order.
    Cancel {
        /// Instrument id.
        instrument: u32,
        /// Timestamp, ns.
        ts_ns: u64,
        /// Order id.
        order_id: u64,
        /// Quantity removed.
        qty: u32,
    },
    /// `qty` of a resting order filled at its price.
    Execute {
        /// Instrument id.
        instrument: u32,
        /// Timestamp, ns.
        ts_ns: u64,
        /// Order id.
        order_id: u64,
        /// Filled quantity.
        qty: u32,
        /// Match id.
        match_id: u64,
    },
    /// Trade against non-displayed liquidity; no book change.
    Trade {
        /// Instrument id.
        instrument: u32,
        /// Timestamp, ns.
        ts_ns: u64,
        /// Aggressor side.
        side: Side,
        /// Match id.
        match_id: u64,
        /// Price in ticks.
        price: i64,
        /// Quantity.
        qty: u32,
    },
    /// One order of a recovery snapshot (TCP channel only).
    Snapshot {
        /// Instrument id.
        instrument: u32,
        /// Timestamp, ns.
        ts_ns: u64,
        /// `SNAP_LAST` or 0.
        flags: u8,
        /// Order id; 0 only on the LAST end marker.
        order_id: u64,
        /// Side (ignored on the end marker, encoded as 0).
        side: Side,
        /// Price in ticks.
        price: i64,
        /// Quantity.
        qty: u32,
        /// Sequence the snapshot reflects.
        as_of_seq: u64,
    },
    /// Liveness only.
    Heartbeat {
        /// Instrument id (0 by spec).
        instrument: u32,
        /// Timestamp, ns.
        ts_ns: u64,
    },
}

impl Msg {
    /// Message type.
    pub fn kind(&self) -> MsgType {
        match self {
            Msg::Add { .. } => MsgType::Add,
            Msg::Modify { .. } => MsgType::Modify,
            Msg::Cancel { .. } => MsgType::Cancel,
            Msg::Execute { .. } => MsgType::Execute,
            Msg::Trade { .. } => MsgType::Trade,
            Msg::Snapshot { .. } => MsgType::Snapshot,
            Msg::Heartbeat { .. } => MsgType::Heartbeat,
        }
    }

    /// Instrument id.
    pub fn instrument(&self) -> u32 {
        match *self {
            Msg::Add { instrument, .. }
            | Msg::Modify { instrument, .. }
            | Msg::Cancel { instrument, .. }
            | Msg::Execute { instrument, .. }
            | Msg::Trade { instrument, .. }
            | Msg::Snapshot { instrument, .. }
            | Msg::Heartbeat { instrument, .. } => instrument,
        }
    }

    /// The Snapshot end marker for `instrument` as of `as_of_seq`.
    pub fn snapshot_end(instrument: u32, ts_ns: u64, as_of_seq: u64) -> Msg {
        Msg::Snapshot {
            instrument,
            ts_ns,
            flags: SNAP_LAST,
            order_id: 0,
            side: Side::Buy,
            price: 0,
            qty: 0,
            as_of_seq,
        }
    }

    /// Writes the message at the start of `out` and returns its size.
    /// Panics if `out` is shorter than the message (callers size buffers).
    pub fn encode(&self, out: &mut [u8]) -> usize {
        let k = self.kind();
        let n = k.size();
        let b = &mut out[..n];
        b.fill(0);
        b[0..2].copy_from_slice(&(n as u16).to_le_bytes());
        b[2] = k.wire();
        let (instrument, ts) = match *self {
            Msg::Add {
                instrument,
                ts_ns,
                order_id,
                side,
                price,
                qty,
            } => {
                put_u64(b, 16, order_id);
                put_u64(b, 24, price as u64);
                put_u32(b, 32, qty);
                b[36] = side.to_byte();
                (instrument, ts_ns)
            }
            Msg::Modify {
                instrument,
                ts_ns,
                order_id,
                price,
                qty,
            } => {
                put_u64(b, 16, order_id);
                put_u64(b, 24, price as u64);
                put_u32(b, 32, qty);
                (instrument, ts_ns)
            }
            Msg::Cancel {
                instrument,
                ts_ns,
                order_id,
                qty,
            } => {
                put_u64(b, 16, order_id);
                put_u32(b, 24, qty);
                (instrument, ts_ns)
            }
            Msg::Execute {
                instrument,
                ts_ns,
                order_id,
                qty,
                match_id,
            } => {
                put_u64(b, 16, order_id);
                put_u32(b, 24, qty);
                put_u64(b, 32, match_id);
                (instrument, ts_ns)
            }
            Msg::Trade {
                instrument,
                ts_ns,
                side,
                match_id,
                price,
                qty,
            } => {
                put_u64(b, 16, match_id);
                put_u64(b, 24, price as u64);
                put_u32(b, 32, qty);
                b[36] = side.to_byte();
                (instrument, ts_ns)
            }
            Msg::Snapshot {
                instrument,
                ts_ns,
                flags,
                order_id,
                side,
                price,
                qty,
                as_of_seq,
            } => {
                b[3] = flags;
                put_u64(b, 16, order_id);
                put_u64(b, 24, price as u64);
                put_u32(b, 32, qty);
                b[36] = if order_id == 0 { 0 } else { side.to_byte() };
                put_u64(b, 40, as_of_seq);
                (instrument, ts_ns)
            }
            Msg::Heartbeat { instrument, ts_ns } => (instrument, ts_ns),
        };
        put_u32(b, 4, instrument);
        put_u64(b, 8, ts);
        n
    }
}

/// Why a packet or message was rejected. The decoder returns these for any
/// malformed input; it never panics on untrusted bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Fewer bytes than a header or a declared message.
    Truncated,
    /// Packet longer than 1,472 bytes.
    TooLong,
    /// Wrong magic.
    BadMagic,
    /// Unsupported version.
    BadVersion(u8),
    /// Unknown message type.
    UnknownType(u8),
    /// Message length field does not match the type.
    BadMsgLen,
    /// Side not 1 or 2.
    BadSide(u8),
    /// Non-zero padding, or flags set where not allowed.
    NonZeroReserved,
    /// Zero order id or zero quantity where non-zero is required, or a
    /// malformed snapshot end marker.
    ZeroField,
    /// Bytes left after `msg_count` messages.
    TrailingBytes,
}

/// Packet header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PacketHeader {
    /// Number of messages.
    pub msg_count: u8,
    /// Sender session.
    pub session_id: u32,
    /// Sequence number of the first message.
    pub seq: u64,
}

impl PacketHeader {
    /// Writes the header into `out[..16]`.
    pub fn encode(&self, out: &mut [u8]) {
        let b = &mut out[..PACKET_HEADER_LEN];
        b[0..2].copy_from_slice(&MAGIC.to_le_bytes());
        b[2] = VERSION;
        b[3] = self.msg_count;
        put_u32(b, 4, self.session_id);
        put_u64(b, 8, self.seq);
    }
}

/// A validated, borrowed packet. Construction checks the header and every
/// message (spec: validate the whole packet before acting on any of it), so
/// iteration afterwards cannot fail.
#[derive(Clone, Copy, Debug)]
pub struct Packet<'a> {
    /// Header fields.
    pub header: PacketHeader,
    payload: &'a [u8],
}

impl<'a> Packet<'a> {
    /// Validates `buf` as exactly one packet.
    pub fn parse(buf: &'a [u8]) -> Result<Self, DecodeError> {
        if buf.len() < PACKET_HEADER_LEN {
            return Err(DecodeError::Truncated);
        }
        if buf.len() > MAX_PACKET_LEN {
            return Err(DecodeError::TooLong);
        }
        if u16::from_le_bytes([buf[0], buf[1]]) != MAGIC {
            return Err(DecodeError::BadMagic);
        }
        if buf[2] != VERSION {
            return Err(DecodeError::BadVersion(buf[2]));
        }
        let header = PacketHeader {
            msg_count: buf[3],
            session_id: get_u32(buf, 4),
            seq: get_u64(buf, 8),
        };
        let payload = &buf[PACKET_HEADER_LEN..];
        let mut off = 0;
        for _ in 0..header.msg_count {
            let (_, len) = decode_msg(&payload[off..])?;
            off += len;
        }
        if off != payload.len() {
            return Err(DecodeError::TrailingBytes);
        }
        Ok(Packet { header, payload })
    }

    /// Iterates `(sequence number, message)`. Snapshot messages carry the
    /// header `seq` (they consume no sequence numbers; spec section 6), so
    /// for them the yielded number is the as-of sequence.
    pub fn messages(&self) -> Messages<'a> {
        Messages {
            rest: self.payload,
            seq: self.header.seq,
        }
    }
}

/// Iterator over a validated packet.
#[derive(Clone, Debug)]
pub struct Messages<'a> {
    rest: &'a [u8],
    seq: u64,
}

impl Iterator for Messages<'_> {
    type Item = (u64, Msg);
    #[inline]
    fn next(&mut self) -> Option<(u64, Msg)> {
        if self.rest.is_empty() {
            return None;
        }
        // Validated in `Packet::parse`, so this cannot fail.
        let (m, len) = decode_msg(self.rest).ok()?;
        self.rest = &self.rest[len..];
        let s = self.seq;
        if m.kind() != MsgType::Snapshot {
            self.seq = self.seq.wrapping_add(1);
        }
        Some((s, m))
    }
}

#[inline(always)]
fn nz64(v: u64) -> Result<u64, DecodeError> {
    if v == 0 {
        Err(DecodeError::ZeroField)
    } else {
        Ok(v)
    }
}
#[inline(always)]
fn nz32(v: u32) -> Result<u32, DecodeError> {
    if v == 0 {
        Err(DecodeError::ZeroField)
    } else {
        Ok(v)
    }
}

/// Decodes one message at the start of `b`; returns it and its length.
#[inline]
pub fn decode_msg(b: &[u8]) -> Result<(Msg, usize), DecodeError> {
    if b.len() < MSG_HEADER_LEN {
        return Err(DecodeError::Truncated);
    }
    let k = MsgType::from_wire(b[2]).ok_or(DecodeError::UnknownType(b[2]))?;
    let n = k.size();
    if u16::from_le_bytes([b[0], b[1]]) as usize != n {
        return Err(DecodeError::BadMsgLen);
    }
    if b.len() < n {
        return Err(DecodeError::Truncated);
    }
    let b = &b[..n];
    let flags = b[3];
    let flags_ok = if k == MsgType::Snapshot {
        flags & !SNAP_LAST == 0
    } else {
        flags == 0
    };
    if !flags_ok {
        return Err(DecodeError::NonZeroReserved);
    }
    let pad = |from: usize, to: usize| -> Result<(), DecodeError> {
        if b[from..to].iter().all(|&x| x == 0) {
            Ok(())
        } else {
            Err(DecodeError::NonZeroReserved)
        }
    };
    let instrument = get_u32(b, 4);
    let ts_ns = get_u64(b, 8);
    let m = match k {
        MsgType::Add => {
            pad(37, 40)?;
            Msg::Add {
                instrument,
                ts_ns,
                order_id: nz64(get_u64(b, 16))?,
                price: get_u64(b, 24) as i64,
                qty: nz32(get_u32(b, 32))?,
                side: Side::from_byte(b[36])?,
            }
        }
        MsgType::Modify => {
            pad(36, 40)?;
            Msg::Modify {
                instrument,
                ts_ns,
                order_id: nz64(get_u64(b, 16))?,
                price: get_u64(b, 24) as i64,
                qty: nz32(get_u32(b, 32))?,
            }
        }
        MsgType::Cancel => {
            pad(28, 32)?;
            Msg::Cancel {
                instrument,
                ts_ns,
                order_id: nz64(get_u64(b, 16))?,
                qty: nz32(get_u32(b, 24))?,
            }
        }
        MsgType::Execute => {
            pad(28, 32)?;
            Msg::Execute {
                instrument,
                ts_ns,
                order_id: nz64(get_u64(b, 16))?,
                qty: nz32(get_u32(b, 24))?,
                match_id: get_u64(b, 32),
            }
        }
        MsgType::Trade => {
            pad(37, 40)?;
            Msg::Trade {
                instrument,
                ts_ns,
                match_id: get_u64(b, 16),
                price: get_u64(b, 24) as i64,
                qty: nz32(get_u32(b, 32))?,
                side: Side::from_byte(b[36])?,
            }
        }
        MsgType::Snapshot => {
            pad(37, 40)?;
            let order_id = get_u64(b, 16);
            let price = get_u64(b, 24) as i64;
            let qty = get_u32(b, 32);
            let as_of_seq = get_u64(b, 40);
            if order_id == 0 {
                // End marker: LAST set, price/qty/side zero (spec section 6).
                if flags & SNAP_LAST == 0 || price != 0 || qty != 0 || b[36] != 0 {
                    return Err(DecodeError::ZeroField);
                }
                Msg::snapshot_end(instrument, ts_ns, as_of_seq)
            } else {
                Msg::Snapshot {
                    instrument,
                    ts_ns,
                    flags,
                    order_id,
                    side: Side::from_byte(b[36])?,
                    price,
                    qty: nz32(qty)?,
                    as_of_seq,
                }
            }
        }
        MsgType::Heartbeat => Msg::Heartbeat { instrument, ts_ns },
    };
    Ok((m, n))
}

/// Builds packets up to `MAX_PACKET_LEN` bytes and 255 messages.
#[derive(Debug)]
pub struct PacketWriter {
    buf: Vec<u8>,
    session_id: u32,
    seq: u64,
    count: u8,
    /// Messages in the open packet that consume a sequence number
    /// (everything but Snapshot).
    consumed: u64,
}

impl PacketWriter {
    /// A writer whose first packet starts at sequence `seq`.
    pub fn new(session_id: u32, seq: u64) -> Self {
        let mut buf = Vec::with_capacity(MAX_PACKET_LEN);
        buf.resize(PACKET_HEADER_LEN, 0);
        PacketWriter {
            buf,
            session_id,
            seq,
            count: 0,
            consumed: 0,
        }
    }

    /// True when `m` would not fit and the packet must be flushed first.
    pub fn would_overflow(&self, m: &Msg) -> bool {
        self.buf.len() + m.kind().size() > MAX_PACKET_LEN
            || self.count as usize == MAX_MSGS_PER_PACKET
    }

    /// Appends a message. Call `would_overflow` first.
    pub fn push(&mut self, m: &Msg) {
        assert!(!self.would_overflow(m), "packet full");
        let at = self.buf.len();
        self.buf.resize(at + m.kind().size(), 0);
        m.encode(&mut self.buf[at..]);
        self.count += 1;
        if m.kind() != MsgType::Snapshot {
            self.consumed += 1;
        }
    }

    /// Messages in the open packet.
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// True when the open packet has no messages.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Finishes the open packet, hands its bytes to `sink`, and starts the
    /// next packet at the following sequence number.
    pub fn flush(&mut self, sink: &mut impl FnMut(&[u8])) {
        PacketHeader {
            msg_count: self.count,
            session_id: self.session_id,
            seq: self.seq,
        }
        .encode(&mut self.buf);
        sink(&self.buf);
        self.seq += self.consumed;
        self.consumed = 0;
        self.count = 0;
        self.buf.truncate(PACKET_HEADER_LEN);
    }

    /// Next sequence number to be assigned.
    pub fn next_seq(&self) -> u64 {
        self.seq + self.consumed
    }
}

/// Appends `pkt` to a recording buffer (spec section 9). Call
/// `recording_header` once first.
pub fn record_packet(out: &mut Vec<u8>, pkt: &[u8]) {
    assert!(pkt.len() <= MAX_PACKET_LEN);
    out.extend_from_slice(&(pkt.len() as u16).to_le_bytes());
    out.extend_from_slice(pkt);
}

/// Starts a recording buffer with the magic.
pub fn recording_header(out: &mut Vec<u8>) {
    out.extend_from_slice(RECORDING_MAGIC);
}

/// Splits a recording into packet slices. Rejects a bad magic, a record
/// over 1,472 bytes, or a truncated record.
pub fn split_recording(bytes: &[u8]) -> Result<Vec<&[u8]>, DecodeError> {
    if bytes.len() < 8 || &bytes[..8] != RECORDING_MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let mut out = Vec::new();
    let mut off = 8;
    while off < bytes.len() {
        if off + 2 > bytes.len() {
            return Err(DecodeError::Truncated);
        }
        let len = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as usize;
        if len > MAX_PACKET_LEN {
            return Err(DecodeError::TooLong);
        }
        let end = off + 2 + len;
        if end > bytes.len() {
            return Err(DecodeError::Truncated);
        }
        out.push(&bytes[off + 2..end]);
        off = end;
    }
    Ok(out)
}

#[inline(always)]
fn get_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline(always)]
fn get_u64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}
fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: String = s.split_whitespace().collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    const GOLDEN: &str = "51 50 01 01 DD CC BB AA E8 03 00 00 00 00 00 00
        28 00 01 00 07 00 00 00 08 07 06 05 04 03 02 01
        2A 00 00 00 00 00 00 00 FD FF FF FF FF FF FF FF
        64 00 00 00 02 00 00 00";

    /// Spec section 10, both directions.
    #[test]
    fn golden_bytes_spec_section_10() {
        let want = hex(GOLDEN);
        assert_eq!(want.len(), 56);
        let m = Msg::Add {
            instrument: 7,
            ts_ns: 0x0102_0304_0506_0708,
            order_id: 42,
            side: Side::Sell,
            price: -3,
            qty: 100,
        };
        let mut w = PacketWriter::new(0xAABB_CCDD, 1000);
        w.push(&m);
        let mut got = Vec::new();
        w.flush(&mut |b| got = b.to_vec());
        assert_eq!(got, want);
        let p = Packet::parse(&want).unwrap();
        assert_eq!(
            p.header,
            PacketHeader {
                msg_count: 1,
                session_id: 0xAABB_CCDD,
                seq: 1000
            }
        );
        assert_eq!(p.messages().collect::<Vec<_>>(), vec![(1000, m)]);
    }

    #[test]
    fn sizes_and_types_round_trip() {
        for k in MsgType::ALL {
            assert_eq!(k.size() % 8, 0);
            assert!(k.size() <= MAX_MSG_LEN);
            assert_eq!(MsgType::from_wire(k.wire()), Some(k));
        }
        assert_eq!(MsgType::from_wire(0), None);
        assert_eq!(MsgType::from_wire(8), None);
    }

    #[test]
    fn rejects_each_malformation() {
        let good = hex(GOLDEN);
        assert!(Packet::parse(&good).is_ok());
        let cases: [(usize, u8, DecodeError); 9] = [
            (0, 0x52, DecodeError::BadMagic),
            (2, 2, DecodeError::BadVersion(2)),
            (3, 2, DecodeError::Truncated),
            (16, 41, DecodeError::BadMsgLen),
            (18, 9, DecodeError::UnknownType(9)),
            (19, 1, DecodeError::NonZeroReserved),
            (52, 3, DecodeError::BadSide(3)),
            (53, 1, DecodeError::NonZeroReserved),
            (32, 0, DecodeError::ZeroField), // order id 42 -> 0
        ];
        for (off, v, e) in cases {
            let mut p = good.clone();
            p[off] = v;
            assert_eq!(Packet::parse(&p).unwrap_err(), e, "offset {off}");
        }
        let mut p = good.clone();
        p[3] = 0;
        assert_eq!(Packet::parse(&p).unwrap_err(), DecodeError::TrailingBytes);
        assert_eq!(
            Packet::parse(&good[..55]).unwrap_err(),
            DecodeError::Truncated
        );
        assert_eq!(
            Packet::parse(&vec![0u8; 1473]).unwrap_err(),
            DecodeError::TooLong
        );
        // Snapshot end marker must have zero price/qty/side.
        let mut b = [0u8; 48];
        Msg::snapshot_end(1, 2, 3).encode(&mut b);
        assert!(decode_msg(&b).is_ok());
        b[32] = 1;
        assert_eq!(decode_msg(&b).unwrap_err(), DecodeError::ZeroField);
    }

    #[test]
    fn snapshot_messages_consume_no_sequence() {
        let mut w = PacketWriter::new(1, 50);
        let s = Msg::Snapshot {
            instrument: 1,
            ts_ns: 0,
            flags: 0,
            order_id: 9,
            side: Side::Buy,
            price: 5,
            qty: 1,
            as_of_seq: 50,
        };
        w.push(&s);
        w.push(&Msg::snapshot_end(1, 0, 50));
        let mut pkt = Vec::new();
        w.flush(&mut |b| pkt = b.to_vec());
        let seqs: Vec<u64> = Packet::parse(&pkt)
            .unwrap()
            .messages()
            .map(|x| x.0)
            .collect();
        assert_eq!(seqs, vec![50, 50]);
    }

    #[test]
    fn recording_round_trip() {
        let mut rec = Vec::new();
        recording_header(&mut rec);
        let g = hex(GOLDEN);
        record_packet(&mut rec, &g);
        record_packet(&mut rec, &g);
        assert_eq!(split_recording(&rec).unwrap(), vec![&g[..], &g[..]]);
        assert_eq!(
            split_recording(&rec[..rec.len() - 1]).unwrap_err(),
            DecodeError::Truncated
        );
        assert_eq!(
            split_recording(b"QPBFREC2").unwrap_err(),
            DecodeError::BadMagic
        );
    }
}
