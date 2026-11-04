//! Non-allocating Binance diff-depth decoder.
//!
//! # Plain idea
//!
//! The generic reader in [`crate::json`] builds a tree of owned strings for
//! every message, which costs several allocations per update. A diff-depth
//! event has a fixed, flat shape, so the hot path reads it straight from the
//! bytes into a [`Delta`] the caller owns and reuses: the change vector is
//! cleared, not dropped, so after the first few messages it never grows.
//!
//! The fast decoder never decides that a message is malformed. Anything it
//! does not fully understand - an escape inside a string, a duplicate key,
//! keys in an unusual order, a REST snapshot, a non-depth event, more
//! changes than the vector already holds room for - makes it answer
//! [`Fast::Fallback`], and the caller runs the generic parser, which owns
//! every error message. So the fast path can only be faster, never
//! different: the differential test in tests/no_alloc.rs checks that every
//! message the fast path accepts parses to the same `Delta` the generic
//! parser produces, on generated days and on fuzzed bytes.
//!
//! # Formal statement
//!
//! `binance_into(raw, d)` returns `Fast::Delta` only if `raw` is a JSON
//! object (optionally the combined-stream wrapper `{"stream":..,"data":{..}}`)
//! that the strict reader accepts, whose event has `"e":"depthUpdate"`, plain
//! unsigned integers `U <= u`, and arrays `b` (before `a`) of
//! `[price, size]` string pairs that `parse_fixed` accepts. Then `*d` equals
//! `venue::binance(raw)`'s `Msg::Delta`.
//!
//! # Worked example
//!
//! `{"e":"depthUpdate","E":1,"s":"X","U":5,"u":6,"b":[["1.5","2"]],"a":[]}`
//! fills `d` with `first = 5`, `last = 6` and one change, bid 1.5 -> 2.
//! The same text with `"s":"X\u0041"` falls back (string escape).
//!
//! # Coinbase level2
//!
//! `coinbase_level2_into(raw, out)` applies the same rule to the `l2update`
//! message of the Coinbase `level2` and `level2_batch` channels. It returns
//! `Fast::Delta` only if `raw` is a strict JSON object with exactly one
//! `"type":"l2update"` and exactly one `changes` array of
//! `[side, price, size]` string triples, `side` being `buy` or `sell`. Then
//! `*out` equals the vector in `venue::coinbase_level2(raw)`'s
//! `Msg::Unsequenced`. Snapshots, subscription acks, four-element changes and
//! everything else fall back. `product_id` is not read here, as it is not in
//! the generic parser.
//!
//! `{"type":"l2update","changes":[["buy","1.5","2"]]}` gives one change, bid
//! 1.5 -> 2.

use crate::book::{Change, Delta, Side, parse_fixed};

/// Outcome of the fast decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fast {
    /// The output argument holds the decoded message.
    Delta,
    /// Use the generic parser for this message.
    Fallback,
}

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

type R<T> = Result<T, ()>;

// Stricter than the generic reader (64), so a deep value always falls back.
const MAX_DEPTH: usize = 32;

impl<'a> Cur<'a> {
    #[inline]
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    #[inline]
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    #[inline]
    fn eat(&mut self, c: u8) -> R<()> {
        self.ws();
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(())
        }
    }
    /// A string with no escapes, no control characters and ASCII only;
    /// returns the bytes between the quotes.
    #[inline]
    fn str(&mut self) -> R<&'a [u8]> {
        self.ws();
        if self.peek() != Some(b'"') {
            return Err(());
        }
        self.i += 1;
        let s = self.i;
        loop {
            match self.peek() {
                Some(b'"') => {
                    let out = &self.b[s..self.i];
                    self.i += 1;
                    return Ok(out);
                }
                Some(c) if (0x20..0x7f).contains(&c) && c != b'\\' => self.i += 1,
                _ => return Err(()),
            }
        }
    }
    /// RFC 8259 number; returns its text.
    fn num(&mut self) -> R<&'a [u8]> {
        let s = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            _ => return Err(()),
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        Ok(&self.b[s..self.i])
    }
    fn lit(&mut self, w: &[u8]) -> R<()> {
        if self.b[self.i..].starts_with(w) {
            self.i += w.len();
            Ok(())
        } else {
            Err(())
        }
    }
    /// Validate and skip any value.
    fn skip(&mut self, depth: usize) -> R<()> {
        if depth > MAX_DEPTH {
            return Err(());
        }
        self.ws();
        match self.peek() {
            Some(b'"') => self.str().map(|_| ()),
            Some(b'{') => {
                self.i += 1;
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    self.str()?;
                    self.eat(b':')?;
                    self.skip(depth + 1)?;
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(()),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    self.skip(depth + 1)?;
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(()),
                    }
                }
            }
            Some(b't') => self.lit(b"true"),
            Some(b'f') => self.lit(b"false"),
            Some(b'n') => self.lit(b"null"),
            _ => self.num().map(|_| ()),
        }
    }
    /// Plain unsigned integer that fits u64.
    fn uint(&mut self) -> R<u64> {
        self.ws();
        let t = self.num()?;
        let mut v: u64 = 0;
        if t.is_empty() {
            return Err(());
        }
        for &c in t {
            if !c.is_ascii_digit() {
                return Err(());
            }
            v = v
                .checked_mul(10)
                .ok_or(())?
                .checked_add((c - b'0') as u64)
                .ok_or(())?;
        }
        Ok(v)
    }
    /// `[["px","qty"], ...]` appended to `out` as level changes, without
    /// growing it past its capacity.
    fn levels(&mut self, side: Side, out: &mut Vec<Change>) -> R<()> {
        self.eat(b'[')?;
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(());
        }
        loop {
            self.eat(b'[')?;
            let p = self.str()?;
            self.eat(b',')?;
            let q = self.str()?;
            self.eat(b']')?;
            let (px, qty) = (fixed(p)?, fixed(q)?);
            if out.len() == out.capacity() {
                return Err(()); // would allocate
            }
            out.push(Change::Level { side, px, qty });
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(());
                }
                _ => return Err(()),
            }
        }
    }
    /// `[["buy","px","qty"], ...]` appended to `out`, without growing it past
    /// its capacity. A change with extra elements falls back.
    fn changes(&mut self, out: &mut Vec<Change>) -> R<()> {
        self.eat(b'[')?;
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(());
        }
        loop {
            self.eat(b'[')?;
            let side = match self.str()? {
                b"buy" => Side::Bid,
                b"sell" => Side::Ask,
                _ => return Err(()),
            };
            self.eat(b',')?;
            let p = self.str()?;
            self.eat(b',')?;
            let q = self.str()?;
            self.eat(b']')?;
            let (px, qty) = (fixed(p)?, fixed(q)?);
            if out.len() == out.capacity() {
                return Err(()); // would allocate
            }
            out.push(Change::Level { side, px, qty });
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(());
                }
                _ => return Err(()),
            }
        }
    }
}

/// `parse_fixed` on bytes `Cur::str` returned. They are ASCII, so the UTF-8
/// check cannot fail.
#[inline]
fn fixed(s: &[u8]) -> R<i64> {
    parse_fixed(std::str::from_utf8(s).map_err(|_| ())?).ok_or(())
}

/// Decode one event object starting at `c.i`. `wrapped` is true inside the
/// combined-stream `data` member.
fn event(c: &mut Cur<'_>, d: &mut Delta, wrapped: bool) -> R<()> {
    c.eat(b'{')?;
    // Seen flags: e, U, u, b, a, data.
    let (mut e, mut fu, mut lu, mut b, mut a, mut data) = (false, None, None, false, false, false);
    c.ws();
    if c.peek() == Some(b'}') {
        return Err(());
    }
    loop {
        let k = c.str()?;
        c.eat(b':')?;
        c.ws();
        match k {
            b"e" if !e => {
                if c.str()? != b"depthUpdate" {
                    return Err(());
                }
                e = true;
            }
            b"U" if fu.is_none() => fu = Some(c.uint()?),
            b"u" if lu.is_none() => lu = Some(c.uint()?),
            b"b" if !b && !a => {
                c.levels(Side::Bid, &mut d.changes)?;
                b = true;
            }
            b"a" if !a && b => {
                c.levels(Side::Ask, &mut d.changes)?;
                a = true;
            }
            b"data" if !wrapped && !data => {
                if e || fu.is_some() || lu.is_some() || b || a {
                    return Err(());
                }
                event(c, d, true)?;
                data = true;
            }
            b"e" | b"U" | b"u" | b"b" | b"a" | b"data" | b"lastUpdateId" => return Err(()),
            _ => c.skip(1)?,
        }
        c.ws();
        match c.peek() {
            Some(b',') => c.i += 1,
            Some(b'}') => {
                c.i += 1;
                break;
            }
            _ => return Err(()),
        }
    }
    if data {
        // The wrapper may carry only other keys besides `data`; the event
        // fields were read inside it.
        return if e || fu.is_some() || lu.is_some() || b || a {
            Err(())
        } else {
            Ok(())
        };
    }
    match (e, fu, lu, b, a) {
        (true, Some(f), Some(l), true, true) if f <= l => {
            d.first = f;
            d.last = l;
            Ok(())
        }
        _ => Err(()),
    }
}

/// Decode a Binance diff-depth event into `d` without allocating.
pub fn binance_into(raw: &[u8], d: &mut Delta) -> Fast {
    d.changes.clear();
    let mut c = Cur { b: raw, i: 0 };
    let ok = event(&mut c, d, false).is_ok() && {
        c.ws();
        c.i == raw.len()
    };
    if ok {
        Fast::Delta
    } else {
        d.changes.clear();
        Fast::Fallback
    }
}

fn l2update(c: &mut Cur<'_>, out: &mut Vec<Change>) -> R<()> {
    c.eat(b'{')?;
    let (mut ty, mut ch) = (false, false);
    c.ws();
    if c.peek() == Some(b'}') {
        return Err(());
    }
    loop {
        let k = c.str()?;
        c.eat(b':')?;
        match k {
            b"type" if !ty => {
                if c.str()? != b"l2update" {
                    return Err(());
                }
                ty = true;
            }
            b"changes" if !ch => {
                c.changes(out)?;
                ch = true;
            }
            b"type" | b"changes" => return Err(()),
            _ => c.skip(1)?,
        }
        c.ws();
        match c.peek() {
            Some(b',') => c.i += 1,
            Some(b'}') => {
                c.i += 1;
                break;
            }
            _ => return Err(()),
        }
    }
    if ty && ch { Ok(()) } else { Err(()) }
}

/// Decode a Coinbase `l2update` into `out` without allocating. On
/// [`Fast::Fallback`] `out` is empty and the generic parser must decide.
pub fn coinbase_level2_into(raw: &[u8], out: &mut Vec<Change>) -> Fast {
    out.clear();
    let mut c = Cur { b: raw, i: 0 };
    let ok = l2update(&mut c, out).is_ok() && {
        c.ws();
        c.i == raw.len()
    };
    if ok {
        Fast::Delta
    } else {
        out.clear();
        Fast::Fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::venue::{Msg, binance, coinbase_level2};

    fn both(raw: &str) -> (Fast, Delta, Msg) {
        let mut d = Delta {
            first: 0,
            last: 0,
            changes: Vec::with_capacity(16),
        };
        let f = binance_into(raw.as_bytes(), &mut d);
        (f, d, binance(raw.as_bytes()).unwrap_or(Msg::Ignore))
    }

    #[test]
    fn worked_example() {
        let (f, d, m) =
            both(r#"{"e":"depthUpdate","E":1,"s":"X","U":5,"u":6,"b":[["1.5","2"]],"a":[]}"#);
        assert_eq!(f, Fast::Delta);
        assert_eq!((d.first, d.last, d.changes.len()), (5, 6, 1));
        assert_eq!(m, Msg::Delta(d));
        let (f, _, _) =
            both(r#"{"e":"depthUpdate","E":1,"s":"X\u0041","U":5,"u":6,"b":[],"a":[]}"#);
        assert_eq!(f, Fast::Fallback);
    }

    #[test]
    fn coinbase_worked_example_and_fallbacks() {
        let mut out = Vec::with_capacity(16);
        let raw = br#" {"time":"t","changes":[["buy","1.5","2"],["sell","3","0.0"]],"product_id":"X","type":"l2update"} "#;
        assert_eq!(coinbase_level2_into(raw, &mut out), Fast::Delta);
        assert_eq!(out.len(), 2);
        assert_eq!(coinbase_level2(raw), Ok(Msg::Unsequenced(out.clone())));
        for raw in [
            r#"{"type":"snapshot","bids":[],"asks":[]}"#,
            r#"{"type":"subscriptions","channels":[]}"#,
            r#"{"changes":[]}"#,
            r#"{"type":"l2update"}"#,
            r#"{"type":"l2update","type":"l2update","changes":[]}"#,
            r#"{"type":"l2update","changes":[],"changes":[]}"#,
            r#"{"typ\u0065":"l2update","changes":[]}"#,
            r#"{"type":"l2update","changes":[["bid","1","2"]]}"#,
            r#"{"type":"l2update","changes":[["buy","1","2","3"]]}"#,
            r#"{"type":"l2update","changes":[["buy","1"]]}"#,
            r#"{"type":"l2update","changes":[["buy","-1","2"]]}"#,
            r#"{"type":"l2update","changes":[["buy",1,"2"]]}"#,
            r#"{"type":"l2update","changes":[["buy","1.000000001","2"]]}"#,
            r#"{"type":"l2update","changes":{}}"#,
            r#"{"type":"l2update","changes":[]} x"#,
            r#"{"type":"l2update","x":01,"changes":[]}"#,
            "",
        ] {
            assert_eq!(
                coinbase_level2_into(raw.as_bytes(), &mut out),
                Fast::Fallback,
                "{raw}"
            );
            assert!(out.is_empty());
        }
        // Capacity is a hard limit here too.
        let mut none = Vec::new();
        let raw = br#"{"type":"l2update","changes":[["buy","1","2"]]}"#;
        assert_eq!(coinbase_level2_into(raw, &mut none), Fast::Fallback);
        assert_eq!(none.capacity(), 0);
    }

    #[test]
    fn wrapper_and_fallbacks() {
        let (f, d, m) = both(
            r#" {"stream":"x@depth","data":{"e":"depthUpdate","E":1,"s":"X","U":5,"u":5,"b":[],"a":[["2.0","0"]]}} "#,
        );
        assert_eq!(f, Fast::Delta);
        assert_eq!(m, Msg::Delta(d));
        for raw in [
            r#"{"lastUpdateId":1,"bids":[],"asks":[]}"#,
            r#"{"e":"trade","U":1,"u":1,"b":[],"a":[]}"#,
            r#"{"e":"depthUpdate","U":2,"u":1,"b":[],"a":[]}"#,
            r#"{"e":"depthUpdate","U":1,"u":1,"a":[],"b":[]}"#,
            r#"{"e":"depthUpdate","U":1,"U":1,"u":1,"b":[],"a":[]}"#,
            r#"{"e":"depthUpdate","U":1.0,"u":1,"b":[],"a":[]}"#,
            r#"{"e":"depthUpdate","U":1,"u":1,"b":[["1","2","3"]],"a":[]}"#,
            r#"{"e":"depthUpdate","U":1,"u":1,"b":[],"a":[]} x"#,
            r#"{"e":"depthUpdate","U":1,"u":1,"x":01,"b":[],"a":[]}"#,
            r#"{"e":"depthUpdate","U":1,"u":99999999999999999999,"b":[],"a":[]}"#,
            "",
        ] {
            assert_eq!(both(raw).0, Fast::Fallback, "{raw}");
        }
        // Capacity is a hard limit: no growth on the hot path.
        let mut d = Delta {
            first: 0,
            last: 0,
            changes: Vec::new(),
        };
        let raw = br#"{"e":"depthUpdate","U":1,"u":1,"b":[["1","2"]],"a":[]}"#;
        assert_eq!(binance_into(raw, &mut d), Fast::Fallback);
        assert_eq!(d.changes.capacity(), 0);
    }
}
