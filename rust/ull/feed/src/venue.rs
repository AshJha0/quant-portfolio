//! Venue message parsers: Binance diff-depth and Coinbase Exchange
//! `full` / `level2`. Public channels only; no authentication is needed.
//!
//! Sources (message shapes and the sync procedure):
//! - Binance Spot WebSocket Streams, "Diff. Depth Stream" and "How to manage
//!   a local order book correctly" (fields `e, E, s, U, u, b, a`; REST
//!   `GET /api/v3/depth` returns `lastUpdateId, bids, asks`).
//! - Coinbase Exchange WebSocket docs, "Full channel" (`received`, `open`,
//!   `done`, `match`, `change`, `activate`, each with `sequence`) and
//!   "Level2 channel" (`snapshot`, `l2update`, no sequence); REST
//!   `GET /products/<id>/book?level=3` returns `sequence, bids, asks` with
//!   `[price, size, order_id]` entries.
//!
//! Every parser maps one raw message to one [`Msg`]. Unknown message types
//! map to `Msg::Ignore` (subscriptions acks, heartbeats); malformed fields are
//! an error, never a panic (fuzzed in tests/inject.rs).

use crate::book::{Change, Delta, Side, Snapshot, parse_fixed};
use crate::json::{Value, parse};

/// One parsed venue message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Msg {
    /// Sequenced delta.
    Delta(Delta),
    /// Book snapshot.
    Snapshot(Snapshot),
    /// Coinbase level2 update: carries no sequence number, so gaps cannot be
    /// detected on this channel (stated limit; use `full` for gap recovery).
    Unsequenced(Vec<Change>),
    /// Not a book message.
    Ignore,
}

/// A malformed message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub &'static str);

fn px(v: Option<&Value>) -> Result<i64, ParseError> {
    v.and_then(Value::as_str)
        .and_then(parse_fixed)
        .ok_or(ParseError("bad decimal"))
}

fn levels(
    v: Option<&Value>,
    side: Side,
    out: &mut Vec<(Side, i64, i64)>,
) -> Result<(), ParseError> {
    for e in v
        .and_then(Value::as_arr)
        .ok_or(ParseError("missing levels"))?
    {
        let a = e.as_arr().ok_or(ParseError("level not an array"))?;
        if a.len() < 2 {
            return Err(ParseError("short level"));
        }
        out.push((side, px(a.first())?, px(a.get(1))?));
    }
    Ok(())
}

fn root(raw: &[u8]) -> Result<Value, ParseError> {
    let s = std::str::from_utf8(raw).map_err(|_| ParseError("not utf-8"))?;
    parse(s).map_err(|e| ParseError(e.1))
}

/// Binance diff-depth event or REST depth snapshot.
pub fn binance(raw: &[u8]) -> Result<Msg, ParseError> {
    let v = root(raw)?;
    // Combined-stream wrapper {"stream": ..., "data": {...}}.
    let v = match v.get("data") {
        Some(d) => d.clone(),
        None => v,
    };
    if let Some(id) = v.get("lastUpdateId") {
        let mut s = Snapshot {
            last: id.as_u64().ok_or(ParseError("bad lastUpdateId"))?,
            ..Default::default()
        };
        levels(v.get("bids"), Side::Bid, &mut s.levels)?;
        levels(v.get("asks"), Side::Ask, &mut s.levels)?;
        return Ok(Msg::Snapshot(s));
    }
    if v.get("e").and_then(Value::as_str) != Some("depthUpdate") {
        return Ok(Msg::Ignore);
    }
    let first = v
        .get("U")
        .and_then(Value::as_u64)
        .ok_or(ParseError("bad U"))?;
    let last = v
        .get("u")
        .and_then(Value::as_u64)
        .ok_or(ParseError("bad u"))?;
    if first > last {
        return Err(ParseError("U > u"));
    }
    let mut l = Vec::new();
    levels(v.get("b"), Side::Bid, &mut l)?;
    levels(v.get("a"), Side::Ask, &mut l)?;
    let changes = l
        .into_iter()
        .map(|(side, px, qty)| Change::Level { side, px, qty })
        .collect();
    Ok(Msg::Delta(Delta {
        first,
        last,
        changes,
    }))
}

/// Parse a UUID (with or without hyphens) into a u128.
pub fn uuid(s: &str) -> Option<u128> {
    let mut v: u128 = 0;
    let mut n = 0;
    for c in s.bytes() {
        if c == b'-' {
            continue;
        }
        v = (v << 4) | (c as char).to_digit(16)? as u128;
        n += 1;
    }
    (n == 32).then_some(v)
}

fn side(v: Option<&Value>) -> Result<Side, ParseError> {
    match v.and_then(Value::as_str) {
        Some("buy") => Ok(Side::Bid),
        Some("sell") => Ok(Side::Ask),
        _ => Err(ParseError("bad side")),
    }
}

fn oid(v: Option<&Value>) -> Result<u128, ParseError> {
    v.and_then(Value::as_str)
        .and_then(uuid)
        .ok_or(ParseError("bad order id"))
}

/// Coinbase Exchange `full` channel message or level-3 REST snapshot.
pub fn coinbase_full(raw: &[u8]) -> Result<Msg, ParseError> {
    let v = root(raw)?;
    let seq = v.get("sequence").and_then(Value::as_u64);
    let Some(t) = v.get("type").and_then(Value::as_str) else {
        // REST level-3 book: {"sequence", "bids": [[px, size, id]], "asks"}.
        if v.get("bids").is_some() {
            let mut s = Snapshot {
                last: seq.ok_or(ParseError("bad sequence"))?,
                ..Default::default()
            };
            for (key, sd) in [("bids", Side::Bid), ("asks", Side::Ask)] {
                for e in v
                    .get(key)
                    .and_then(Value::as_arr)
                    .ok_or(ParseError("missing levels"))?
                {
                    let a = e.as_arr().ok_or(ParseError("entry not an array"))?;
                    if a.len() < 3 {
                        return Err(ParseError("short entry"));
                    }
                    s.orders
                        .push((oid(a.get(2))?, sd, px(a.first())?, px(a.get(1))?));
                }
            }
            return Ok(Msg::Snapshot(s));
        }
        return Ok(Msg::Ignore);
    };
    let changes = match t {
        "received" | "activate" => vec![],
        "open" => vec![Change::Open {
            id: oid(v.get("order_id"))?,
            side: side(v.get("side"))?,
            px: px(v.get("price"))?,
            qty: px(v.get("remaining_size"))?,
        }],
        "done" => vec![Change::Done {
            id: oid(v.get("order_id"))?,
        }],
        "match" => vec![Change::Match {
            id: oid(v.get("maker_order_id"))?,
            qty: px(v.get("size"))?,
        }],
        "change" => match v.get("new_size") {
            Some(n) => vec![Change::Resize {
                id: oid(v.get("order_id"))?,
                qty: px(Some(n))?,
            }],
            None => vec![], // new_funds on market orders: not on the book
        },
        _ => return Ok(Msg::Ignore),
    };
    let s = seq.ok_or(ParseError("bad sequence"))?;
    Ok(Msg::Delta(Delta {
        first: s,
        last: s,
        changes,
    }))
}

/// Coinbase Exchange `level2` channel message.
pub fn coinbase_level2(raw: &[u8]) -> Result<Msg, ParseError> {
    let v = root(raw)?;
    match v.get("type").and_then(Value::as_str) {
        Some("snapshot") => {
            let mut s = Snapshot::default();
            levels(v.get("bids"), Side::Bid, &mut s.levels)?;
            levels(v.get("asks"), Side::Ask, &mut s.levels)?;
            Ok(Msg::Snapshot(s))
        }
        Some("l2update") => {
            let mut out = Vec::new();
            for e in v
                .get("changes")
                .and_then(Value::as_arr)
                .ok_or(ParseError("missing changes"))?
            {
                let a = e.as_arr().ok_or(ParseError("change not an array"))?;
                if a.len() < 3 {
                    return Err(ParseError("short change"));
                }
                out.push(Change::Level {
                    side: side(a.first())?,
                    px: px(a.get(1))?,
                    qty: px(a.get(2))?,
                });
            }
            Ok(Msg::Unsequenced(out))
        }
        _ => Ok(Msg::Ignore),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binance_shapes_from_docs() {
        // Shape from the Binance Spot docs "Diff. Depth Stream" example.
        let d = br#"{"e":"depthUpdate","E":1672515782136,"s":"BNBBTC","U":157,"u":160,"b":[["0.0024","10"]],"a":[["0.0026","100"]]}"#;
        let Msg::Delta(d) = binance(d).unwrap() else {
            panic!()
        };
        assert_eq!((d.first, d.last, d.changes.len()), (157, 160, 2));
        assert_eq!(
            d.changes[0],
            Change::Level {
                side: Side::Bid,
                px: 240_000,
                qty: 10 * crate::book::SCALE
            }
        );
        let s = br#"{"lastUpdateId":1027024,"bids":[["4.00000000","431.00000000"]],"asks":[["4.00000200","12.00000000"]]}"#;
        let Msg::Snapshot(s) = binance(s).unwrap() else {
            panic!()
        };
        assert_eq!((s.last, s.levels.len()), (1027024, 2));
        assert_eq!(binance(br#"{"result":null,"id":1}"#).unwrap(), Msg::Ignore);
        assert!(binance(br#"{"e":"depthUpdate","U":5,"u":4,"b":[],"a":[]}"#).is_err());
    }

    #[test]
    fn coinbase_shapes_from_docs() {
        let id = "d50ec984-77a8-460a-b958-66f114b0de9b";
        let open = format!(
            r#"{{"type":"open","time":"2014-11-07T08:19:27.028459Z","product_id":"BTC-USD","sequence":10,"order_id":"{id}","price":"200.2","remaining_size":"1.00","side":"sell"}}"#
        );
        let Msg::Delta(d) = coinbase_full(open.as_bytes()).unwrap() else {
            panic!()
        };
        assert_eq!((d.first, d.last), (10, 10));
        assert!(matches!(
            d.changes[0],
            Change::Open {
                side: Side::Ask,
                ..
            }
        ));
        let rec = br#"{"type":"received","sequence":11,"order_id":"d50ec984-77a8-460a-b958-66f114b0de9b"}"#;
        let Msg::Delta(d) = coinbase_full(rec).unwrap() else {
            panic!()
        };
        assert!(d.changes.is_empty());
        let snap = format!(r#"{{"sequence":9,"bids":[["200.1","2","{id}"]],"asks":[]}}"#);
        let Msg::Snapshot(s) = coinbase_full(snap.as_bytes()).unwrap() else {
            panic!()
        };
        assert_eq!((s.last, s.orders.len()), (9, 1));
        let l2 = br#"{"type":"l2update","product_id":"BTC-USD","time":"2019-08-14T20:42:27.265Z","changes":[["buy","10101.80000000","0.162567"]]}"#;
        assert!(matches!(coinbase_level2(l2).unwrap(), Msg::Unsequenced(c) if c.len() == 1));
        assert_eq!(uuid(id), Some(0xd50ec98477a8460ab95866f114b0de9b));
        assert_eq!(uuid("xyz"), None);
    }
}
