//! W30-rs part 2: order gateway with pre-trade checks and the shared kill
//! contract.
//!
//! Plain idea: every order passes the kill switch first (K1 in
//! docs/KILL_CONTRACT.md), then the book-health gate (no orders priced off a
//! stale book), then size, notional and price-band limits. An accepted order
//! is registered as working so a later kill can return it for cancel.
//!
//! Formal statement: `submit` returns the first failing check in the fixed
//! order kill, stale, quantity, notional, band. Notional is
//! `px * qty / SCALE` in the 1e8 fixed point of `qp_ull_feed::book`; the band
//! rejects `|px - mid| * 10_000 > band_bps * mid`.
//!
//! Worked example: limits qty 10, notional 1,000, band 50 bp, mid 100. Buy 5
//! at 100.4 passes (notional 502, 40 bp). Buy 5 at 100.6 fails the band
//! (60 bp). After a DESK kill the same order fails with
//! `KILLED:DESK:QP/RATES/*/*` before any limit is looked at.
//!
//! Scope note: the W28d check set in the Python OMS is wider (working-order
//! exposure, self-trade prevention). This gateway carries the kill contract
//! at full parity and the four checks above; matching the rest of W28d is an
//! integration item recorded in the handback.

#![warn(missing_docs)]

pub mod kill;

use kill::{KillError, KillSwitch, Path};
use qp_ull_feed::book::{SCALE, Side};
use qp_ull_feed::json::{self, Value};

/// Pre-trade limits (fixed point 1e8 for qty and notional).
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Largest order quantity.
    pub max_qty: i64,
    /// Largest order notional.
    pub max_notional: i64,
    /// Price band around mid, in basis points.
    pub band_bps: i64,
}

/// A new order.
#[derive(Clone, Debug)]
pub struct Order {
    /// Client order id.
    pub id: u64,
    /// `(firm, desk, strategy, symbol)`.
    pub path: Path,
    /// Side.
    pub side: Side,
    /// Limit price (1e8).
    pub px: i64,
    /// Quantity (1e8).
    pub qty: i64,
}

/// Why an order was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reject {
    /// K1, with the contract's reason string.
    Killed(String),
    /// The book is stale (or there is no mid): no orders priced off it.
    StaleBook,
    /// Quantity not in `1..=max_qty`.
    Qty,
    /// Notional over the limit.
    Notional,
    /// Price outside the band.
    Band,
}

/// Gateway: kill switch plus limits.
#[derive(Debug)]
pub struct Gateway {
    /// The kill hierarchy (shared contract).
    pub kill: KillSwitch,
    limits: Limits,
}

impl Gateway {
    /// New gateway.
    pub fn new(limits: Limits) -> Self {
        Self {
            kill: KillSwitch::new(),
            limits,
        }
    }

    /// Pre-trade checks. `mid` is `None` when the feed is stale.
    pub fn submit(&mut self, o: &Order, mid: Option<i64>) -> Result<(), Reject> {
        if let Some(r) = self.kill.check(&o.path) {
            return Err(Reject::Killed(r));
        }
        let mid = mid.ok_or(Reject::StaleBook)?;
        let l = &self.limits;
        if o.qty <= 0 || o.qty > l.max_qty {
            return Err(Reject::Qty);
        }
        let notional = (o.px as i128 * o.qty as i128) / SCALE as i128;
        if notional > l.max_notional as i128 {
            return Err(Reject::Notional);
        }
        if ((o.px - mid).abs() as i128) * 10_000 > l.band_bps as i128 * mid as i128 {
            return Err(Reject::Band);
        }
        self.kill.on_order_working(o.id, o.path.clone());
        Ok(())
    }
}

/// Golden-vector replay errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// Command not understood.
    BadCommand(String),
    /// Contract error from the switch.
    Kill(KillError),
}

fn path(v: Option<&Value>) -> Result<Path, ReplayError> {
    let a = v
        .and_then(Value::as_arr)
        .filter(|a| a.len() == 4)
        .ok_or(ReplayError::BadCommand("path".into()))?;
    let s = |i: usize| {
        a[i].as_str()
            .map(str::to_string)
            .ok_or(ReplayError::BadCommand("path field".into()))
    };
    Ok([s(0)?, s(1)?, s(2)?, s(3)?])
}

fn text(c: &Value, k: &str) -> Result<String, ReplayError> {
    c.get(k)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or(ReplayError::BadCommand(k.into()))
}

fn int(c: &Value, k: &str) -> Result<u64, ReplayError> {
    c.get(k)
        .and_then(Value::as_u64)
        .ok_or(ReplayError::BadCommand(k.into()))
}

/// Run one golden-vector command list; returns the wire lines, the same
/// shape as `qp_oms.kill.replay`.
pub fn replay(commands: &[Value]) -> Result<Vec<String>, ReplayError> {
    let mut ks = KillSwitch::new();
    let mut out = Vec::new();
    let ke = ReplayError::Kill;
    for c in commands {
        let before = ks.events.len();
        match c.get("op").and_then(Value::as_str) {
            Some("working") => ks.on_order_working(int(c, "id")?, path(c.get("path"))?),
            Some("terminal") => ks.on_order_terminal(int(c, "id")?),
            Some("engage") => {
                let reason = match c.get("reason") {
                    Some(_) => text(c, "reason")?,
                    None => String::new(),
                };
                ks.engage(path(c.get("scope"))?, &text(c, "actor")?, &reason)
                    .map_err(ke)?;
            }
            Some("disconnect") => {
                let ps = c
                    .get("paths")
                    .and_then(Value::as_arr)
                    .ok_or(ReplayError::BadCommand("paths".into()))?;
                let ps: Vec<Path> = ps.iter().map(|p| path(Some(p))).collect::<Result<_, _>>()?;
                ks.cancel_on_disconnect(&ps).map_err(ke)?;
            }
            Some("request_reset") => {
                ks.request_reset(int(c, "kill_id")?, &text(c, "actor")?)
                    .map_err(ke)?;
            }
            Some("approve_reset") => {
                ks.approve_reset(int(c, "kill_id")?, &text(c, "actor")?)
                    .map_err(ke)?;
            }
            Some("check") => {
                let p = path(c.get("path"))?;
                let mut o = String::from("{\"check\":");
                kill::write_path(&mut o, &p);
                o.push_str(",\"result\":");
                match ks.check(&p) {
                    Some(r) => json::write_str(&mut o, &r),
                    None => o.push_str("null"),
                }
                o.push('}');
                out.push(o);
                continue;
            }
            other => return Err(ReplayError::BadCommand(format!("op {other:?}"))),
        }
        out.extend(ks.events[before..].iter().map(kill::KillEvent::to_wire));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(a: &str, b: &str, c: &str, d: &str) -> Path {
        [a.into(), b.into(), c.into(), d.into()]
    }

    #[test]
    fn worked_example() {
        let mut g = Gateway::new(Limits {
            max_qty: 10 * SCALE,
            max_notional: 1_000 * SCALE,
            band_bps: 50,
        });
        let mid = Some(100 * SCALE);
        let mut o = Order {
            id: 1,
            path: p("QP", "RATES", "MM1", "UST10Y"),
            side: Side::Bid,
            px: 10_040_000_000,
            qty: 5 * SCALE,
        };
        assert_eq!(g.submit(&o, mid), Ok(()));
        o.id = 2;
        o.px = 10_060_000_000;
        assert_eq!(g.submit(&o, mid), Err(Reject::Band));
        assert_eq!(g.submit(&o, None), Err(Reject::StaleBook));
        o.qty = 11 * SCALE;
        o.px = 100 * SCALE;
        assert_eq!(g.submit(&o, mid), Err(Reject::Qty));
        let (_, cancel) = g
            .kill
            .engage(p("QP", "RATES", "*", "*"), "alice", "")
            .unwrap();
        assert_eq!(cancel, vec![1]);
        o.qty = SCALE;
        assert_eq!(
            g.submit(&o, mid),
            Err(Reject::Killed("KILLED:DESK:QP/RATES/*/*".into()))
        );
    }

    #[test]
    fn notional_and_bad_scope() {
        let mut g = Gateway::new(Limits {
            max_qty: 100 * SCALE,
            max_notional: 1_000 * SCALE,
            band_bps: 10_000,
        });
        let o = Order {
            id: 1,
            path: p("QP", "FX", "S", "EURUSD"),
            side: Side::Ask,
            px: 101 * SCALE,
            qty: 10 * SCALE,
        };
        assert_eq!(g.submit(&o, Some(100 * SCALE)), Err(Reject::Notional));
        assert_eq!(
            g.kill.engage(p("QP", "*", "MM1", "*"), "a", ""),
            Err(KillError::InvalidScope)
        );
        assert_eq!(
            g.kill.engage(p("*", "*", "*", "*"), "a", ""),
            Err(KillError::InvalidScope)
        );
        assert_eq!(g.kill.request_reset(9, "a"), Err(KillError::UnknownKill(9)));
    }
}
