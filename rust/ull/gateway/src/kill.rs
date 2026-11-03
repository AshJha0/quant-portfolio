//! Kill-switch hierarchy: the Rust implementation of docs/KILL_CONTRACT.md.
//!
//! The Python OMS (`python/oms/kill/contract.py`) is the reference. This file
//! was written from the contract and the golden vectors; the parity test
//! (`tests/kill_parity.rs`) replays `python/oms/kill/vectors.json` and
//! compares every output line byte for byte.
//!
//! Plain idea: a kill stops new orders under a scope, returns the working
//! orders to cancel, and needs two different people to clear. Formal rules
//! K1-K7 are cited where each is enforced. Worked example: the vector case
//! `worked_example` is the contract's worked example, step for step.

use std::collections::BTreeMap;

use qp_ull_feed::json::write_str;

/// The wildcard field.
pub const WILD: &str = "*";

/// Kill level, by how many leading fields are fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Firm.
    Firm = 1,
    /// Desk.
    Desk = 2,
    /// Strategy.
    Strategy = 3,
    /// Symbol.
    Symbol = 4,
}

impl Level {
    /// Wire name.
    pub fn name(self) -> &'static str {
        match self {
            Level::Firm => "FIRM",
            Level::Desk => "DESK",
            Level::Strategy => "STRATEGY",
            Level::Symbol => "SYMBOL",
        }
    }
}

/// State of one kill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KillState {
    /// Cancels outstanding.
    Engaged,
    /// All cancels acknowledged.
    Killed,
    /// Reset requested, awaiting a second actor.
    ResetPending,
    /// Removed.
    Cleared,
}

impl KillState {
    /// Wire name.
    pub fn name(self) -> &'static str {
        match self {
            KillState::Engaged => "ENGAGED",
            KillState::Killed => "KILLED",
            KillState::ResetPending => "RESET_PENDING",
            KillState::Cleared => "CLEARED",
        }
    }
}

/// `(firm, desk, strategy, symbol)`.
pub type Path = [String; 4];

/// Contract errors (the Python reference raises on these).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KillError {
    /// Wildcards not a pure suffix after the firm.
    InvalidScope,
    /// No such kill id.
    UnknownKill(u64),
}

/// Level of a scope; wildcards must be a suffix after the firm.
pub fn validate_scope(scope: &Path) -> Result<Level, KillError> {
    let n = scope.iter().take_while(|s| *s != WILD).count();
    if n == 0 || scope[n..].iter().any(|s| s != WILD) {
        return Err(KillError::InvalidScope);
    }
    Ok([Level::Firm, Level::Desk, Level::Strategy, Level::Symbol][n - 1])
}

/// Truncate a path to a scope at `level`.
pub fn make_scope(level: Level, p: &Path) -> Path {
    let n = level as usize;
    std::array::from_fn(|i| {
        if i == 0 || i < n {
            p[i].clone()
        } else {
            WILD.to_string()
        }
    })
}

/// Every non-wildcard field of `scope` equals the path's field.
pub fn covers(scope: &Path, p: &Path) -> bool {
    scope.iter().zip(p).all(|(s, x)| s == WILD || s == x)
}

#[derive(Clone, Debug)]
struct Kill {
    id: u64,
    level: Level,
    scope: Path,
    state: KillState,
    outstanding: Vec<u64>,
    requester: Option<String>,
}

/// One emitted transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KillEvent {
    /// Sequence number, from 1.
    pub seq: u64,
    /// Event name.
    pub event: &'static str,
    /// Kill id.
    pub kill_id: u64,
    /// Level.
    pub level: Level,
    /// Scope.
    pub scope: Path,
    /// Actor.
    pub actor: String,
    /// State after the transition.
    pub state: KillState,
    /// Orders to cancel (engage only).
    pub cancel: Vec<u64>,
    /// Reason.
    pub reason: String,
}

impl KillEvent {
    /// Canonical JSON: sorted keys, no spaces, ASCII (the parity form).
    pub fn to_wire(&self) -> String {
        let mut o = String::with_capacity(160);
        o.push_str("{\"actor\":");
        write_str(&mut o, &self.actor);
        o.push_str(",\"cancel\":[");
        for (i, c) in self.cancel.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&c.to_string());
        }
        o.push_str("],\"event\":");
        write_str(&mut o, self.event);
        o.push_str(",\"kill_id\":");
        o.push_str(&self.kill_id.to_string());
        o.push_str(",\"level\":");
        write_str(&mut o, self.level.name());
        o.push_str(",\"reason\":");
        write_str(&mut o, &self.reason);
        o.push_str(",\"scope\":");
        write_path(&mut o, &self.scope);
        o.push_str(",\"seq\":");
        o.push_str(&self.seq.to_string());
        o.push_str(",\"state\":");
        write_str(&mut o, self.state.name());
        o.push('}');
        o
    }
}

/// `["a","b","c","d"]` in canonical form.
pub fn write_path(o: &mut String, p: &Path) {
    o.push('[');
    for (i, s) in p.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write_str(o, s);
    }
    o.push(']');
}

/// The firm-wide kill hierarchy.
#[derive(Debug, Default)]
pub struct KillSwitch {
    kills: Vec<Kill>, // id order == insertion order (ids are 1, 2, ...)
    seq: u64,
    working: BTreeMap<u64, Path>,
    /// Every transition, in order.
    pub events: Vec<KillEvent>,
}

impl KillSwitch {
    /// Empty switch.
    pub fn new() -> Self {
        Self::default()
    }

    fn emit(&mut self, ev: &'static str, k: usize, actor: &str, reason: &str, cancel: Vec<u64>) {
        self.seq += 1;
        let k = &self.kills[k];
        self.events.push(KillEvent {
            seq: self.seq,
            event: ev,
            kill_id: k.id,
            level: k.level,
            scope: k.scope.clone(),
            actor: actor.to_string(),
            state: k.state,
            cancel,
            reason: reason.to_string(),
        });
    }

    fn idx(&self, id: u64) -> Result<usize, KillError> {
        self.kills
            .iter()
            .position(|k| k.id == id)
            .ok_or(KillError::UnknownKill(id))
    }

    /// An order is working on `path`.
    pub fn on_order_working(&mut self, id: u64, path: Path) {
        self.working.insert(id, path);
    }

    /// K3: a terminal report (cancel ack, fill, reject, expiry).
    pub fn on_order_terminal(&mut self, id: u64) {
        self.working.remove(&id);
        for i in 0..self.kills.len() {
            let k = &mut self.kills[i];
            if let Some(p) = k.outstanding.iter().position(|&o| o == id) {
                k.outstanding.swap_remove(p);
                if k.state == KillState::Engaged && k.outstanding.is_empty() {
                    k.state = KillState::Killed;
                    self.emit("killed", i, "system", "", vec![]);
                }
            }
        }
    }

    /// K2, K7: engage; returns (kill id, sorted orders to cancel).
    pub fn engage(
        &mut self,
        scope: Path,
        actor: &str,
        reason: &str,
    ) -> Result<(u64, Vec<u64>), KillError> {
        let level = validate_scope(&scope)?;
        if let Some(i) = self
            .kills
            .iter()
            .position(|k| k.scope == scope && k.state != KillState::Cleared)
        {
            self.emit("engage_dup", i, actor, reason, vec![]);
            return Ok((self.kills[i].id, vec![]));
        }
        let id = self.kills.len() as u64 + 1;
        // BTreeMap iterates in ascending id order: the cancel list is sorted.
        let cancels: Vec<u64> = self
            .working
            .iter()
            .filter(|(_, p)| covers(&scope, p))
            .map(|(&o, _)| o)
            .collect();
        let state = if cancels.is_empty() {
            KillState::Killed
        } else {
            KillState::Engaged
        };
        self.kills.push(Kill {
            id,
            level,
            scope,
            state,
            outstanding: cancels.clone(),
            requester: None,
        });
        self.emit(
            "engage",
            self.kills.len() - 1,
            actor,
            reason,
            cancels.clone(),
        );
        Ok((id, cancels))
    }

    /// K6: a STRATEGY kill for every strategy on a dropped session.
    pub fn cancel_on_disconnect(
        &mut self,
        paths: &[Path],
    ) -> Result<Vec<(u64, Vec<u64>)>, KillError> {
        let mut s: Vec<Path> = paths
            .iter()
            .map(|p| make_scope(Level::Strategy, p))
            .collect();
        s.sort();
        s.dedup();
        s.into_iter()
            .map(|sc| self.engage(sc, "system:cod", "cancel_on_disconnect"))
            .collect()
    }

    /// K1: `None` if allowed, else the reason naming the widest covering kill.
    pub fn check(&self, p: &Path) -> Option<String> {
        let k = self
            .kills
            .iter()
            .filter(|k| k.state != KillState::Cleared && covers(&k.scope, p))
            .min_by_key(|k| (k.level, k.id))?;
        Some(format!("KILLED:{}:{}", k.level.name(), k.scope.join("/")))
    }

    /// K4 step 1.
    pub fn request_reset(&mut self, id: u64, actor: &str) -> Result<bool, KillError> {
        let i = self.idx(id)?;
        let st = self.kills[i].state;
        if st != KillState::Killed {
            self.emit(
                "reset_rejected",
                i,
                actor,
                &format!("state={}", st.name()),
                vec![],
            );
            return Ok(false);
        }
        self.kills[i].state = KillState::ResetPending;
        self.kills[i].requester = Some(actor.to_string());
        self.emit("reset_requested", i, actor, "", vec![]);
        Ok(true)
    }

    /// K4 step 2: the approver must differ from the requester.
    pub fn approve_reset(&mut self, id: u64, actor: &str) -> Result<bool, KillError> {
        let i = self.idx(id)?;
        let st = self.kills[i].state;
        if st != KillState::ResetPending {
            self.emit(
                "reset_rejected",
                i,
                actor,
                &format!("state={}", st.name()),
                vec![],
            );
            return Ok(false);
        }
        if self.kills[i].requester.as_deref() == Some(actor) {
            self.emit("reset_rejected", i, actor, "same_actor", vec![]);
            return Ok(false);
        }
        self.kills[i].state = KillState::Cleared;
        self.emit("cleared", i, actor, "", vec![]);
        Ok(true)
    }

    /// State of a kill.
    pub fn state(&self, id: u64) -> Result<KillState, KillError> {
        Ok(self.kills[self.idx(id)?].state)
    }
}
