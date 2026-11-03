//! Kill-contract parity: replay python/oms/kill/vectors.json (the W28d golden
//! vectors written by the Python reference) and compare every line byte for
//! byte. The file is read from the repository, never copied, so a change to
//! the reference shows up here at once.

use std::path::PathBuf;

use qp_ull_feed::json::{self, Value};
use qp_ull_gateway::replay;

fn vectors() -> Value {
    load(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../python/oms/kill/vectors.json"))
}

fn load(p: PathBuf) -> Value {
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    json::parse(&s).expect("vectors.json parses")
}

/// Differential run: set QP_KILL_VECTORS to a file of the same shape written
/// by the Python reference (for example from randomised command lists) and
/// every case is compared the same way. Skipped when the variable is unset.
#[test]
fn extra_vectors_from_env() {
    if let Ok(p) = std::env::var("QP_KILL_VECTORS") {
        check(&load(PathBuf::from(p)));
    }
}

#[test]
fn golden_vectors_byte_for_byte() {
    let v = vectors();
    check(&v);
}

fn check(v: &Value) {
    assert_eq!(
        v.get("contract").and_then(Value::as_str),
        Some("docs/KILL_CONTRACT.md")
    );
    assert_eq!(v.get("version").and_then(Value::as_u64), Some(1));
    let cases = v.get("cases").and_then(Value::as_arr).unwrap();
    assert!(cases.len() >= 3);
    let mut lines = 0;
    for c in cases {
        let name = c.get("name").and_then(Value::as_str).unwrap();
        let got = replay(c.get("commands").and_then(Value::as_arr).unwrap()).unwrap();
        let want: Vec<&str> = c
            .get("expected")
            .and_then(Value::as_arr)
            .unwrap()
            .iter()
            .map(|e| e.as_str().unwrap())
            .collect();
        assert_eq!(got.len(), want.len(), "case {name}: line count");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.as_bytes(), w.as_bytes(), "case {name} line {i}");
        }
        lines += got.len();
    }
    assert!(lines > 0);
}
