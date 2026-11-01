//! The JSON-lines shim answers a good request and reports a bad one.

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn shim_answers_one_good_and_one_bad_request() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_eq_jsonl"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn shim");
    let good = "{\"id\":1,\"fn\":\"bs\",\"S\":100,\"K\":100,\"T\":1,\"r\":0.05,\"sigma\":0.2,\"q\":0,\"type\":\"call\"}";
    let bad = good
        .replace("\"S\":100", "\"S\":-1")
        .replace("\"id\":1", "\"id\":2");
    let input = format!(
        "{good}
{bad}
"
    );
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    assert!(
        lines[0].starts_with("{\"id\":1,\"ok\":true,\"out\":{\"price\":10.45058357218"),
        "{}",
        lines[0]
    );
    assert!(
        lines[1].starts_with("{\"id\":2,\"ok\":false,\"err\":"),
        "{}",
        lines[1]
    );
}
