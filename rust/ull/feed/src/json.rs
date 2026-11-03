//! A small JSON reader and the canonical writer used for parity output.
//!
//! Plain idea: the venues send JSON text and the kill-switch vectors are JSON,
//! and this crate is std only, so it carries a strict RFC 8259 reader of its
//! own. Numbers are kept as their source text, so a price such as
//! `"0.00012300"` or an id such as `160` is never routed through a float.
//!
//! The writer reproduces Python's `json.dumps(sort_keys=True,
//! separators=(",", ":"))` with the default `ensure_ascii=True`: object keys
//! sorted by code point, no spaces, non-ASCII escaped as `\uXXXX` (with
//! surrogate pairs above U+FFFF), and the short escapes `\" \\ \n \r \t \b \f`.
//! That is the canonical wire form in docs/KILL_CONTRACT.md.

use std::fmt::Write as _;

/// A parsed JSON value. Object members keep source order.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// A number, kept as its exact source text.
    Num(String),
    /// A string, unescaped.
    Str(String),
    /// An array.
    Arr(Vec<Value>),
    /// An object, members in source order.
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// Member lookup on an object (first match); `None` otherwise.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// The string payload, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    /// The array payload, if this is an array.
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
    /// An unsigned integer: a JSON number with only digits.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Num(n) => n.parse().ok(),
            _ => None,
        }
    }
    /// A signed integer.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Num(n) => n.parse().ok(),
            _ => None,
        }
    }
}

/// A parse failure with the byte offset where it was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError(pub usize, pub &'static str);

/// Parse one JSON document; trailing non-whitespace is an error.
pub fn parse(src: &str) -> Result<Value, JsonError> {
    let mut p = P {
        b: src.as_bytes(),
        i: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(JsonError(p.i, "trailing data"));
    }
    Ok(v)
}

const MAX_DEPTH: u32 = 64;

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: u32,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, m: &'static str) -> Result<T, JsonError> {
        Err(JsonError(self.i, m))
    }
    fn lit(&mut self, s: &[u8], v: Value) -> Result<Value, JsonError> {
        if self.b[self.i..].starts_with(s) {
            self.i += s.len();
            Ok(v)
        } else {
            self.err("bad literal")
        }
    }
    fn value(&mut self) -> Result<Value, JsonError> {
        self.ws();
        let Some(&c) = self.b.get(self.i) else {
            return self.err("unexpected end");
        };
        match c {
            b'n' => self.lit(b"null", Value::Null),
            b't' => self.lit(b"true", Value::Bool(true)),
            b'f' => self.lit(b"false", Value::Bool(false)),
            b'"' => Ok(Value::Str(self.string()?)),
            b'[' => {
                self.enter()?;
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                } else {
                    loop {
                        a.push(self.value()?);
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b']') => {
                                self.i += 1;
                                break;
                            }
                            _ => return self.err("expected , or ]"),
                        }
                    }
                }
                self.depth -= 1;
                Ok(Value::Arr(a))
            }
            b'{' => {
                self.enter()?;
                self.i += 1;
                let mut m = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                } else {
                    loop {
                        self.ws();
                        if self.b.get(self.i) != Some(&b'"') {
                            return self.err("expected key");
                        }
                        let k = self.string()?;
                        self.ws();
                        if self.b.get(self.i) != Some(&b':') {
                            return self.err("expected :");
                        }
                        self.i += 1;
                        let v = self.value()?;
                        m.push((k, v));
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b'}') => {
                                self.i += 1;
                                break;
                            }
                            _ => return self.err("expected , or }"),
                        }
                    }
                }
                self.depth -= 1;
                Ok(Value::Obj(m))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => self.err("unexpected character"),
        }
    }
    fn enter(&mut self) -> Result<(), JsonError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.err("too deep")
        } else {
            Ok(())
        }
    }
    fn number(&mut self) -> Result<Value, JsonError> {
        let s = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let d0 = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        if self.i == d0 || (self.b[d0] == b'0' && self.i - d0 > 1) {
            return self.err("bad integer part");
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            let f0 = self.i;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == f0 {
                return self.err("bad fraction");
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            let e0 = self.i;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == e0 {
                return self.err("bad exponent");
            }
        }
        // Safe: only ASCII bytes were consumed.
        Ok(Value::Num(
            String::from_utf8_lossy(&self.b[s..self.i]).into_owned(),
        ))
    }
    fn hex4(&mut self) -> Result<u32, JsonError> {
        if self.i + 4 > self.b.len() {
            return self.err("short \\u escape");
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4])
            .map_err(|_| JsonError(self.i, "bad hex"))?;
        let v = u32::from_str_radix(s, 16).map_err(|_| JsonError(self.i, "bad hex"))?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, JsonError> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let s = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                if self.b[self.i] < 0x20 {
                    return self.err("control character in string");
                }
                self.i += 1;
            }
            // Input came from &str, and we only stop on ASCII, so this slice is UTF-8.
            out.push_str(
                std::str::from_utf8(&self.b[s..self.i]).map_err(|_| JsonError(s, "utf8"))?,
            );
            match self.b.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => {
                    self.i += 1;
                    let Some(&e) = self.b.get(self.i) else {
                        return self.err("bad escape");
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.b[self.i..].starts_with(b"\\u") {
                                    return self.err("lone high surrogate");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.err("bad low surrogate");
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(
                                char::from_u32(cp).ok_or(JsonError(self.i, "bad code point"))?,
                            );
                        }
                        _ => return self.err("bad escape"),
                    }
                }
            }
        }
    }
}

/// Append `s` as a JSON string in Python's `ensure_ascii=True` form.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) >= 0x7f => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", u);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Canonical encoding of a value: sorted keys, no spaces, ASCII only.
pub fn write_canonical(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(n) => out.push_str(n),
        Value::Str(s) => write_str(out, s),
        Value::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, x);
            }
            out.push(']');
        }
        Value::Obj(m) => {
            // Python sorts str keys by code point; UTF-8 byte order is the same order.
            let mut idx: Vec<usize> = (0..m.len()).collect();
            idx.sort_by(|&a, &b| m[a].0.cmp(&m[b].0));
            out.push('{');
            for (j, &i) in idx.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                write_str(out, &m[i].0);
                out.push(':');
                write_canonical(out, &m[i].1);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_canonical() {
        let v =
            parse(r#" {"b": [1, -2.5e3, "x\u00e9\ud83d\ude00"], "a": null, "c": true} "#).unwrap();
        let mut s = String::new();
        write_canonical(&mut s, &v);
        // Python: json.dumps(..., sort_keys=True, separators=(",", ":"))
        assert_eq!(
            s,
            r#"{"a":null,"b":[1,-2.5e3,"x\u00e9\ud83d\ude00"],"c":true}"#
        );
    }

    #[test]
    fn rejects_malformed() {
        for bad in [
            "",
            "{",
            "[1,]",
            "01",
            "\"\\x\"",
            "{\"a\" 1}",
            "1 2",
            "\"\u{1}\"",
            "\"\\ud800\"",
        ] {
            assert!(parse(bad).is_err(), "accepted {bad:?}");
        }
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(parse(&deep).is_err());
    }

    #[test]
    fn control_chars_escape_like_python() {
        let mut s = String::new();
        write_str(&mut s, "a\u{1}\u{7f}\"\\\n");
        assert_eq!(s, "\"a\\u0001\\u007f\\\"\\\\\\n\"");
    }
}
