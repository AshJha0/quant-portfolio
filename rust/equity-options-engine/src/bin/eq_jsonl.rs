//! JSON-lines shim over the engine for the cross-language differential
//! test (see `python/equity/01-options-pricing/tests/test_differential.py`).
//!
//! One flat JSON object per stdin line (numbers, strings, bools); one
//! response per stdout line: `{"id":..,"ok":true,"out":{..}}` or
//! `{"id":..,"ok":false,"err":".."}`. Functions: `bs`, `iv`, `crr`, `b76`.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use eq_options_engine::{Exercise, OptionType, black76_greeks, bs_greeks, crr_price, implied_vol};

type Req = HashMap<String, String>;

fn parse(line: &str) -> Result<Req, String> {
    let s = line.trim();
    let inner = s
        .strip_prefix('{')
        .and_then(|t| t.strip_suffix('}'))
        .ok_or("expected a flat JSON object")?;
    let mut out = HashMap::new();
    let chars: Vec<char> = inner.chars().collect();
    let mut i = 0;
    let read_str = |i: &mut usize| -> Result<String, String> {
        if chars.get(*i) != Some(&'"') {
            return Err("expected string".into());
        }
        *i += 1;
        let mut v = String::new();
        while *i < chars.len() && chars[*i] != '"' {
            if chars[*i] == '\\' {
                *i += 1;
            }
            if let Some(&c) = chars.get(*i) {
                v.push(c);
            }
            *i += 1;
        }
        *i += 1;
        Ok(v)
    };
    while i < chars.len() {
        while i < chars.len() && (chars[i].is_whitespace() || chars[i] == ',') {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        let key = read_str(&mut i)?;
        while i < chars.len() && (chars[i].is_whitespace() || chars[i] == ':') {
            i += 1;
        }
        let value = if chars.get(i) == Some(&'"') {
            read_str(&mut i)?
        } else {
            let mut v = String::new();
            while i < chars.len() && chars[i] != ',' && !chars[i].is_whitespace() {
                v.push(chars[i]);
                i += 1;
            }
            v
        };
        out.insert(key, value);
    }
    Ok(out)
}

fn field<'a>(r: &'a Req, k: &str) -> Result<&'a str, String> {
    r.get(k)
        .map(String::as_str)
        .ok_or(format!("missing field: {k}"))
}

fn num(r: &Req, k: &str) -> Result<f64, String> {
    field(r, k)?
        .parse::<f64>()
        .map_err(|_| format!("not a number: {k}"))
}

fn option_type(r: &Req) -> Result<OptionType, String> {
    match field(r, "type")? {
        "call" => Ok(OptionType::Call),
        "put" => Ok(OptionType::Put),
        other => Err(format!("type must be call or put, got {other}")),
    }
}

fn number(v: f64) -> String {
    if v.is_nan() {
        "\"nan\"".into()
    } else if v.is_infinite() {
        if v > 0.0 {
            "\"inf\"".into()
        } else {
            "\"-inf\"".into()
        }
    } else {
        format!("{v:?}")
    }
}

fn handle(r: &Req) -> Result<Vec<(&'static str, f64)>, String> {
    let e = |err: eq_options_engine::PricingError| err.to_string();
    match field(r, "fn")? {
        "bs" => {
            let g = bs_greeks(
                num(r, "S")?,
                num(r, "K")?,
                num(r, "T")?,
                num(r, "r")?,
                num(r, "sigma")?,
                num(r, "q")?,
                option_type(r)?,
            )
            .map_err(e)?;
            Ok(vec![
                ("price", g.price),
                ("delta", g.delta),
                ("gamma", g.gamma),
                ("vega", g.vega),
                ("theta", g.theta),
                ("rho", g.rho),
                ("vanna", g.vanna),
                ("volga", g.volga),
            ])
        }
        "iv" => Ok(vec![(
            "sigma",
            implied_vol(
                num(r, "price")?,
                num(r, "S")?,
                num(r, "K")?,
                num(r, "T")?,
                num(r, "r")?,
                num(r, "q")?,
                option_type(r)?,
            )
            .map_err(e)?,
        )]),
        "crr" => {
            let ex = if field(r, "american")? == "true" {
                Exercise::American
            } else {
                Exercise::European
            };
            let steps = num(r, "steps")? as usize;
            Ok(vec![(
                "price",
                crr_price(
                    num(r, "S")?,
                    num(r, "K")?,
                    num(r, "T")?,
                    num(r, "r")?,
                    num(r, "sigma")?,
                    num(r, "q")?,
                    option_type(r)?,
                    ex,
                    steps,
                )
                .map_err(e)?,
            )])
        }
        "b76" => {
            let g = black76_greeks(
                num(r, "F")?,
                num(r, "K")?,
                num(r, "T")?,
                num(r, "r")?,
                num(r, "sigma")?,
                option_type(r)?,
            )
            .map_err(e)?;
            Ok(vec![
                ("price", g.price),
                ("delta", g.delta),
                ("gamma", g.gamma),
                ("vega", g.vega),
                ("theta", g.theta),
                ("rho", g.rho),
            ])
        }
        other => Err(format!("unknown fn: {other}")),
    }
}

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let (id, result) = match parse(&line) {
            Ok(r) => (
                r.get("id").cloned().unwrap_or_else(|| "null".into()),
                handle(&r),
            ),
            Err(msg) => ("null".into(), Err(msg)),
        };
        let text = match result {
            Ok(vals) => {
                let body: Vec<String> = vals
                    .iter()
                    .map(|(k, v)| format!("\"{k}\":{}", number(*v)))
                    .collect();
                format!("{{\"id\":{id},\"ok\":true,\"out\":{{{}}}}}", body.join(","))
            }
            Err(msg) => format!(
                "{{\"id\":{id},\"ok\":false,\"err\":\"{}\"}}",
                msg.replace('\\', "\\\\").replace('"', "\\\"")
            ),
        };
        let _ = writeln!(out, "{text}");
        let _ = out.flush();
    }
}
