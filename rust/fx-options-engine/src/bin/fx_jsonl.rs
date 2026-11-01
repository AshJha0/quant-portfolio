//! JSON-lines shim over the engine for the cross-language differential
//! test (see `python/fx/01-options-pricing/tests/test_differential.py`).
//!
//! One flat JSON object per stdin line (numbers, strings, bools); one
//! response per stdout line: `{"id":..,"ok":true,"out":{..}}` or
//! `{"id":..,"ok":false,"err":".."}`. Functions: `gk`, `iv`, `strike`, `dns_atm`, `crr`, `fwd`.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use fx_options_engine::{
    analytic_greeks, atm_dns_strike, binomial_price, cip_forward, delta, implied_vol,
    strike_from_delta, DeltaConvention, Exercise, OptionType,
};

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

fn convention(c: &str) -> Result<DeltaConvention, String> {
    match c {
        "spot" => Ok(DeltaConvention::Spot),
        "forward" => Ok(DeltaConvention::Forward),
        "spot_pa" => Ok(DeltaConvention::SpotPa),
        "forward_pa" => Ok(DeltaConvention::ForwardPa),
        other => Err(format!("unknown convention: {other}")),
    }
}

fn handle(r: &Req) -> Result<Vec<(&'static str, f64)>, String> {
    let e = |err: fx_options_engine::FxError| err.to_string();
    match field(r, "fn")? {
        "gk" => {
            let (s, k, t, rd, rf, sig) = (
                num(r, "S")?,
                num(r, "K")?,
                num(r, "T")?,
                num(r, "rd")?,
                num(r, "rf")?,
                num(r, "sigma")?,
            );
            let ot = option_type(r)?;
            let g = analytic_greeks(s, k, t, rd, rf, sig, ot).map_err(e)?;
            Ok(vec![
                ("price", g.price),
                ("delta_spot", g.delta_spot),
                ("delta_fwd", g.delta_forward),
                (
                    "delta_spot_pa",
                    delta(s, k, t, rd, rf, sig, ot, DeltaConvention::SpotPa).map_err(e)?,
                ),
                (
                    "delta_fwd_pa",
                    delta(s, k, t, rd, rf, sig, ot, DeltaConvention::ForwardPa).map_err(e)?,
                ),
                ("gamma", g.gamma),
                ("vega", g.vega),
                ("theta", g.theta),
                ("rho_d", g.rho_domestic),
                ("rho_f", g.rho_foreign),
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
                num(r, "rd")?,
                num(r, "rf")?,
                option_type(r)?,
            )
            .map_err(e)?,
        )]),
        "strike" => Ok(vec![(
            "K",
            strike_from_delta(
                num(r, "delta")?,
                num(r, "S")?,
                num(r, "T")?,
                num(r, "rd")?,
                num(r, "rf")?,
                num(r, "sigma")?,
                option_type(r)?,
                convention(field(r, "convention")?)?,
            )
            .map_err(e)?,
        )]),
        "dns_atm" => {
            let conv = if field(r, "pa")? == "true" {
                DeltaConvention::SpotPa
            } else {
                DeltaConvention::Spot
            };
            Ok(vec![(
                "K",
                atm_dns_strike(
                    num(r, "S")?,
                    num(r, "T")?,
                    num(r, "rd")?,
                    num(r, "rf")?,
                    num(r, "sigma")?,
                    conv,
                )
                .map_err(e)?,
            )])
        }
        "crr" => {
            let ex = if field(r, "american")? == "true" {
                Exercise::American
            } else {
                Exercise::European
            };
            let steps = num(r, "steps")? as u32;
            Ok(vec![(
                "price",
                binomial_price(
                    num(r, "S")?,
                    num(r, "K")?,
                    num(r, "T")?,
                    num(r, "rd")?,
                    num(r, "rf")?,
                    num(r, "sigma")?,
                    option_type(r)?,
                    steps,
                    ex,
                )
                .map_err(e)?,
            )])
        }
        "fwd" => Ok(vec![(
            "forward",
            cip_forward(num(r, "S")?, num(r, "T")?, num(r, "rd")?, num(r, "rf")?).map_err(e)?,
        )]),
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
