//! Per-call latency benchmark for the FX options engine.
//!
//! Every kernel is timed with the method shared with the C++ benches
//! (`qp_core::bench`, docs/BENCHMARKS.md): a calibrated batch of
//! back-to-back calls per sample (at least 2 us), median with p10/p90,
//! over a rotating set of varied inputs. Results print as JSON and are written to `bench/results.json`
//! (or the path given with `--out <path>`).
//!
//! Run: `cargo run --release --bin bench`. Set `BENCH_FILTER=<substring>`
//! to time only the kernels whose name contains it (results.json is then
//! partial).

use std::hint::black_box;
use std::time::{SystemTime, UNIX_EPOCH};

use fx_options_engine::{
    DeltaConvention, Exercise, OptionType, analytic_greeks, binomial_price, gk_price, implied_vol,
    mc_price, strike_from_delta,
};
// FAST-BEGIN
use fx_options_engine::fast::{
    BinomialWorkspace, GkBatch, GreeksBatchOut, McWorkspace, binomial_price_ws, gk_price_batch,
    gk_price_k, greeks_batch, greeks_k, implied_vol_k, mc_price_par, mc_price_ws,
};
// FAST-END

const NIN: usize = 1024;

struct Inputs {
    s: Vec<f64>,
    k: Vec<f64>,
    t: Vec<f64>,
    rd: Vec<f64>,
    rf: Vec<f64>,
    v: Vec<f64>,
    ty: Vec<OptionType>,
    px: Vec<f64>,
}

fn inputs() -> Inputs {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut u = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut i = Inputs {
        s: vec![],
        k: vec![],
        t: vec![],
        rd: vec![],
        rf: vec![],
        v: vec![],
        ty: vec![],
        px: vec![],
    };
    for j in 0..NIN {
        let s = 1.0 + 0.3 * u();
        let k = s * (0.8 + 0.4 * u());
        let t = 0.02 + 2.0 * u();
        let rd = 0.05 * u();
        let rf = 0.05 * u();
        let v = 0.05 + 0.25 * u();
        let ty = if j % 2 == 0 {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let px = gk_price(s, k, t, rd, rf, v, ty).unwrap_or(0.0);
        i.s.push(s);
        i.k.push(k);
        i.t.push(t);
        i.rd.push(rd);
        i.rf.push(rf);
        i.v.push(v);
        i.ty.push(ty);
        i.px.push(px);
    }
    i
}

struct Row {
    kernel: &'static str,
    n_iter: usize,
    calls_per_sample: usize,
    /// p10, p50, p90, p99, p99.9, max (ns per call).
    q: [f64; 6],
}

/// Time `f(idx)` with the shared method (`qp_core::bench`). `warm` is in
/// calls and `samples * per` is the call budget; the batch size itself is
/// calibrated. `idx` cycles over the `NIN` inputs.
fn measure(
    kernel: &'static str,
    warm: usize,
    samples: usize,
    per: usize,
    mut f: impl FnMut(usize),
) -> Row {
    if std::env::var("BENCH_FILTER").is_ok_and(|p| !kernel.contains(p.as_str())) {
        return Row {
            kernel,
            n_iter: 0,
            calls_per_sample: per,
            q: [0.0; 6],
        };
    }
    let s = qp_core::bench::measure(warm, samples * per, |i| f(i % NIN));
    Row {
        kernel,
        n_iter: s.samples,
        calls_per_sample: s.calls_per_sample,
        q: [s.p10, s.p50, s.p90, s.p99, s.p999, s.max],
    }
}

fn utc_date() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64;
    // Civil-from-days (H. Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

fn cpu() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|c| {
            c.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn toolchain() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".into(), |s| s.trim().to_string())
}

fn main() {
    let inp = inputs();
    let mut rows = Vec::new();
    let mut sink = 0.0f64;

    rows.push(measure("gk_price", 100_000, 200_000, 64, |j| {
        sink += gk_price(
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            inp.ty[j],
        )
        .unwrap_or(0.0);
    }));
    rows.push(measure("analytic_greeks", 100_000, 100_000, 64, |j| {
        sink += analytic_greeks(
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            inp.ty[j],
        )
        .map_or(0.0, |g| g.vega);
    }));
    rows.push(measure("implied_vol", 20_000, 50_000, 16, |j| {
        sink += implied_vol(
            black_box(inp.px[j]),
            inp.s[j],
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.ty[j],
        )
        .unwrap_or(0.0);
    }));
    rows.push(measure(
        "strike_from_delta_spot_pa",
        5_000,
        20_000,
        4,
        |j| {
            let d = 0.05 + 0.4 * (j % 100) as f64 / 100.0;
            sink += strike_from_delta(
                black_box(d),
                inp.s[j],
                inp.t[j],
                inp.rd[j],
                inp.rf[j],
                inp.v[j],
                OptionType::Call,
                DeltaConvention::SpotPa,
            )
            .unwrap_or(0.0);
        },
    ));
    rows.push(measure("binomial_price_american_500", 200, 3_000, 1, |j| {
        sink += binomial_price(
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            OptionType::Put,
            500,
            Exercise::American,
        )
        .unwrap_or(0.0);
    }));
    rows.push(measure("mc_price_100k_anti_cv", 5, 200, 1, |j| {
        sink += mc_price(
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            inp.ty[j],
            100_000,
            j as u64,
            true,
            true,
        )
        .map_or(0.0, |r| r.price);
    }));

    // FAST-BEGIN
    rows.push(measure("fast::gk_price_k", 100_000, 200_000, 64, |j| {
        sink += gk_price_k(
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            inp.ty[j],
        )
        .unwrap_or(0.0);
    }));
    rows.push(measure("fast::greeks_k", 100_000, 100_000, 64, |j| {
        sink += greeks_k(
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            inp.ty[j],
        )
        .map_or(0.0, |g| g.vega);
    }));
    rows.push(measure("fast::implied_vol_k", 20_000, 50_000, 16, |j| {
        sink += implied_vol_k(
            black_box(inp.px[j]),
            inp.s[j],
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.ty[j],
        )
        .unwrap_or(0.0);
    }));
    {
        let b = GkBatch {
            s: &inp.s,
            k: &inp.k,
            t: &inp.t,
            r_d: &inp.rd,
            r_f: &inp.rf,
            sigma: &inp.v,
            option_type: &inp.ty,
        };
        let mut out = vec![0.0; NIN];
        // One sample = one 1024-element batch; reported per element.
        let mut r = measure("fast::gk_price_batch_per_option", 200, 20_000, 1, |_| {
            let _ = gk_price_batch(black_box(&b), &mut out);
            sink += out[0];
        });
        for q in &mut r.q {
            *q /= NIN as f64;
        }
        r.calls_per_sample *= NIN;
        rows.push(r);
        let mut cols = vec![vec![0.0; NIN]; 10];
        let mut r = measure("fast::greeks_batch_per_option", 200, 10_000, 1, |_| {
            if let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] = &mut cols[..] {
                let mut o = GreeksBatchOut {
                    price: a0,
                    delta_spot: a1,
                    delta_forward: a2,
                    gamma: a3,
                    vega: a4,
                    theta: a5,
                    rho_domestic: a6,
                    rho_foreign: a7,
                    vanna: a8,
                    volga: a9,
                };
                let _ = greeks_batch(black_box(&b), &mut o);
                sink += o.vega[0];
            }
        });
        for q in &mut r.q {
            *q /= NIN as f64;
        }
        r.calls_per_sample *= NIN;
        rows.push(r);
    }
    let mut bws = BinomialWorkspace::with_steps(500);
    rows.push(measure(
        "fast::binomial_price_ws_american_500",
        200,
        3_000,
        1,
        |j| {
            sink += binomial_price_ws(
                &mut bws,
                black_box(inp.s[j]),
                inp.k[j],
                inp.t[j],
                inp.rd[j],
                inp.rf[j],
                inp.v[j],
                OptionType::Put,
                500,
                Exercise::American,
            )
            .unwrap_or(0.0);
        },
    ));
    let mut mws = McWorkspace::with_paths(1_000_000);
    rows.push(measure("fast::mc_price_ws_100k_anti_cv", 5, 200, 1, |j| {
        sink += mc_price_ws(
            &mut mws,
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            inp.ty[j],
            100_000,
            j as u64,
            true,
            true,
        )
        .map_or(0.0, |r| r.price);
    }));
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    rows.push(measure(
        "fast::mc_price_par_1m_anti_cv_all_threads",
        3,
        60,
        1,
        |j| {
            sink += mc_price_par(
                &mut mws,
                black_box(inp.s[j]),
                inp.k[j],
                inp.t[j],
                inp.rd[j],
                inp.rf[j],
                inp.v[j],
                inp.ty[j],
                1_000_000,
                j as u64,
                true,
                true,
                threads,
            )
            .map_or(0.0, |r| r.price);
        },
    ));
    // FAST-END
    rows.push(measure("mc_price_1m_anti_cv", 2, 30, 1, |j| {
        sink += mc_price(
            black_box(inp.s[j]),
            inp.k[j],
            inp.t[j],
            inp.rd[j],
            inp.rf[j],
            inp.v[j],
            inp.ty[j],
            1_000_000,
            j as u64,
            true,
            true,
        )
        .map_or(0.0, |r| r.price);
    }));
    black_box(sink);

    let (tc, cpu, date) = (toolchain(), cpu(), utc_date());
    let flags = format!(
        "release profile opt-level=3 lto=fat codegen-units=1; RUSTFLAGS={}",
        std::env::var("RUSTFLAGS").unwrap_or_default()
    );
    rows.retain(|r| r.n_iter > 0);
    let mut json = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        json.push_str(&format!(
            "  {{\"engine\": \"rust/fx-options-engine\", \"kernel\": \"{}\", \"toolchain\": \"{tc}\", \"flags\": \"{flags}\", \"cpu\": \"{cpu} (WSL2)\", \"date\": \"{date}\", \"n_iter\": {}, \"calls_per_sample\": {}, \"unit\": \"ns\", \"p10\": {:.1}, \"p50\": {:.1}, \"p90\": {:.1}, \"p99\": {:.1}, \"p999\": {:.1}, \"max\": {:.1}}}{}\n",
            r.kernel, r.n_iter, r.calls_per_sample, r.q[0], r.q[1], r.q[2], r.q[3], r.q[4], r.q[5],
            if i + 1 < rows.len() { "," } else { "" }
        ));
    }
    json.push_str("]\n");
    print!("{json}");
    write_results(&out_path(), &json);
}

/// Where to write the JSON: `--out <path>` (or `--out=<path>`) if given,
/// otherwise `bench/results.json` inside this crate.
fn out_path() -> std::path::PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--out" {
            if let Some(p) = args.next() {
                return p.into();
            }
        } else if let Some(p) = a.strip_prefix("--out=") {
            return p.into();
        }
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("bench")
        .join("results.json")
}

/// Write `json` to `path`, creating the parent directory.
fn write_results(path: &std::path::Path, json: &str) {
    let written = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => std::fs::create_dir_all(dir),
        _ => Ok(()),
    }
    .and_then(|()| std::fs::write(path, json));
    match written {
        Ok(()) => eprintln!("wrote {}", path.display()),
        Err(e) => eprintln!("could not write {}: {e}", path.display()),
    }
}
