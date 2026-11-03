//! `ull-bench`: per-call latency of the W30-rs hot path.
//!
//! Every figure printed is software in-process latency, not wire-to-wire.
//!
//! Usage:
//!   ull-bench [--msgs N] [--seed S] [--rate-mps R] [--host-class LABEL]
//!             [--input day.qpbf] [--write-day day.qpbf]
//!
//! `--input` reads a QPBF recording (`docs/QPBF.md` section 9: magic
//! `QPBFREC1`, then a u16 length and the bytes of each packet), so the Rust
//! and C++ harnesses can run on identical bytes. `--write-day` writes the
//! generated day in that format.
//! `--host-class` defaults to "shared-VM indicative" unless set; set it to
//! "dedicated" only on a host you control and have pinned.

use qp_ull_bench::{Histogram, RealClock, env, run_fixed_rate, timer_overhead};
use qp_ull_book::Book;
use qp_ull_qpbf::generator::{GenConfig, generate_day};
use qp_ull_qpbf::{
    Msg, PACKET_HEADER_LEN, Packet, decode_msg, record_packet, recording_header, split_recording,
};
use std::fmt::Write as _;

struct Args {
    msgs: u64,
    seed: u64,
    rate_mps: f64,
    host_class: String,
    input: Option<String>,
    write_day: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        msgs: 2_000_000,
        seed: 20261103,
        rate_mps: 0.2,
        host_class: "shared-VM indicative".into(),
        input: None,
        write_day: None,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--msgs" => a.msgs = val.parse().expect("--msgs"),
            "--seed" => a.seed = val.parse().expect("--seed"),
            "--rate-mps" => a.rate_mps = val.parse().expect("--rate-mps"),
            "--host-class" => a.host_class = val,
            "--input" => a.input = Some(val),
            "--write-day" => a.write_day = Some(val),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    a
}

fn summary(name: &str, unit_note: &str, h: &Histogram, out: &mut String) {
    let _ = writeln!(
        out,
        "  {{\"name\": \"{name}\", \"note\": \"{unit_note}\", \"n\": {}, \"mean_ns\": {:.1}, \"p50_ns\": {}, \"p99_ns\": {}, \"p99_9_ns\": {}, \"p99_99_ns\": {}, \"max_ns\": {}}},",
        h.count(),
        h.mean(),
        h.percentile(50.0),
        h.percentile(99.0),
        h.percentile(99.9),
        h.percentile(99.99),
        h.max()
    );
}

fn main() {
    let a = parse_args();
    let owned: Vec<u8>;
    let packets: Vec<&[u8]> = if let Some(p) = &a.input {
        owned = std::fs::read(p).expect("read --input");
        split_recording(&owned).expect("--input is not a QPBF recording")
    } else {
        let mut buf = Vec::new();
        recording_header(&mut buf);
        generate_day(
            GenConfig {
                seed: a.seed,
                ..GenConfig::default()
            },
            1,
            a.msgs,
            |p| record_packet(&mut buf, p),
        );
        if let Some(w) = &a.write_day {
            std::fs::write(w, &buf).expect("write --write-day");
        }
        owned = buf;
        split_recording(&owned).expect("generated recording")
    };
    for p in &packets {
        Packet::parse(p).expect("day file has an invalid packet");
    }
    let n_msgs: u64 = packets
        .iter()
        .map(|p| Packet::parse(p).unwrap().header.msg_count as u64)
        .sum();

    let clk = RealClock::calibrate(200);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{{\"label\": \"software in-process latency, not wire-to-wire\","
    );
    let _ = writeln!(out, " \"host_class\": \"{}\",", a.host_class);
    let _ = writeln!(
        out,
        " \"timer\": \"{}\", \"ns_per_tick\": {:.6},",
        if clk.uses_tsc {
            "lfence;rdtsc;lfence .. rdtscp;lfence"
        } else {
            "Instant"
        },
        clk.ns_per_tick
    );
    let _ = writeln!(
        out,
        " \"messages\": {n_msgs}, \"packets\": {},",
        packets.len()
    );
    let _ = writeln!(out, " \"env\": {{");
    for (k, v) in env::collect() {
        let _ = writeln!(out, "  \"{k}\": \"{}\",", v.replace('"', "'"));
    }
    let _ = writeln!(out, "  \"_\": \"\"}},\n \"results\": [");

    summary(
        "timer_overhead",
        "empty timed region",
        &timer_overhead(&clk, 1_000_000),
        &mut out,
    );

    // Decode only: one decode_msg call per message.
    let mut h = Histogram::new();
    let mut sink = 0u64;
    for p in &packets {
        let mut off = PACKET_HEADER_LEN;
        while off < p.len() {
            let s = clk.start();
            let (m, len) = decode_msg(&p[off..]).unwrap();
            let e = clk.end();
            sink = sink.wrapping_add(m.instrument() as u64);
            h.record(e.saturating_sub(s));
            off += len;
        }
    }
    summary(
        "qpbf_decode_msg",
        "per message, includes timer overhead",
        &h,
        &mut out,
    );

    // Decode + book apply per message (tick-to-book).
    let mid = GenConfig::default().start_mid;
    let n_books = packets
        .iter()
        .flat_map(|p| Packet::parse(p).unwrap().messages())
        .map(|(_, m)| m.instrument() as usize + 1)
        .max()
        .unwrap_or(1);
    let new_books = || -> Vec<Book> {
        (0..n_books)
            .map(|_| Book::new(mid - 30_000, 60_000, 1 << 16))
            .collect()
    };
    let mut books = new_books();
    let mut h = Histogram::new();
    for p in &packets {
        let mut off = PACKET_HEADER_LEN;
        while off < p.len() {
            let s = clk.start();
            let (m, len) = decode_msg(&p[off..]).unwrap();
            let r = books[m.instrument() as usize].apply(&m); // index 0 = heartbeats, never touched
            let e = clk.end();
            debug_assert!(r.is_ok());
            h.record(e.saturating_sub(s));
            off += len;
        }
    }
    summary(
        "decode_plus_book_apply",
        "per message, includes timer overhead",
        &h,
        &mut out,
    );

    // Fixed-rate, coordinated-omission-aware decode+apply.
    let all: Vec<Msg> = packets
        .iter()
        .flat_map(|p| Packet::parse(p).unwrap().messages().map(|x| x.1))
        .collect();
    let mut books = new_books();
    let period = (1_000.0 / a.rate_mps) as u64;
    let n = (all.len() as u64).min(1_000_000);
    let mut rc = RealClock::calibrate(50);
    let r = run_fixed_rate(&mut rc, n, period.max(1), |i, _| {
        let m = &all[i as usize];
        let _ = books[m.instrument() as usize].apply(m);
    });
    summary(
        "book_apply_fixed_rate_co_aware",
        &format!("{} M msg/s offered; end minus intended start", a.rate_mps),
        &r.co,
        &mut out,
    );
    summary(
        "book_apply_fixed_rate_naive",
        "same run; end minus actual start (closed-loop view)",
        &r.naive,
        &mut out,
    );

    // If the median CO-aware latency is far above the naive tail, calls were
    // queueing behind each other for most of the run: the offered rate is
    // above this host's capacity and the CO figures measure the backlog.
    let saturated = r.co.percentile(50.0) > 10 * r.naive.percentile(99.0).max(1);
    let _ = writeln!(
        out,
        "  {{\"name\": \"fixed_rate_saturated\", \"value\": {saturated}, \"note\": \"true means offered load exceeded capacity; lower --rate-mps\"}},"
    );

    // SPSC ring round trip between two threads (not pinned unless the
    // operator used taskset; see env.cpus_allowed_list).
    let (mut p_tx, mut c_rx) = qp_ull_ring::channel::<u64>(1024);
    let (mut b_tx, mut b_rx) = qp_ull_ring::channel::<u64>(1024);
    let iters = 200_000u64;
    let echo = std::thread::spawn(move || {
        for _ in 0..iters {
            let v = loop {
                if let Some(v) = c_rx.pop() {
                    break v;
                }
                std::hint::spin_loop();
            };
            while b_tx.push(v).is_err() {
                std::hint::spin_loop();
            }
        }
    });
    let mut h = Histogram::new();
    for i in 0..iters {
        let s = clk.start();
        while p_tx.push(i).is_err() {
            std::hint::spin_loop();
        }
        loop {
            if b_rx.pop().is_some() {
                break;
            }
            std::hint::spin_loop();
        }
        h.record(clk.end().saturating_sub(s));
    }
    echo.join().unwrap();
    summary(
        "spsc_ring_round_trip",
        "two hops across threads; halve for one-way only if cores are symmetric",
        &h,
        &mut out,
    );
    let _ = writeln!(out, "  {{\"name\": \"_sink\", \"value\": {sink}}}\n ]\n}}");
    print!("{out}");
}
