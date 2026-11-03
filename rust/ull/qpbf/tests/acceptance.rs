//! W30 decoder acceptance: bit-exact round trip with ledger equality, and
//! malformed input never panics.
//!
//! The plan's figure is 1e8 messages. The default run uses 2e6 so the suite
//! stays fast; set `QPBF_ROUNDTRIP_N=100000000` for the full figure (about a
//! minute in release on a laptop; the handback records the measured run).

use qp_ull_qpbf::generator::{GenConfig, generate_day};
use qp_ull_qpbf::{N_TYPES, Packet, PacketHeader, decode_msg};

fn n_from_env(default: u64) -> u64 {
    std::env::var("QPBF_ROUNDTRIP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
fn roundtrip_bit_exact_and_ledger_matches() {
    let n = n_from_env(2_000_000);
    let mut counts = [0u64; N_TYPES];
    let mut decoded = 0u64;
    let mut next_seq = 1u64;
    let mut re = vec![0u8; 2048];
    let ledger = generate_day(
        GenConfig {
            seed: 20261103,
            ..GenConfig::default()
        },
        7,
        n,
        |pkt| {
            let p = Packet::parse(pkt).expect("generated packet must parse");
            assert_eq!(p.header.seq, next_seq, "sequence gap");
            // Re-encode header and every message; bytes must match exactly.
            p.header.encode(&mut re);
            let mut off = qp_ull_qpbf::PACKET_HEADER_LEN;
            for (seq, m) in p.messages() {
                assert_eq!(seq, next_seq);
                next_seq += 1;
                counts[m.kind() as usize] += 1;
                off += m.encode(&mut re[off..]);
                decoded += 1;
            }
            assert_eq!(&re[..off], pkt, "re-encoded packet differs");
        },
    );
    assert_eq!(decoded, n);
    assert_eq!(counts, ledger);
    // Every type the stream draws from appears (snapshot is not in the
    // random mix).
    for (i, c) in counts.iter().enumerate() {
        if i != qp_ull_qpbf::MsgType::Snapshot as usize {
            assert!(*c > 0, "type {i} never generated");
        }
    }
}

/// Deterministic mutation fuzzing: flip, truncate, extend and splice bytes
/// of real packets; every call must return (Ok or Err) without panicking.
/// cargo-fuzz targets for coverage-guided runs are listed in the handback
/// as a CI follow-up; this test is the in-suite floor.
#[test]
fn malformed_packets_never_panic() {
    let mut pkts: Vec<Vec<u8>> = Vec::new();
    generate_day(GenConfig::default(), 1, 20_000, |p| pkts.push(p.to_vec()));
    let mut rng = qp_ull_qpbf::generator::Rng::new(99);
    let mut ok = 0u64;
    let mut err = 0u64;
    let iters = 300_000;
    for _ in 0..iters {
        let src = &pkts[rng.below(pkts.len() as u64) as usize];
        let mut p = src.clone();
        match rng.below(5) {
            0 => {
                for _ in 0..1 + rng.below(4) {
                    let i = rng.below(p.len() as u64) as usize;
                    p[i] = rng.next_u64() as u8;
                }
            }
            1 => p.truncate(rng.below(p.len() as u64) as usize),
            2 => {
                for _ in 0..rng.below(64) {
                    p.push(rng.next_u64() as u8);
                }
            }
            3 => {
                // Make the header self-consistent around random payload bytes,
                // so the message walker (not just the header check) is fuzzed.
                let body: Vec<u8> = (0..rng.below(600)).map(|_| rng.next_u64() as u8).collect();
                p.truncate(qp_ull_qpbf::PACKET_HEADER_LEN);
                p.extend_from_slice(&body);
                PacketHeader {
                    msg_count: rng.below(40) as u8,
                    session_id: 1,
                    seq: rng.next_u64(),
                }
                .encode(&mut p);
            }
            _ => {
                let i = rng.below(p.len() as u64) as usize;
                let _ = decode_msg(&p[i..]);
            }
        }
        match Packet::parse(&p) {
            Ok(pk) => {
                ok += 1;
                // Every accepted packet is canonical: it re-encodes to the
                // same bytes (spec section 12).
                let mut re = vec![0u8; p.len()];
                pk.header.encode(&mut re);
                let mut off = qp_ull_qpbf::PACKET_HEADER_LEN;
                for (_, m) in pk.messages() {
                    off += m.encode(&mut re[off..]);
                }
                assert_eq!(re, p, "accepted packet is not canonical");
            }
            Err(_) => err += 1,
        }
    }
    assert_eq!(ok + err, iters);
    assert!(
        err > iters / 2,
        "mutations should mostly be rejected: {err}"
    );
}
