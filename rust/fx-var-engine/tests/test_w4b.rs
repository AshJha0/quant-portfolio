//! W4(b) for the FX engine: scenario `s` on Philox stream `(seed, s)` with
//! AS 241 normals, the same bits on any thread count and the same bits as
//! the C++ engine (`cpp/fx-var-engine/tests/test_w4b.cpp` pins the same
//! words). The pinned words were produced by the C++ engine (GCC 15.2,
//! glibc, `-ffp-contract=off`); this test passing is the C++/Rust bit
//! parity check.

use fx_var_engine::McDist;
use fx_var_engine::matrix::Matrix;
use fx_var_engine::monte_carlo::{
    JumpSpec, simulate_factor_returns, var_standard_error_bootstrap,
    var_standard_error_bootstrap_with_threads,
};
use fx_var_engine::returns::FactorCov;

/// (dist, scenario, factor, f64 bits) from the C++ engine.
const PINNED: &[(u8, usize, usize, u64)] = &[
    (0, 0, 0, 0x3f5169b979aa098c),
    (0, 0, 1, 0xbf86b7306eb673e3),
    (0, 0, 2, 0xbf94594746d60c6f),
    (0, 1, 0, 0xbf7e2f36a9e33b5a),
    (0, 1, 1, 0x3f9194a0f89ac69d),
    (0, 1, 2, 0x3f6496dfb0aca743),
    (0, 2, 0, 0x3f8694c20ebe07d8),
    (0, 2, 1, 0x3f9fe2c5b6884ca4),
    (0, 2, 2, 0x3f74a1138da23b8c),
    (0, 3, 0, 0x3f9b762aff9934ed),
    (0, 3, 1, 0x3fa21d3c98b692d6),
    (0, 3, 2, 0x3f92644d20a3da6f),
    (1, 0, 0, 0x3f41418fa1bcf832),
    (1, 0, 1, 0xbf7682cb90d27836),
    (1, 0, 2, 0xbf842a57f4fbf190),
    (1, 1, 0, 0xbf76fa017cf7b336),
    (1, 1, 1, 0x3f8ac3d108973e7f),
    (1, 1, 2, 0x3f5f587299f61997),
    (1, 2, 0, 0x3f82d80f52840f28),
    (1, 2, 1, 0x3f9a9bdb9e5d2cdc),
    (1, 2, 2, 0x3f713712f2c36ebc),
    (1, 3, 0, 0x3f8d08ca0ab874b9),
    (1, 3, 1, 0x3f9326d0476157f7),
    (1, 3, 2, 0x3f8371f2b488df08),
    (2, 0, 0, 0x3f9dc2a3c8c18a12),
    (2, 0, 1, 0xbf86b7306eb673e3),
    (2, 0, 2, 0xbfb646e48b31f81e),
    (2, 1, 0, 0x3f81ad0050f1b64b),
    (2, 1, 1, 0x3f9194a0f89ac69d),
    (2, 1, 2, 0xbfa6453768d26736),
    (2, 2, 0, 0x3f97941b65944369),
    (2, 2, 1, 0x3f9fe2c5b6884ca4),
    (2, 2, 2, 0xbf9f4fc65fae9621),
    (2, 3, 0, 0x3fac111998600f33),
    (2, 3, 1, 0x3fa21d3c98b692d6),
    (2, 3, 2, 0xbfb08a4045a7296f),
];
const BOOT_SE_BITS: u64 = 0x402de47205bbe00b;

fn cov3() -> FactorCov {
    FactorCov {
        factors: vec!["FX:EUR".into(), "FX:GBP".into(), "FX:JPY".into()],
        cov: Matrix::from_rows(&[
            vec![3.6e-5, 1.1e-5, 0.4e-5],
            vec![1.1e-5, 4.9e-5, 0.9e-5],
            vec![0.4e-5, 0.9e-5, 2.5e-5],
        ])
        .unwrap(),
    }
}

fn jumps() -> JumpSpec {
    let mut j = JumpSpec {
        prob: 0.3,
        ..JumpSpec::default()
    };
    j.mean.insert("FX:JPY".into(), -0.01);
    j.mean.insert("FX:EUR".into(), 0.004);
    j.stdev.insert("FX:JPY".into(), 0.005);
    j.stdev.insert("FX:EUR".into(), 0.0);
    j
}

fn pnl300() -> Vec<f64> {
    (0..300)
        .map(|i| (1.7 * f64::from(i)).sin() * 1000.0 + f64::from(i))
        .collect()
}

#[test]
fn scenarios_match_the_cpp_engine_bit_for_bit() {
    for (code, dist) in [
        (0u8, McDist::Normal),
        (1, McDist::StudentT),
        (2, McDist::Jump),
    ] {
        let (r, _) = simulate_factor_returns(&cov3(), 4, dist, 5.0, &jumps(), 2026, 10.0).unwrap();
        for &(d, s, i, bits) in PINNED.iter().filter(|p| p.0 == code) {
            let got = r.data.get(s, i);
            assert_eq!(
                got.to_bits(),
                bits,
                "dist {d} scenario {s} factor {i}: {got:e}"
            );
        }
    }
}

#[test]
fn scenario_depends_only_on_its_index() {
    let (a, _) =
        simulate_factor_returns(&cov3(), 5000, McDist::Jump, 5.0, &jumps(), 7, 3.0).unwrap();
    let (b, _) = simulate_factor_returns(&cov3(), 4, McDist::Jump, 5.0, &jumps(), 7, 3.0).unwrap();
    for s in 0..4 {
        for i in 0..3 {
            assert_eq!(a.data.get(s, i).to_bits(), b.data.get(s, i).to_bits());
        }
    }
}

#[test]
fn bootstrap_se_matches_the_cpp_engine_bit_for_bit() {
    let se = var_standard_error_bootstrap(&pnl300(), 0.01, 200, 99).unwrap();
    assert_eq!(se.to_bits(), BOOT_SE_BITS, "{se}");
}

#[test]
fn bootstrap_se_is_the_same_bits_on_any_thread_count() {
    let pnl: Vec<f64> = (0..5000)
        .map(|i| (0.37 * f64::from(i)).sin() * 50.0)
        .collect();
    let one = var_standard_error_bootstrap_with_threads(&pnl, 0.01, 300, 5, 1).unwrap();
    for t in [2, 3, 4, 8] {
        let got = var_standard_error_bootstrap_with_threads(&pnl, 0.01, 300, 5, t).unwrap();
        assert_eq!(got.to_bits(), one.to_bits(), "threads {t}");
    }
}
