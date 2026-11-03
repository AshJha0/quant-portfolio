//! Stable replay of the shared C++ Cholesky corpus plus a sweep of short
//! inputs through the Rust PSD-repair property.

use std::fs;
use std::path::Path;

#[test]
fn cholesky_repair_corpus() {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../cpp/core/fuzz/corpus/cholesky_repair");
    let mut n = 0;
    for e in fs::read_dir(dir).unwrap().flatten() {
        eq_var_engine_fuzz::check_cholesky_repair(&fs::read(e.path()).unwrap());
        n += 1;
    }
    assert!(n >= 2);
}

#[test]
fn byte_sweep() {
    // 3 repair kinds x 6 sizes x a spread of correlation patterns.
    for seed in 0u64..600 {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&seed.to_le_bytes());
        bytes.extend_from_slice(&(seed / 6).to_le_bytes());
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for _ in 0..21 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        eq_var_engine_fuzz::check_cholesky_repair(&bytes);
    }
}
