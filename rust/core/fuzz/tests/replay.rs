//! Stable-toolchain replay of the checked-in corpus and every regression
//! reproducer, so a fixed bug stays fixed without nightly or cargo-fuzz.
//! The C++ regression inputs are replayed too: both languages share the
//! byte-to-input mapping.

use std::fs;
use std::path::Path;

fn replay(dirs: &[&str], check: fn(&[u8])) -> usize {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut n = 0;
    for d in dirs {
        let dir = root.join(d);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            if e.path().is_file() {
                check(&fs::read(e.path()).unwrap());
                n += 1;
            }
        }
    }
    n
}

#[test]
fn implied_vol_corpus() {
    let n = replay(
        &[
            "corpus/implied_vol",
            "../../../cpp/core/fuzz/corpus/implied_vol",
            "../../../cpp/core/fuzz/regressions/implied_vol",
        ],
        qp_core_fuzz::check_implied_vol,
    );
    assert!(n >= 3, "expected the seed corpus, found {n} files");
}

#[test]
fn cholesky_corpus() {
    let n = replay(
        &[
            "corpus/cholesky_lower",
            "../../../cpp/core/fuzz/corpus/cholesky_repair",
        ],
        qp_core_fuzz::check_cholesky_lower,
    );
    assert!(n >= 2, "expected the seed corpus, found {n} files");
}

#[test]
fn short_and_empty_inputs() {
    for len in 0..80 {
        let bytes: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
        qp_core_fuzz::check_implied_vol(&bytes);
        qp_core_fuzz::check_cholesky_lower(&bytes);
    }
}
