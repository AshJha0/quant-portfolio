//! Deterministic, zero-dependency random number generation.
//!
//! The generator lives in `qp_core::rng` and is re-exported here, so
//! `eq_options_engine::rng::Xoshiro256PlusPlus` keeps working.
//!
//! # Algorithm choice (documented per the engine conventions)
//!
//! * **State generator: xoshiro256++** (Blackman & Vigna, 2019). 256-bit
//!   state, period 2^256 - 1, passes BigCrush, ~1 ns per `u64`. (NumPy's
//!   default `Generator` uses PCG64, a different family.) Chosen over
//!   `std`-only alternatives because Rust's standard library ships **no**
//!   RNG at all, and over a bare LCG/PCG32 because 64-bit output with a
//!   large state gives comfortable headroom for 10^6-10^8-path Monte
//!   Carlo without detectable correlation artefacts.
//! * **Seeding: SplitMix64** (Steele, Lea & Flood, 2014). A single `u64`
//!   seed is expanded through four SplitMix64 steps into the 256-bit
//!   xoshiro state. This is the seeding procedure recommended by the
//!   xoshiro authors: it guarantees a non-zero, well-mixed state for
//!   *every* seed including 0, and makes nearby seeds statistically
//!   independent.
//! * **Normal sampling: Box-Muller (trigonometric form)**, caching the
//!   second variate. Chosen over Ziggurat (large tables, harder to audit)
//!   and over inverse-CDF (needs a high-degree rational approximation
//!   whose tail accuracy would dominate the error budget). Box-Muller is
//!   exact in distribution, branch-free per pair, and trivially
//!   reproducible: consuming exactly two `u64` draws per pair makes the
//!   stream layout easy to reason about for bit-reproducibility tests.
//!
//! Same seed => identical `u64` stream on every platform. The normal
//! deviates go through `ln`/`sin`/`cos` from the platform libm, which is
//! deterministic on a given target but not correctly rounded, so normals
//! can differ in the last bits between targets.

pub use qp_core::rng::Xoshiro256PlusPlus;
pub(crate) use qp_core::rng::{box_muller, uniform_from_bits};
