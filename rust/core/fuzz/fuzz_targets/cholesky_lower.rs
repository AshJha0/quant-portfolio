#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| qp_core_fuzz::check_cholesky_lower(data));
