#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| eq_var_engine_fuzz::check_cholesky_repair(data));
