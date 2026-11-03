//! Throughput check for `qp_core::vmath` against the scalar Cody `erfc`
//! and `f64::exp`. Run: `cargo run --release --example vmath_speed`
//! (add `RUSTFLAGS="-C target-cpu=native"` for the AVX2 figure).
//! Prints the median ns per element over 31 runs of 100,000 elements.
use std::hint::black_box;
use std::time::Instant;

fn median_ns(f: &mut dyn FnMut()) -> f64 {
    let n = 100_000.0;
    let mut v: Vec<f64> = (0..31)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64 / n
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[15]
}

fn main() {
    let x: Vec<f64> = (0..100_000)
        .map(|i| -6.0 + 12.0 * i as f64 / 100_000.0)
        .collect();
    let mut y = vec![0.0; x.len()];
    let e_b = median_ns(&mut || qp_core::vmath::erfc_batch(black_box(&x), black_box(&mut y)));
    let e_c = median_ns(&mut || {
        for (o, &v) in y.iter_mut().zip(black_box(&x)) {
            *o = qp_core::special::erfc(v);
        }
    });
    let x_b = median_ns(&mut || qp_core::vmath::exp_batch(black_box(&x), black_box(&mut y)));
    let x_s = median_ns(&mut || {
        for (o, &v) in y.iter_mut().zip(black_box(&x)) {
            *o = v.exp();
        }
    });
    let n_b = median_ns(&mut || qp_core::vmath::norm_cdf_batch(black_box(&x), black_box(&mut y)));
    println!("erfc_batch {e_b:.2} ns  cody erfc {e_c:.2} ns");
    println!("exp_batch  {x_b:.2} ns  std exp   {x_s:.2} ns");
    println!("norm_cdf_batch {n_b:.2} ns");
}
