//! Parity of `qp_core::exotics` with the Python pricers (D1), 1e-12 relative.
//! Golden lines come from `python/core/tests/gen_exotics_golden.py`.

use qp_core::exotics::{
    Barrier, Cp, Touch, barrier_price, double_knockout_price, double_no_touch,
    essvi_total_variance, geometric_asian_price, svi_total_variance, touch_price,
    turnbull_wakeman_continuous, turnbull_wakeman_price, vanilla_price,
};

fn cp(s: &str) -> Cp {
    match s {
        "call" => Cp::Call,
        "put" => Cp::Put,
        _ => panic!("bad cp {s}"),
    }
}

fn close(got: f64, want: f64, line: &str) {
    // 1e-12 relative, floored at 1e-11 absolute: on a 100 notional some prices are
    // small differences of O(10) terms, where the last-ulp noise is ~1e-14.
    let tol = (1e-12 * want.abs()).max(1e-11);
    assert!(
        (got - want).abs() <= tol,
        "{line}: got {got:e}, want {want:e}, diff {:e}",
        got - want
    );
}

#[test]
fn exotics_match_python_golden() {
    let text = include_str!("data/exotics_golden.txt");
    let fix: Vec<f64> = (1..=12).map(|i| f64::from(i) / 12.0).collect();
    let mut count = 0;
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let w: Vec<&str> = line.split_whitespace().collect();
        let f = |i: usize| -> f64 { w[i].parse().unwrap() };
        let want: f64 = w.last().unwrap().parse().unwrap();
        let got = match w[0] {
            "vanilla" => vanilla_price(cp(w[1]), f(2), f(3), f(4), f(5), f(6), f(7)),
            "barrier" => {
                let kind = match w[1] {
                    "down_in" => Barrier::DownIn,
                    "down_out" => Barrier::DownOut,
                    "up_in" => Barrier::UpIn,
                    "up_out" => Barrier::UpOut,
                    _ => panic!(),
                };
                barrier_price(
                    kind,
                    cp(w[2]),
                    f(3),
                    f(4),
                    f(5),
                    f(6),
                    f(7),
                    f(8),
                    f(9),
                    f(10),
                )
            }
            "touch" => {
                let kind = match w[1] {
                    "one_touch_down" => Touch::OneTouchDown,
                    "one_touch_up" => Touch::OneTouchUp,
                    "no_touch_down" => Touch::NoTouchDown,
                    "no_touch_up" => Touch::NoTouchUp,
                    _ => panic!(),
                };
                touch_price(kind, f(2), f(3), f(4), f(5), f(6), f(7), f(8), w[9] == "1")
            }
            "dko" => {
                double_knockout_price(cp(w[1]), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), 10)
            }
            "dnt" => double_no_touch(f(1), f(2), f(3), f(4), f(5), f(6), f(7), 1.0, 50),
            "geo" => geometric_asian_price(cp(w[1]), f(2), f(3), &fix, f(4), f(5), f(6), f(7)),
            "tw" => turnbull_wakeman_price(cp(w[1]), f(2), f(3), &fix, f(4), f(5), f(6), f(7)),
            "twc" => {
                turnbull_wakeman_continuous(cp(w[1]), f(2), f(3), f(4), f(5), f(6), f(7), f(8))
            }
            "essvi" => essvi_total_variance(f(1), f(2), f(3), f(4)),
            "svi" => svi_total_variance(f(1), f(2), f(3), f(4), f(5), f(6)),
            other => panic!("unknown golden kind {other}"),
        };
        close(got, want, line);
        count += 1;
    }
    assert!(count > 150, "only {count} golden lines");
}

#[test]
fn in_out_parity_and_haug_point() {
    // Haug (2007) Table 4-13: down-and-out call X=90, H=95, rebate 3, sigma 0.25 = 9.0246.
    let p = barrier_price(
        Barrier::DownOut,
        Cp::Call,
        100.0,
        90.0,
        95.0,
        0.5,
        0.08,
        0.04,
        0.25,
        3.0,
    );
    assert!((p - 9.0246).abs() < 1e-4);
    let a = barrier_price(
        Barrier::UpIn,
        Cp::Put,
        100.0,
        100.0,
        105.0,
        0.5,
        0.08,
        0.04,
        0.3,
        0.0,
    );
    let b = barrier_price(
        Barrier::UpOut,
        Cp::Put,
        100.0,
        100.0,
        105.0,
        0.5,
        0.08,
        0.04,
        0.3,
        0.0,
    );
    let v = vanilla_price(Cp::Put, 100.0, 100.0, 0.5, 0.08, 0.04, 0.3);
    assert!((a + b - v).abs() < 1e-12);
}
