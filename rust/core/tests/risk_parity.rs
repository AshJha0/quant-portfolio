//! Parity of the Rust risk kernels with the Python reference values in
//! `data/risk_parity.txt` (written by
//! `python/equity/04-fixed-income/tools/gen_risk_parity.py`). Bound: 1e-12
//! relative to the largest entry of each output, the repository's
//! cross-language parity bound.

use qp_core::risk::{
    Conversion, GAMMA_REL_STEP, Smile, SmileDynamics, VolGrid, explain, node_to_quote_risk,
    par_dv01, premium_adjusted_delta, quote_gamma, rank_smile_dynamics, reporting_conversion,
    smile_delta, smile_gamma, smile_price, taylor_attribution, tenor_weights, vega_grid,
};

fn floats(p: &[&str]) -> Vec<f64> {
    p.iter().map(|s| s.parse().unwrap()).collect()
}

fn close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * scale,
            "{what}[{k}]: {g} vs {w} (scale {scale})"
        );
    }
}

/// [`close`] entry by entry: scalars and outputs of unrelated scale.
fn close_each(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (g, w) in got.iter().zip(want) {
        close(&[*g], &[*w], what);
    }
}

#[test]
fn matches_python() {
    let text = include_str!("data/risk_parity.txt");
    let rows: Vec<Vec<&str>> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split_whitespace().collect())
        .collect();
    let mut i = 0;
    let (mut n_curve, mut n_grid, mut n_ta) = (0, 0, 0);
    let (mut n_sd, mut n_rank, mut n_pa, mut n_rc, mut n_tw) = (0, 0, 0, 0, 0);
    while i < rows.len() {
        let r = &rows[i];
        match r[0] {
            "N" => {
                let n: usize = r[1].parse().unwrap();
                let jac = floats(&rows[i + 1][1..]);
                let gz = floats(&rows[i + 2][1..]);
                let gamz = floats(&rows[i + 3][1..]);
                let hs: Vec<f64> = (0..n).flat_map(|k| floats(&rows[i + 4 + k][1..])).collect();
                let base = i + 4 + n;
                let (gq_w, dv_w, gam_w) = (
                    floats(&rows[base][1..]),
                    floats(&rows[base + 1][1..]),
                    floats(&rows[base + 2][1..]),
                );
                let gq = node_to_quote_risk(&jac, n, &gz).unwrap();
                close(&gq, &gq_w, "quote risk");
                close(&par_dv01(&gq), &dv_w, "par DV01");
                // Use the Python g_q so the gamma check isolates the gamma algebra.
                close(
                    &quote_gamma(&jac, n, &gamz, &gq_w, &hs).unwrap(),
                    &gam_w,
                    "quote gamma",
                );
                // The shortcut without the curvature term is far off: keep it so.
                let zero = vec![0.0; n];
                let shortcut = quote_gamma(&jac, n, &gamz, &zero, &hs).unwrap();
                let worst = shortcut
                    .iter()
                    .zip(&gam_w)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                let scale = gam_w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                assert!(
                    worst > 0.01 * scale,
                    "shortcut unexpectedly close: {worst} vs {scale}"
                );
                n_curve += 1;
                i = base + 3;
            }
            "GRID" => {
                let d: usize = r[1].parse().unwrap();
                let axes: Vec<Vec<f64>> = (0..d).map(|k| floats(&rows[i + 1 + k][1..])).collect();
                let vols = floats(&rows[i + 1 + d][1..]);
                let grid = VolGrid::new(axes, vols).unwrap();
                i += 2 + d;
                let mut pts: Vec<(Vec<f64>, f64)> = Vec::new();
                while rows[i][0] == "PT" {
                    let p = floats(&rows[i][1..]);
                    let (c, n) = (p[..d].to_vec(), p[d]);
                    close(
                        &grid.weights(&c).unwrap(),
                        &floats(&rows[i + 1][1..]),
                        "weights",
                    );
                    close(&[grid.vol(&c).unwrap()], &floats(&rows[i + 2][1..]), "vol");
                    pts.push((c, n));
                    i += 3;
                }
                assert_eq!(rows[i][0], "VEGA");
                let pv = |g: &VolGrid| {
                    pts.iter()
                        .map(|(c, n)| n * g.vol(c).unwrap().powi(2))
                        .sum::<f64>()
                };
                close(
                    &vega_grid(pv, &grid, 1e-4).unwrap(),
                    &floats(&rows[i][1..]),
                    "vega",
                );
                n_grid += 1;
                i += 1;
            }
            "TA" => {
                let n: usize = r[1].parse().unwrap();
                let with_g = r[2] == "1";
                let d = floats(&rows[i + 1][1..]);
                let m = floats(&rows[i + 2][1..]);
                let mut k = i + 3;
                let g = if with_g {
                    k += 1;
                    Some(floats(&rows[k - 1][1..]))
                } else {
                    None
                };
                assert_eq!(d.len(), n);
                close(
                    &taylor_attribution(&d, &m, g.as_deref()).unwrap(),
                    &floats(&rows[k][1..]),
                    "attribution",
                );
                let ex_in = floats(&rows[k + 1][1..]);
                let time = floats(&rows[k + 2][1..]);
                let want = floats(&rows[k + 3][1..]);
                let ex = explain(ex_in[0], &time, &d, &m, g.as_deref(), ex_in[1]).unwrap();
                close(&[ex.rtpl(), ex.unexplained(), ex.total()], &want, "explain");
                n_ta += 1;
                i = k + 4;
            }
            "SM" => {
                let p = floats(&r[1..]);
                let (k0, a, b, c) = (p[4], p[5], p[6], p[7]);
                let vol0 = move |k: f64| {
                    let x = k - k0;
                    a + b * x + c * x * x
                };
                let smile = Smile::new(p[0], p[1], p[2], p[3], vol0).unwrap();
                i += 1;
                while rows[i][0] == "SD" {
                    let v = floats(&rows[i][1..]);
                    let (k, s, call) = (v[0], v[1], v[2] == 1.0);
                    let d = SmileDynamics::ALL[v[3] as usize];
                    let got = [
                        smile_price(&smile, k, s, call, d).unwrap(),
                        smile_delta(&smile, k, s, call, d).unwrap(),
                    ];
                    close_each(&got, &v[4..6], "smile price and delta");
                    // Gamma differences two deltas 2h apart, so it gets the
                    // 1e-12 of those deltas, not 1e-12 of itself (see the
                    // generator's docstring).
                    let h = GAMMA_REL_STEP * s;
                    close_each(
                        &[
                            smile_delta(&smile, k, s + h, call, d).unwrap(),
                            smile_delta(&smile, k, s - h, call, d).unwrap(),
                        ],
                        &v[7..9],
                        "bumped smile delta",
                    );
                    let gamma = smile_gamma(&smile, k, s, call, d, GAMMA_REL_STEP).unwrap();
                    let bound = 1e-12 * v[7].abs().max(v[8].abs()) / h;
                    assert!(
                        (gamma - v[6]).abs() <= bound,
                        "smile gamma: {gamma} vs {} (bound {bound})",
                        v[6]
                    );
                    n_sd += 1;
                    i += 1;
                }
                assert_eq!(rows[i][0], "RK");
                let h = floats(&rows[i][1..]);
                let (sp, pr, tt) = (
                    floats(&rows[i + 1][1..]),
                    floats(&rows[i + 2][1..]),
                    floats(&rows[i + 3][1..]),
                );
                let got = rank_smile_dynamics(&sp, &pr, &tt, &smile, h[0], h[1] == 1.0).unwrap();
                close_each(&got, &floats(&rows[i + 4][1..]), "rank");
                n_rank += 1;
                i += 5;
            }
            "PA" => {
                let v = floats(&r[1..]);
                let got = premium_adjusted_delta(v[0], v[1], v[2], v[3], v[4], v[5], v[6] == 1.0)
                    .unwrap();
                close_each(&[got], &v[7..8], "premium-adjusted delta");
                n_pa += 1;
                i += 1;
            }
            "RC" => {
                let h = floats(&r[1..]);
                let conv = if h[0] == 1.0 {
                    Conversion::Spot
                } else {
                    Conversion::Scenario
                };
                let (v1, x1) = (floats(&rows[i + 1][1..]), floats(&rows[i + 2][1..]));
                close(
                    &reporting_conversion(h[1], &v1, h[2], &x1, conv).unwrap(),
                    &floats(&rows[i + 3][1..]),
                    "reporting P&L",
                );
                n_rc += 1;
                i += 4;
            }
            "TB" => {
                let b = floats(&r[1..]);
                i += 1;
                while i < rows.len() && rows[i][0] == "TW" {
                    let v = floats(&rows[i][1..]);
                    close(&tenor_weights(v[0], &b).unwrap(), &v[1..], "tenor weights");
                    n_tw += 1;
                    i += 1;
                }
            }
            other => panic!("unexpected tag {other}"),
        }
    }
    assert_eq!((n_curve, n_grid, n_ta), (1, 2, 3));
    assert_eq!((n_sd, n_rank, n_pa, n_rc, n_tw), (96, 3, 6, 4, 8));
}
