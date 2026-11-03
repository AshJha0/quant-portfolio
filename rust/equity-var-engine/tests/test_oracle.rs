//! W5: the Rust equity engine against the independent oracle.
//!
//! Expected values come from `tests/oracle/mod.rs`, generated from
//! `python/equity/03-var-es-engine/tests/golden/var_oracle.py` (mpmath at
//! 50 digits and exact rationals; it imports nothing from `eq_var` or
//! `fx_var`) by `emit_oracle_consts.py` in the same directory. Closed forms
//! must agree to 1e-10 relative; the historical VaR order statistic
//! exactly; Monte Carlo VaR of a Gaussian book within 3 standard errors.

mod oracle;

use eq_var_engine::prelude::*;
use oracle::*;

const REL: f64 = 1e-10;
const SIG: f64 = 12345.678;
const MEAN: f64 = -37.5;

fn close(actual: f64, expected: f64, what: &str) {
    close_rel(actual, expected, what, REL);
}

fn close_rel(actual: f64, expected: f64, what: &str, rel: f64) {
    assert!(actual.is_finite(), "{what}: not finite");
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{what}: engine {actual:e} oracle {expected:e} rel {:e}",
        (actual / expected - 1.0).abs()
    );
}

fn conv_sample() -> Vec<f64> {
    (0..250)
        .map(|i| {
            let t = f64::from(i);
            let x = 1000.0 * (0.37 * t + 0.1).sin() * (1.0 + 0.5 * (0.11 * t).cos());
            (x * 100.0 + 0.5).floor() / 100.0
        })
        .collect()
}

fn case_b() -> (Vec<f64>, Matrix) {
    let v = [0.01, 0.015, 0.02];
    let c = [[1.0, 0.5, 0.25], [0.5, 1.0, 0.3], [0.25, 0.3, 1.0]];
    let mut cov = Matrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            cov.set(i, j, v[i] * c[i][j] * v[j]);
        }
    }
    (vec![1.0e6, -5.0e5, 2.0e5], cov)
}

macro_rules! t_rows {
    ($($a:expr, $df:expr, $vu:ident, $eu:ident, $vo:ident, $eo:ident;)*) => {
        [$(($a, $df, $vu, $eu, $vo, $eo)),*]
    };
}

#[test]
fn normal_and_student_t_var_es() {
    let one = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let rows = [
        (
            0.001,
            NORMAL_VAR_UNIT_A0001,
            NORMAL_ES_UNIT_A0001,
            NORMAL_VAR_ODD_A0001,
            NORMAL_ES_ODD_A0001,
        ),
        (
            0.01,
            NORMAL_VAR_UNIT_A001,
            NORMAL_ES_UNIT_A001,
            NORMAL_VAR_ODD_A001,
            NORMAL_ES_ODD_A001,
        ),
        (
            0.025,
            NORMAL_VAR_UNIT_A0025,
            NORMAL_ES_UNIT_A0025,
            NORMAL_VAR_ODD_A0025,
            NORMAL_ES_ODD_A0025,
        ),
        (
            0.05,
            NORMAL_VAR_UNIT_A005,
            NORMAL_ES_UNIT_A005,
            NORMAL_VAR_ODD_A005,
            NORMAL_ES_ODD_A005,
        ),
    ];
    for (a, vu, eu, vo, eo) in rows {
        close(
            parametric_var(&[1.0], &one, a, TailModel::Normal).unwrap(),
            vu,
            "N VaR unit",
        );
        close(
            parametric_var_full(&[SIG], &one, a, TailModel::Normal, MEAN, 1).unwrap(),
            vo,
            "N VaR odd",
        );
        close(normal_es(1.0, a, 0.0).unwrap(), eu, "N ES unit");
        close(normal_es(SIG, a, MEAN).unwrap(), eo, "N ES odd");
    }
    let t = t_rows! {
        0.001, 3.0, T3_VAR_UNIT_A0001, T3_ES_UNIT_A0001, T3_VAR_ODD_A0001, T3_ES_ODD_A0001;
        0.001, 4.0, T4_VAR_UNIT_A0001, T4_ES_UNIT_A0001, T4_VAR_ODD_A0001, T4_ES_ODD_A0001;
        0.001, 6.0, T6_VAR_UNIT_A0001, T6_ES_UNIT_A0001, T6_VAR_ODD_A0001, T6_ES_ODD_A0001;
        0.001, 10.0, T10_VAR_UNIT_A0001, T10_ES_UNIT_A0001, T10_VAR_ODD_A0001, T10_ES_ODD_A0001;
        0.001, 30.0, T30_VAR_UNIT_A0001, T30_ES_UNIT_A0001, T30_VAR_ODD_A0001, T30_ES_ODD_A0001;
        0.01, 3.0, T3_VAR_UNIT_A001, T3_ES_UNIT_A001, T3_VAR_ODD_A001, T3_ES_ODD_A001;
        0.01, 4.0, T4_VAR_UNIT_A001, T4_ES_UNIT_A001, T4_VAR_ODD_A001, T4_ES_ODD_A001;
        0.01, 6.0, T6_VAR_UNIT_A001, T6_ES_UNIT_A001, T6_VAR_ODD_A001, T6_ES_ODD_A001;
        0.01, 10.0, T10_VAR_UNIT_A001, T10_ES_UNIT_A001, T10_VAR_ODD_A001, T10_ES_ODD_A001;
        0.01, 30.0, T30_VAR_UNIT_A001, T30_ES_UNIT_A001, T30_VAR_ODD_A001, T30_ES_ODD_A001;
        0.025, 3.0, T3_VAR_UNIT_A0025, T3_ES_UNIT_A0025, T3_VAR_ODD_A0025, T3_ES_ODD_A0025;
        0.025, 4.0, T4_VAR_UNIT_A0025, T4_ES_UNIT_A0025, T4_VAR_ODD_A0025, T4_ES_ODD_A0025;
        0.025, 6.0, T6_VAR_UNIT_A0025, T6_ES_UNIT_A0025, T6_VAR_ODD_A0025, T6_ES_ODD_A0025;
        0.025, 10.0, T10_VAR_UNIT_A0025, T10_ES_UNIT_A0025, T10_VAR_ODD_A0025, T10_ES_ODD_A0025;
        0.025, 30.0, T30_VAR_UNIT_A0025, T30_ES_UNIT_A0025, T30_VAR_ODD_A0025, T30_ES_ODD_A0025;
        0.05, 3.0, T3_VAR_UNIT_A005, T3_ES_UNIT_A005, T3_VAR_ODD_A005, T3_ES_ODD_A005;
        0.05, 4.0, T4_VAR_UNIT_A005, T4_ES_UNIT_A005, T4_VAR_ODD_A005, T4_ES_ODD_A005;
        0.05, 6.0, T6_VAR_UNIT_A005, T6_ES_UNIT_A005, T6_VAR_ODD_A005, T6_ES_ODD_A005;
        0.05, 10.0, T10_VAR_UNIT_A005, T10_ES_UNIT_A005, T10_VAR_ODD_A005, T10_ES_ODD_A005;
        0.05, 30.0, T30_VAR_UNIT_A005, T30_ES_UNIT_A005, T30_VAR_ODD_A005, T30_ES_ODD_A005;
    };
    for (a, df, vu, eu, vo, eo) in t {
        let tm = TailModel::StudentT { df };
        let tag = format!("a={a} df={df}");
        close(
            parametric_var(&[1.0], &one, a, tm).unwrap(),
            vu,
            &format!("t VaR unit {tag}"),
        );
        close(
            parametric_var_full(&[SIG], &one, a, tm, MEAN, 1).unwrap(),
            vo,
            &format!("t VaR odd {tag}"),
        );
        close(
            student_t_es(1.0, a, df, 0.0).unwrap(),
            eu,
            &format!("t ES unit {tag}"),
        );
        close(
            student_t_es(SIG, a, df, MEAN).unwrap(),
            eo,
            &format!("t ES odd {tag}"),
        );
    }
}

#[test]
fn cornish_fisher() {
    let rows = [
        (
            -0.5,
            1.0,
            0.01,
            CF_SM05_K1_A001_VALID,
            CF_SM05_K1_A001_VAR_ODD,
        ),
        (
            0.122,
            -0.427,
            0.01,
            CF_S0122_KM0427_A001_VALID,
            CF_S0122_KM0427_A001_VAR_ODD,
        ),
        (
            0.2,
            0.5,
            0.025,
            CF_S02_K05_A0025_VALID,
            CF_S02_K05_A0025_VAR_ODD,
        ),
        (
            -0.3,
            2.0,
            0.05,
            CF_SM03_K2_A005_VALID,
            CF_SM03_K2_A005_VAR_ODD,
        ),
        (0.0, 0.0, 0.01, CF_S0_K0_A001_VALID, CF_S0_K0_A001_VAR_ODD),
    ];
    for (s, k, a, valid, var) in rows {
        assert_eq!(cornish_fisher_valid(s, k, a).unwrap(), valid == 1.0);
        close(
            cornish_fisher_var(SIG, a, s, k, MEAN, true).unwrap(),
            var,
            "CF",
        );
    }
}

#[test]
fn case_b_parametric() {
    let (w, cov) = case_b();
    let sigma = portfolio_sigma(&w, &cov).unwrap();
    close(sigma, B_SIGMA, "B sigma");
    let t5 = TailModel::StudentT { df: 5.0 };
    let rows = [
        (
            0.01,
            B_NORMAL_VAR_A001,
            B_NORMAL_VAR_10D_A001,
            B_NORMAL_ES_A001,
            B_T5_VAR_A001,
            B_T5_ES_A001,
        ),
        (
            0.025,
            B_NORMAL_VAR_A0025,
            B_NORMAL_VAR_10D_A0025,
            B_NORMAL_ES_A0025,
            B_T5_VAR_A0025,
            B_T5_ES_A0025,
        ),
    ];
    for (a, var, var10, es, tv, te) in rows {
        close(
            parametric_var(&w, &cov, a, TailModel::Normal).unwrap(),
            var,
            "B N VaR",
        );
        close(
            parametric_var_full(&w, &cov, a, TailModel::Normal, 0.0, 10).unwrap(),
            var10,
            "B N VaR 10d",
        );
        close(normal_es(sigma, a, 0.0).unwrap(), es, "B N ES");
        close(parametric_var(&w, &cov, a, t5).unwrap(), tv, "B t5 VaR");
        close(student_t_es(sigma, a, 5.0, 0.0).unwrap(), te, "B t5 ES");
    }
}

#[test]
fn historical_kth_worst_exact() {
    let p = conv_sample();
    let rows = [
        (0.004, HIST_KTH_VAR_A0004, HIST_KTH_ES_A0004),
        (0.01, HIST_KTH_VAR_A001, HIST_KTH_ES_A001),
        (0.025, HIST_KTH_VAR_A0025, HIST_KTH_ES_A0025),
        (0.05, HIST_KTH_VAR_A005, HIST_KTH_ES_A005),
        (0.1, HIST_KTH_VAR_A01, HIST_KTH_ES_A01),
    ];
    for (a, var, es) in rows {
        let (v, e) = empirical_var_es(&p, a).unwrap();
        assert_eq!(v.to_bits(), var.to_bits(), "VaR {a}");
        assert_eq!(
            historical_var(&p, a).unwrap().to_bits(),
            var.to_bits(),
            "hist {a}"
        );
        close_rel(e, es, &format!("ES {a}"), 1e-12);
    }
    assert_eq!(HIST_KTH_RANK_A001, 3.0); // MAR99: the 3rd worst of 250
}

macro_rules! rows4 {
    ($($n:expr, $x:expr, $a:expr, $lr:ident, $p:ident;)*) => { [$(($n, $x, $a, $lr, $p)),*] };
}
macro_rules! rows3 {
    ($($n:expr, $x:expr, $a:expr, $c:ident;)*) => { [$(($n, $x, $a, $c)),*] };
}

#[test]
fn kupiec_christoffersen_basel() {
    let ks = rows4! {
        250, 0, 0.01, KUPIEC_LR_N250_X0_A001, KUPIEC_P_N250_X0_A001;
        250, 1, 0.01, KUPIEC_LR_N250_X1_A001, KUPIEC_P_N250_X1_A001;
        250, 2, 0.01, KUPIEC_LR_N250_X2_A001, KUPIEC_P_N250_X2_A001;
        250, 3, 0.01, KUPIEC_LR_N250_X3_A001, KUPIEC_P_N250_X3_A001;
        250, 4, 0.01, KUPIEC_LR_N250_X4_A001, KUPIEC_P_N250_X4_A001;
        250, 5, 0.01, KUPIEC_LR_N250_X5_A001, KUPIEC_P_N250_X5_A001;
        250, 6, 0.01, KUPIEC_LR_N250_X6_A001, KUPIEC_P_N250_X6_A001;
        250, 7, 0.01, KUPIEC_LR_N250_X7_A001, KUPIEC_P_N250_X7_A001;
        250, 8, 0.01, KUPIEC_LR_N250_X8_A001, KUPIEC_P_N250_X8_A001;
        250, 9, 0.01, KUPIEC_LR_N250_X9_A001, KUPIEC_P_N250_X9_A001;
        250, 10, 0.01, KUPIEC_LR_N250_X10_A001, KUPIEC_P_N250_X10_A001;
        250, 11, 0.01, KUPIEC_LR_N250_X11_A001, KUPIEC_P_N250_X11_A001;
        250, 12, 0.01, KUPIEC_LR_N250_X12_A001, KUPIEC_P_N250_X12_A001;
        500, 9, 0.01, KUPIEC_LR_N500_X9_A001, KUPIEC_P_N500_X9_A001;
        1000, 15, 0.01, KUPIEC_LR_N1000_X15_A001, KUPIEC_P_N1000_X15_A001;
        250, 12, 0.025, KUPIEC_LR_N250_X12_A0025, KUPIEC_P_N250_X12_A0025;
        250, 5, 0.05, KUPIEC_LR_N250_X5_A005, KUPIEC_P_N250_X5_A005;
        1000, 50, 0.05, KUPIEC_LR_N1000_X50_A005, KUPIEC_P_N1000_X50_A005;
    };
    for (n, x, a, lr, p) in ks {
        let r = kupiec_pof(&KupiecInput {
            n_obs: n,
            n_exceptions: x,
            alpha: a,
        })
        .unwrap();
        if lr < 1e-300 {
            assert!(r.lr.abs() < 1e-12, "{n}/{x}");
        } else {
            close(r.lr, lr, &format!("kupiec LR {n}/{x}"));
        }
        close(r.pvalue, p, &format!("kupiec p {n}/{x}"));
    }
    let pats = [
        (
            40,
            0,
            CHR_EQ_IND_LR,
            CHR_EQ_IND_P,
            CHR_EQ_CC_LR,
            CHR_EQ_CC_P,
        ),
        (
            37,
            5,
            CHR_FX_IND_LR,
            CHR_FX_IND_P,
            CHR_FX_CC_LR,
            CHR_FX_CC_P,
        ),
    ];
    for (m, rem, il, ip, cl, cp) in pats {
        let ex: Vec<u8> = (0..250)
            .map(|t| u8::from(t % m == rem || t == 100 || t == 101))
            .collect();
        let ind = christoffersen_independence(&ex).unwrap();
        let cc = christoffersen_cc(&ex, 0.01).unwrap();
        close(ind.lr, il, "chr ind LR");
        close(ind.pvalue, ip, "chr ind p");
        close(cc.lr, cl, "chr cc LR");
        close(cc.pvalue, cp, "chr cc p");
    }
    let bs = rows3! {
        250, 0, 0.01, BASEL_CDF_N250_X0_A001; 250, 1, 0.01, BASEL_CDF_N250_X1_A001;
        250, 2, 0.01, BASEL_CDF_N250_X2_A001; 250, 3, 0.01, BASEL_CDF_N250_X3_A001;
        250, 4, 0.01, BASEL_CDF_N250_X4_A001; 250, 5, 0.01, BASEL_CDF_N250_X5_A001;
        250, 6, 0.01, BASEL_CDF_N250_X6_A001; 250, 7, 0.01, BASEL_CDF_N250_X7_A001;
        250, 8, 0.01, BASEL_CDF_N250_X8_A001; 250, 9, 0.01, BASEL_CDF_N250_X9_A001;
        250, 10, 0.01, BASEL_CDF_N250_X10_A001; 500, 8, 0.01, BASEL_CDF_N500_X8_A001;
        500, 12, 0.01, BASEL_CDF_N500_X12_A001; 1000, 15, 0.01, BASEL_CDF_N1000_X15_A001;
        250, 12, 0.025, BASEL_CDF_N250_X12_A0025;
    };
    for (n, x, a, c) in bs {
        let r = basel_traffic_light(&BaselInput {
            n_obs: n,
            n_exceptions: x,
            alpha: a,
        })
        .unwrap();
        close(r.cumulative_prob, c, &format!("basel {n}/{x}"));
    }
}

/// Monte Carlo VaR of the Gaussian case-B book converges to the exact
/// quantile: |MC - exact| <= 3 SE (bootstrap SE of the same run).
#[test]
fn monte_carlo_gaussian_book_within_3_se() {
    let (w, cov) = case_b();
    for seed in 1..=3u64 {
        for (tail, exact) in [
            (TailModel::Normal, B_NORMAL_VAR_A001),
            (TailModel::StudentT { df: 5.0 }, B_T5_VAR_A001),
        ] {
            let r = monte_carlo_run(&w, &cov, 0.01, 200_000, tail, seed, 200).unwrap();
            assert!(
                (r.var - exact).abs() <= 3.0 * r.se,
                "seed {seed} {tail:?}: mc {} exact {exact} se {}",
                r.var,
                r.se
            );
        }
    }
}
